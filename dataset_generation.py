import asyncio
import base64
import json
import re
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from datasets import load_dataset
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from paddleocr import DocImgOrientationClassification
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
from pydantic import BaseModel

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
DATASET_NAME = "tech4humans/signature-detection"
SPLITS_TO_PROCESS = ["train", "validation", "test"]  # Or None for all splits

MAX_SAMPLES_TO_RUN: int | None = 1
CONCURRENCY_LIMIT = 8
ORI_BATCH_SIZE = 16
ORI_CONFIDENCE_THRESHOLD = 0.45
TARGET_MIN_DIM = 1600

# Per-Task Question Caps
MAX_POSITIVE_GROUNDING = 4
MAX_HARD_NEGATIVE_GROUNDING = 2
MAX_OPEN_DISCOVERY = 1
MAX_PURE_VQA = 2

# vLLM OpenAI-Compatible Endpoint
VLLM_BASE_URL = "http://192.9.200.28:7416/v1"
VLLM_API_KEY = "EMPTY"
TEACHER_MODEL = "LLM"

# Output Directories
BASE_OUT_DIR = Path("teacher_annotations")
JSON_OUT_DIR = BASE_OUT_DIR / "json"
CLEAN_IMG_DIR = BASE_OUT_DIR / "clean_images"
SOM_DEBUG_DIR = BASE_OUT_DIR / "som_debug"

for d in (JSON_OUT_DIR, CLEAN_IMG_DIR, SOM_DEBUG_DIR):
  d.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 2. MINIMAL STRUCTURED OUTPUT SCHEMA (vLLM Grammar Enforcement Only)
# ==============================================================================
class SignatureMark(BaseModel):
  id: int
  label: str


class SignatureGroundingQA(BaseModel):
  queried_role: str
  question: str
  answer_text: str
  target_ids: list[int]


class OpenDiscoveryQA(BaseModel):
  question: str
  answer_text: str
  target_ids: list[int]


class PureVQA(BaseModel):
  question: str
  answer_text: str


class TeacherAnnotationOutput(BaseModel):
  marks: list[SignatureMark]
  positive_grounding: list[SignatureGroundingQA]
  hard_negative_grounding: list[SignatureGroundingQA]
  open_discovery: list[OpenDiscoveryQA]
  pure_vqa: list[PureVQA]


# ==============================================================================
# 3. ORIENTATION CORRECTION (Image + COCO BBoxes)
# ==============================================================================
def rotate_coco_bboxes(
    bboxes: list[list[float]], width: int, height: int, ccw_angle: int
) -> tuple[list[list[float]], int, int]:
  if not bboxes or ccw_angle not in (90, 180, 270):
    new_w, new_h = (height, width) if ccw_angle in (90, 270) else (width, height)
    return bboxes, new_w, new_h

  rotated = []
  for x, y, w, h in bboxes:
    if ccw_angle == 90:
      nx = y
      ny = width - (x + w)
      nw, nh = h, w
    elif ccw_angle == 180:
      nx = width - (x + w)
      ny = height - (y + h)
      nw, nh = w, h
    elif ccw_angle == 270:
      nx = height - (y + h)
      ny = x
      nw, nh = h, w
    rotated.append([nx, ny, nw, nh])

  new_w, new_h = (height, width) if ccw_angle in (90, 270) else (width, height)
  return rotated, new_w, new_h


def batch_correct_orientation(
    items: list[dict],
    ori_model: DocImgOrientationClassification,
    batch_size: int = ORI_BATCH_SIZE,
    conf_threshold: float = ORI_CONFIDENCE_THRESHOLD,
) -> list[dict]:
  np_images = [
      np.array(ImageOps.exif_transpose(row["image"]).convert("RGB"))
      for row in items
  ]

  for start in range(0, len(items), batch_size):
    batch_imgs = np_images[start : start + batch_size]
    results = ori_model.predict(batch_imgs, batch_size=len(batch_imgs))

    for offset, res in enumerate(results):
      idx = start + offset
      res_dict = res.get("res", res)
      angle = int(res_dict["label_names"][0])
      score = float(res_dict["scores"][0])

      applied_angle = (
          angle if (angle in (90, 180, 270) and score >= conf_threshold) else 0
      )
      pil_img = Image.fromarray(batch_imgs[offset])
      orig_w, orig_h = pil_img.size
      raw_bboxes = items[idx]["objects"]["bbox"]

      if applied_angle != 0:
        pil_img = pil_img.rotate(applied_angle, expand=True)
        rot_bboxes, new_w, new_h = rotate_coco_bboxes(
            raw_bboxes, orig_w, orig_h, applied_angle
        )
      else:
        rot_bboxes, new_w, new_h = raw_bboxes, orig_w, orig_h

      items[idx]["corrected_image"] = pil_img
      items[idx]["corrected_bboxes"] = rot_bboxes
      items[idx]["corrected_size"] = (new_w, new_h)
      items[idx]["applied_rotation"] = applied_angle

  return items


# ==============================================================================
# 4. IMAGE ENHANCEMENT & SET-OF-MARK (SoM) VISUALIZATION
# ==============================================================================
def preprocess_document_image(
    img: Image.Image, target_min_dim: int = TARGET_MIN_DIM
) -> Image.Image:
  w, h = img.size
  min_dim = min(w, h)
  if min_dim < target_min_dim:
    scale = target_min_dim / min_dim
    img = img.resize(
        (round(w * scale), round(h * scale)), Image.Resampling.LANCZOS
    )

  img = img.filter(ImageFilter.UnsharpMask(radius=1.4, percent=150, threshold=3))
  img = ImageEnhance.Contrast(img).enhance(1.15)
  return img


def coco_to_norm_xyxy(
    bbox: list[float], width: int, height: int
) -> list[int]:
  x, y, w, h = bbox
  return [
      max(0, min(1000, round((x / width) * 1000))),
      max(0, min(1000, round((y / height) * 1000))),
      max(0, min(1000, round(((x + w) / width) * 1000))),
      max(0, min(1000, round(((y + h) / height) * 1000))),
  ]


def draw_som_marks(
    enhanced_img: Image.Image, norm_bboxes: list[list[int]]
) -> Image.Image:
  marked = enhanced_img.copy()
  if not norm_bboxes:
    return marked

  draw = ImageDraw.Draw(marked)
  w, h = marked.size

  # Balanced badge font (~44px on 1600px images — readable without covering text)
  font_size = max(28, round(min(w, h) * 0.028))
  try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
  except OSError:
    try:
      font = ImageFont.truetype("arialbd.ttf", font_size)
    except OSError:
      font = ImageFont.load_default(size=font_size)

  box_color = (225, 25, 25)
  line_width = max(4, round(min(w, h) * 0.003))

  for idx, (nx1, ny1, nx2, ny2) in enumerate(norm_bboxes, start=1):
    x1 = round((nx1 / 1000) * w)
    y1 = round((ny1 / 1000) * h)
    x2 = round((nx2 / 1000) * w)
    y2 = round((ny2 / 1000) * h)

    draw.rectangle([x1, y1, x2, y2], outline=box_color, width=line_width)

    label_str = str(idx)
    text_bbox = draw.textbbox((0, 0), label_str, font=font)
    tw = text_bbox[2] - text_bbox[0]
    th = text_bbox[3] - text_bbox[1]
    pad_x = max(8, round(font_size * 0.22))
    pad_y = max(5, round(font_size * 0.15))

    badge_w = tw + pad_x * 2
    badge_h = th + pad_y * 2

    bx1 = max(0, min(w - badge_w, x1))
    by1 = (y1 - badge_h) if (y1 - badge_h >= 0) else y1
    bx2 = bx1 + badge_w
    by2 = by1 + badge_h

    # Solid crimson badge with a crisp white outline for high separation
    draw.rectangle(
        [bx1, by1, bx2, by2],
        fill=box_color,
        outline=(255, 255, 255),
        width=2,
    )
    draw.text(
        (bx1 + pad_x - text_bbox[0], by1 + pad_y - text_bbox[1]),
        label_str,
        fill=(255, 255, 255),
        font=font,
    )

  return marked


def pil_to_base64_jpeg(img: Image.Image, quality: int = 92) -> str:
  buf = BytesIO()
  img.save(buf, format="JPEG", quality=quality)
  return base64.b64encode(buf.getvalue()).decode("utf-8")


# ==============================================================================
# 5. EXPLICIT SYSTEM PROMPT (WITH CONCRETE JSON EXAMPLE) & POST-PROCESSING
# ==============================================================================
SYSTEM_PROMPT = f"""You are a document signature detection and visual grounding dataset annotator.
You will receive a document image where signature regions (if any) are enclosed in red boxes with numbered badges (1, 2, 3...). If the user states there are 0 marked regions, the document has no signatures.

Return a single JSON object with these 5 exact keys:

1. "marks":
   - For EVERY numbered red badge on the image, output {{"id": <int>, "label": "<Role or Title>"}}.
   - Read the entire document (headers, letterheads, printed lines below signatures, body text) to infer a concise 1-4 word Title Case role/title (e.g., "Branch Manager", "Director", "Secretary", "Media Planner", "Sender", "Applicant", "Committee Member").
   - Prefer an official role or title over a personal name whenever one applies.
   - NEVER use "unlabeled", "unknown", or "signature".
   - If there are 0 badges on the image, output "marks": [].

2. "positive_grounding" (0 to {MAX_POSITIVE_GROUNDING} items MAX; MUST be [] if 0 badges exist):
   - Create AT MOST 1 item per unique role present in "marks". Never duplicate a role or question.
   - Every item MUST ask about that role's SIGNATURE presence/location and MUST set "target_ids" to the matching badge ID integer(s) from "marks" (e.g., [1] or [2, 4]).
   - NEVER include negative ("No, ...") answers in "positive_grounding".

3. "hard_negative_grounding" (1 to {MAX_HARD_NEGATIVE_GROUNDING} items MAX):
   - Pick 1 to {MAX_HARD_NEGATIVE_GROUNDING} distinct, plausible document roles whose SIGNATURE is NOT present on this page.
   - Ask whether that role's SIGNATURE is present, confirm it is not present in "answer_text", and set "target_ids": [].

4. "open_discovery" (1 item MAX):
   - Ask to identify and locate all signatures present on the document.
   - Set "target_ids" to ALL badge IDs from "marks" (e.g., [1, 2, 3]), or [] if 0 badges exist.

5. "pure_vqa" (1 to {MAX_PURE_VQA} items MAX):
   - Ask 1 to {MAX_PURE_VQA} distinct questions strictly about the SIGNATURES (e.g., total signature count, whether the document is signed, or which roles signed).

CRITICAL TEXT RULES:
- EVERY question in ALL categories MUST be explicitly about SIGNATURES ("signature", "signatures", "signed"). Never ask general questions like "Who is the recipient?".
- Inside "question" and "answer_text", NEVER mention badge numbers, mark IDs, or red boxes (e.g., never write "mark 1" or "red box"). Only put badge ID integers inside "marks[].id" and "target_ids".

EXAMPLE OUTPUT FOR A DOCUMENT WITH 2 BADGES (1 = Secretary, 2 = Director):
{{
  "marks": [
    {{"id": 1, "label": "Secretary"}},
    {{"id": 2, "label": "Director"}}
  ],
  "positive_grounding": [
    {{
      "queried_role": "Secretary",
      "question": "Is the Secretary signature present on this document? If yes, provide its bounding box.",
      "answer_text": "Yes, the Secretary signature is present on this document.",
      "target_ids": [1]
    }},
    {{
      "queried_role": "Director",
      "question": "Is the Director signature present on this document? If yes, provide its bounding box.",
      "answer_text": "Yes, the Director signature is present on this document.",
      "target_ids": [2]
    }}
  ],
  "hard_negative_grounding": [
    {{
      "queried_role": "Branch Manager",
      "question": "Is the Branch Manager signature present on this document? If yes, provide its bounding box.",
      "answer_text": "No, the Branch Manager signature is not present on this document.",
      "target_ids": []
    }},
    {{
      "queried_role": "Witness",
      "question": "Is the Witness signature present on this document? If yes, provide its bounding box.",
      "answer_text": "No, the Witness signature is not present on this document.",
      "target_ids": []
    }}
  ],
  "open_discovery": [
    {{
      "question": "Identify and locate all signatures present on this document.",
      "answer_text": "There are two signatures present on this document, signed by the Secretary and the Director.",
      "target_ids": [1, 2]
    }}
  ],
  "pure_vqa": [
    {{
      "question": "How many signatures are present on this document?",
      "answer_text": "There are two signatures present on this document."
    }},
    {{
      "question": "Which roles have signed this document?",
      "answer_text": "The Secretary and the Director signed this document."
    }}
  ]
}}"""

MARK_LEAK_PATTERN = re.compile(
    r"\b(mark\s*(id|ids|number)?\s*\d+|red\s*box)\b", re.IGNORECASE
)


def has_mark_leakage(record: dict) -> bool:
  for section in (
      "positive_grounding",
      "hard_negative_grounding",
      "open_discovery",
      "pure_vqa",
  ):
    for item in record.get(section, []):
      if MARK_LEAK_PATTERN.search(
          item.get("question", "")
      ) or MARK_LEAK_PATTERN.search(item.get("answer_text", "")):
        return True
  return False


def _dedupe_qa_list(
    items: list[dict], check_role: bool = False
) -> list[dict]:
  seen_questions = set()
  seen_roles = set()
  unique_items = []
  for item in items:
    q_norm = re.sub(r"\s+", " ", item.get("question", "").strip().lower())
    role_norm = item.get("queried_role", "").strip().lower()

    if q_norm in seen_questions:
      continue
    if check_role and role_norm and role_norm in seen_roles:
      continue

    seen_questions.add(q_norm)
    if check_role and role_norm:
      seen_roles.add(role_norm)
    unique_items.append(item)
  return unique_items


def enforce_task_caps(record: dict, has_signatures: bool, num_gt_boxes: int) -> dict:
  valid_ids = set(range(1, num_gt_boxes + 1))

  if not has_signatures:
    record["marks"] = []
    record["positive_grounding"] = []
  else:
    marks = [
        m for m in record.get("marks", [])
        if isinstance(m.get("id"), int) and m["id"] in valid_ids
    ]
    record["marks"] = marks
    all_mark_ids = sorted(valid_ids)

    label_to_ids: dict[str, list[int]] = {}
    for m in marks:
      lbl = m.get("label", "").strip().lower()
      if lbl:
        label_to_ids.setdefault(lbl, []).append(m["id"])

    cleaned_pos = []
    for item in record.get("positive_grounding", []):
      ans_lower = item.get("answer_text", "").strip().lower()
      if ans_lower.startswith("no"):
        continue

      # Filter to valid IDs only
      t_ids = [i for i in item.get("target_ids", []) if i in valid_ids]

      # Recover target_ids from marks if model left it empty
      if not t_ids:
        q_role = item.get("queried_role", "").strip().lower()
        if q_role in label_to_ids:
          t_ids = label_to_ids[q_role]
        elif len(all_mark_ids) == 1:
          t_ids = all_mark_ids

      if t_ids:
        item["target_ids"] = sorted(set(t_ids))
        cleaned_pos.append(item)

    record["positive_grounding"] = _dedupe_qa_list(
        cleaned_pos, check_role=True
    )[:MAX_POSITIVE_GROUNDING]

  record["hard_negative_grounding"] = _dedupe_qa_list(
      record.get("hard_negative_grounding", []), check_role=True
  )[:MAX_HARD_NEGATIVE_GROUNDING]
  for item in record["hard_negative_grounding"]:
    item["target_ids"] = []

  record["open_discovery"] = _dedupe_qa_list(
      record.get("open_discovery", []), check_role=False
  )[:MAX_OPEN_DISCOVERY]
  all_ids = sorted(valid_ids) if has_signatures else []
  for item in record["open_discovery"]:
    t_ids = [i for i in item.get("target_ids", []) if i in valid_ids]
    item["target_ids"] = sorted(set(t_ids)) if t_ids else all_ids

  record["pure_vqa"] = _dedupe_qa_list(
      record.get("pure_vqa", []), check_role=False
  )[:MAX_PURE_VQA]
  return record


def build_langchain_message(item: dict) -> list:
  num_marks = len(item["norm_bboxes"])
  b64 = pil_to_base64_jpeg(item["marked_image"])

  if num_marks > 0:
    user_text = (
        f"This document has {num_marks} marked signature region(s) with red"
        f" badges numbered 1 to {num_marks}. Label every badge ID in 'marks',"
        " populate 'target_ids' with the exact matching badge ID integers for"
        " 'positive_grounding' and 'open_discovery', and follow the JSON"
        " format."
    )
  else:
    user_text = (
        "This document has 0 marked signature regions (unsigned document)."
        " Output 'marks': [], 'positive_grounding': [], and 'target_ids': []"
        " across all QA lists."
    )

  return [
      ("system", SYSTEM_PROMPT),
      HumanMessage(
          content=[
              {"type": "text", "text": user_text},
              {
                  "type": "image_url",
                  "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
              },
          ]
      ),
  ]


async def run_teacher_batch(processed_items: list[dict]) -> None:
  llm = ChatOpenAI(
      model=TEACHER_MODEL,
      base_url=VLLM_BASE_URL,
      api_key=VLLM_API_KEY,
      temperature=0.0,
      max_tokens=2048,
  )
  structured_llm = llm.with_structured_output(TeacherAnnotationOutput)

  messages_batch = [build_langchain_message(x) for x in processed_items]
  print(
      f"Sending {len(messages_batch)} items to vLLM via unified abatch "
      f"(max_concurrency={CONCURRENCY_LIMIT})..."
  )
  t0 = time.perf_counter()

  responses = await structured_llm.abatch(
      messages_batch,
      config={"max_concurrency": CONCURRENCY_LIMIT},
      return_exceptions=True,
  )

  saved_count = 0
  for item, resp in zip(processed_items, responses):
    split_name = item["split_name"]
    img_id = item["image_id"]
    sample_key = f"{split_name}_{img_id}"
    num_gt = len(item["norm_bboxes"])
    has_signatures = num_gt > 0

    if isinstance(resp, Exception):
      print(f"[ERROR] {sample_key} failed: {resp}")
      continue

    teacher_dict = enforce_task_caps(resp.model_dump(), has_signatures, num_gt)
    if has_mark_leakage(teacher_dict):
      print(
          f"[WARN] {sample_key} skipped due to mark ID leakage in QA text (will"
          " retry on next run)."
      )
      continue

    clean_rel = f"clean_images/{sample_key}.jpg"
    item["clean_image"].save(BASE_OUT_DIR / clean_rel, quality=95)

    if has_signatures:
      som_rel = f"som_debug/{sample_key}_som.jpg"
      item["marked_image"].save(BASE_OUT_DIR / som_rel, quality=85)

    id_to_bbox = {
        idx: bbox for idx, bbox in enumerate(item["norm_bboxes"], start=1)
    }

    final_record = {
        "split": split_name,
        "image_id": img_id,
        "sample_key": sample_key,
        "has_signatures": has_signatures,
        "image_path": clean_rel,
        "applied_rotation_ccw": item["applied_rotation"],
        "ground_truth_bboxes_norm_1000": id_to_bbox,
        **teacher_dict,
    }

    out_json_path = JSON_OUT_DIR / f"{sample_key}.json"
    tmp_path = out_json_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
      json.dump(final_record, f, indent=2, ensure_ascii=False)
    tmp_path.replace(out_json_path)
    saved_count += 1

  elapsed = time.perf_counter() - t0
  print(f"Saved {saved_count}/{len(processed_items)} JSON files in {elapsed:.2f}s.")


# ==============================================================================
# 6. MAIN ENTRYPOINT ACROSS ALL SPLITS WITH RESUME & N-LIMIT FILTERING
# ==============================================================================
def main():
  total_t0 = time.perf_counter()
  print(f"Loading all splits for '{DATASET_NAME}'...")
  ds_dict = load_dataset(DATASET_NAME)

  target_splits = (
      SPLITS_TO_PROCESS if SPLITS_TO_PROCESS is not None else list(ds_dict.keys())
  )
  print(f"Available splits: {list(ds_dict.keys())} | Target splits: {target_splits}")

  existing_keys = {p.stem for p in JSON_OUT_DIR.glob("*.json")}
  print(f"Found {len(existing_keys)} already processed samples in '{JSON_OUT_DIR}'.")

  pending_rows = []
  split_counts = {s: 0 for s in target_splits}

  for split_name in target_splits:
    if split_name not in ds_dict:
      print(f"[WARN] Split '{split_name}' not found in dataset, skipping.")
      continue

    for row in ds_dict[split_name]:
      sample_key = f"{split_name}_{row['image_id']}"
      if sample_key in existing_keys:
        continue

      row_data = dict(row)
      row_data["split_name"] = split_name
      pending_rows.append(row_data)
      split_counts[split_name] += 1

      if MAX_SAMPLES_TO_RUN is not None and len(pending_rows) >= MAX_SAMPLES_TO_RUN:
        break

    if MAX_SAMPLES_TO_RUN is not None and len(pending_rows) >= MAX_SAMPLES_TO_RUN:
      break

  if not pending_rows:
    print("Nothing new to process across any split. All requested samples are done!")
    return

  neg_queued = sum(1 for r in pending_rows if not r["objects"]["bbox"])
  print(
      f"Queued {len(pending_rows)} new samples ({len(pending_rows) - neg_queued}"
      f" positive, {neg_queued} zero-signature negative) -> breakdown:"
      f" {split_counts}\n"
  )

  # Step 1: Load orientation model & batch-rotate images + bounding boxes
  ori_model = DocImgOrientationClassification(
      model_name="PP-LCNet_x1_0_doc_ori",
      engine="onnxruntime",
  )
  pending_rows = batch_correct_orientation(pending_rows, ori_model)
  del ori_model

  # Step 2: Enhance image quality, normalize boxes, and draw SoM badges (if boxes exist)
  for item in pending_rows:
    rot_w, rot_h = item["corrected_size"]
    norm_bboxes = [
        coco_to_norm_xyxy(b, rot_w, rot_h) for b in item["corrected_bboxes"]
    ]
    clean_enhanced = preprocess_document_image(item["corrected_image"])
    marked_img = draw_som_marks(clean_enhanced, norm_bboxes)

    item["norm_bboxes"] = norm_bboxes
    item["clean_image"] = clean_enhanced
    item["marked_image"] = marked_img

  # Step 3: Run single unified LangChain async structured-output batch against vLLM
  asyncio.run(run_teacher_batch(pending_rows))
  print(f"\nTotal pipeline time: {time.perf_counter() - total_t0:.2f}s")


if __name__ == "__main__":
  main()