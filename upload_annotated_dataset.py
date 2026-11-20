"""Upload the teacher-annotated signature detection dataset to Hugging Face.

Every original column and every original split row from
``tech4humans/signature-detection`` is kept. Seven teacher-annotation columns
produced by ``dataset_generation.py`` are added on top:

  - ground_truth_bboxes_norm_1000
  - has_signatures
  - marks
  - positive_grounding
  - hard_negative_grounding
  - open_discovery
  - pure_vqa

Rows without a generated JSON get null / empty values in those columns, so the
splits still contain the exact same rows as the source dataset. Set
``ONLY_ANNOTATED = True`` below if you instead want to publish just the
annotated subset.

Authentication is read from `hf auth login`; no token is passed or stored here.

Run with no arguments after editing the configuration block:
    uv run upload_annotated_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import DatasetDict, Features, Sequence, Value, load_dataset

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
REPO_ID = "Vishva007/signature-detection-annotated"
SOURCE_DATASET = "tech4humans/signature-detection"
SPLITS = ["train", "validation", "test"]
ANNOTATION_DIR = Path(__file__).resolve().parent / "teacher_annotations" / "json"

PRIVATE = False
ONLY_ANNOTATED = False
DRY_RUN = False
MAX_SHARD_SIZE = "300MB"
COMMIT_MESSAGE = "Add teacher-generated annotation columns"

GROUNDING_FIELDS = {
    "queried_role": Value("string"),
    "question": Value("string"),
    "answer_text": Value("string"),
    "target_ids": Sequence(Value("int64")),
}
DISCOVERY_FIELDS = {
    "question": Value("string"),
    "answer_text": Value("string"),
    "target_ids": Sequence(Value("int64")),
}
PURE_VQA_FIELDS = {
    "question": Value("string"),
    "answer_text": Value("string"),
}


def load_annotations(annotation_dir: Path) -> dict[str, dict]:
    """Read every generated JSON, keyed by the generator's sample_key."""
    annotations: dict[str, dict] = {}
    for path in sorted(annotation_dir.glob("*.json")):
        if path.name.endswith(".tmp"):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        key = record.get("sample_key") or path.stem
        annotations[key] = record
    return annotations


def collect_bbox_ids(annotations: dict[str, dict]) -> list[int]:
    """Union of all ground-truth mark ids across the generated annotations."""
    ids = {
        int(mark_id)
        for record in annotations.values()
        for mark_id in (record.get("ground_truth_bboxes_norm_1000") or {})
        if str(mark_id).isdigit()
    }
    return sorted(ids) or [1]


def build_new_features(bbox_ids: list[int]) -> Features:
    """Explicit schema so nested columns encode identically for every row."""
    bbox_struct = {str(i): Sequence(Value("int64")) for i in bbox_ids}
    return Features(
        {
            "ground_truth_bboxes_norm_1000": bbox_struct,
            "has_signatures": Value("bool"),
            "marks": [{"id": Value("int64"), "label": Value("string")}],
            "positive_grounding": [GROUNDING_FIELDS],
            "hard_negative_grounding": [GROUNDING_FIELDS],
            "open_discovery": [DISCOVERY_FIELDS],
            "pure_vqa": [PURE_VQA_FIELDS],
        }
    )


def make_mapper(split: str, annotations: dict[str, dict], bbox_ids: list[int]):
    def attach(row: dict) -> dict:
        key = f"{split}_{row['image_id']}"
        record = annotations.get(key)

        if record is None:
            return {
                "ground_truth_bboxes_norm_1000": {str(i): None for i in bbox_ids},
                "has_signatures": None,
                "marks": [],
                "positive_grounding": [],
                "hard_negative_grounding": [],
                "open_discovery": [],
                "pure_vqa": [],
            }

        raw_bboxes = record.get("ground_truth_bboxes_norm_1000") or {}
        return {
            "ground_truth_bboxes_norm_1000": {
                str(i): raw_bboxes.get(str(i)) for i in bbox_ids
            },
            "has_signatures": bool(record.get("has_signatures")),
            "marks": record.get("marks") or [],
            "positive_grounding": record.get("positive_grounding") or [],
            "hard_negative_grounding": record.get("hard_negative_grounding") or [],
            "open_discovery": record.get("open_discovery") or [],
            "pure_vqa": record.get("pure_vqa") or [],
        }

    return attach


def main() -> None:
    if not ANNOTATION_DIR.is_dir():
        raise SystemExit(f"Annotation directory not found: {ANNOTATION_DIR}")

    annotations = load_annotations(ANNOTATION_DIR)
    if not annotations:
        raise SystemExit(f"No annotation JSON files found in {ANNOTATION_DIR}")

    source = load_dataset(SOURCE_DATASET)
    missing = [s for s in SPLITS if s not in source]
    if missing:
        raise SystemExit(f"Missing source split(s): {', '.join(missing)}")

    bbox_ids = collect_bbox_ids(annotations)
    new_features = build_new_features(bbox_ids)
    base_features = Features(dict(source[SPLITS[0]].features.items()))
    overlap = set(new_features) & set(base_features)
    if overlap:
        raise SystemExit(f"Source already has column(s): {', '.join(sorted(overlap))}")
    all_features = Features({**base_features, **new_features})

    source_keys = {
        f"{split}_{image_id}"
        for split in SPLITS
        for image_id in source[split]["image_id"]
    }
    unmatched = sorted(set(annotations) - source_keys)
    if unmatched:
        print(f"WARNING: {len(unmatched):,} annotation key(s) not found in the source splits")
        for key in unmatched[:10]:
            print(f"  {key}")
        if len(unmatched) > 10:
            print(f"  ... and {len(unmatched) - 10:,} more")

    built: DatasetDict = DatasetDict()
    for split in SPLITS:
        split_ds = source[split]
        if ONLY_ANNOTATED:
            split_ds = split_ds.filter(
                lambda row, s=split: f"{s}_{row['image_id']}" in annotations,
                desc=f"{split}: keep annotated",
            )
        built[split] = split_ds.map(
            make_mapper(split, annotations, bbox_ids),
            features=all_features,
            desc=f"{split}: attach teacher columns",
        )

    print(f"\nSource: {SOURCE_DATASET}")
    print(f"Annotations loaded: {len(annotations):,}")
    print(f"Bbox id fields: {', '.join(new_features['ground_truth_bboxes_norm_1000'])}")
    print(f"Target: {REPO_ID}\n")
    for split in SPLITS:
        annotated = sum(1 for rec in annotations.values() if rec.get("split") == split)
        print(
            f"  {split:<12} {len(built[split]):>7,} rows | "
            f"{annotated:>6,} annotated | {len(built[split]) - annotated:>7,} without annotation"
        )

    if DRY_RUN:
        print("\nDry run: nothing was pushed.")
        return

    built.push_to_hub(
        repo_id=REPO_ID,
        private=PRIVATE,
        max_shard_size=MAX_SHARD_SIZE,
        commit_message=COMMIT_MESSAGE,
    )
    print(f"\nPushed to https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
