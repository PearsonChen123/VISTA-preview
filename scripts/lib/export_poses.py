import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch

from nerfstudio.utils.eval_utils import eval_setup


def _collect_frames(dataset) -> List[Dict]:
    cameras = dataset.cameras
    image_filenames = dataset.image_filenames
    frames: List[Dict] = []
    for idx in range(len(cameras)):
        transform = cameras.camera_to_worlds[idx].tolist()
        frames.append({
            "file_path": str(image_filenames[idx]),
            "transform": transform,
        })
    return frames


def _frame_key(path: str) -> str:
    if not path:
        return ""
    return Path(path).name.lower()


def _load_reference_frames(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        frames = data.get("frames")
        if frames is None:
            raise ValueError(f"Reference file {path} missing 'frames' key")
        return frames
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported reference format in {path}")


def _merge_frames(
    train_frames: Iterable[Dict],
    eval_frames: Iterable[Dict],
    reference_frames: Optional[Iterable[Dict]] = None,
) -> List[Dict]:
    frame_lookup: Dict[str, Dict] = {}
    duplicates: List[str] = []

    def _register(frames: Iterable[Dict]) -> None:
        for frame in frames:
            key = _frame_key(frame.get("file_path", ""))
            if not key:
                continue
            if key in frame_lookup:
                duplicates.append(key)
            frame_lookup[key] = frame

    _register(train_frames)
    _register(eval_frames)

    if reference_frames is not None:
        merged: List[Dict] = []
        missing: List[str] = []
        for ref in reference_frames:
            ref_path = ref.get("file_path") or ref.get("image_path") or ref.get("path")
            key = _frame_key(ref_path or "")
            if not key or key not in frame_lookup:
                missing.append(str(ref_path))
                continue
            source = frame_lookup[key]
            merged.append({
                "file_path": ref_path if ref_path is not None else source.get("file_path", key),
                "transform": source["transform"],
            })
        if missing:
            print(f"Warning: missing {len(missing)} frames from reference when merging: {missing[:5]}")
        if duplicates:
            print(f"Info: duplicate frame basenames encountered during merge: {sorted(set(duplicates))}")
        return merged

    # Fallback: sort by key for deterministic ordering.
    ordered_keys = sorted(frame_lookup.keys())
    return [frame_lookup[key] for key in ordered_keys]


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight replacement for 'ns-export cameras'.")
    parser.add_argument("--load-config", dest="load_config", required=True, type=Path,
                        help="Path to Nerfstudio config.yml")
    parser.add_argument("--output-dir", dest="output_dir", required=True, type=Path,
                        help="Directory to write transforms JSON files")
    parser.add_argument("--no-train", dest="skip_train", action="store_true",
                        help="Skip exporting train poses")
    parser.add_argument("--no-eval", dest="skip_eval", action="store_true",
                        help="Skip exporting eval poses")
    parser.add_argument("--combine-train-eval", action="store_true",
                        help="Include eval poses in transforms_train.json")
    parser.add_argument("--reference-transforms", type=Path,
                        help="Path to a transforms.json file used to define ordering when combining datasets")
    args = parser.parse_args()

    _, pipeline, _, _ = eval_setup(args.load_config)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = pipeline.datamanager.train_dataset
    train_frames = _collect_frames(train_dataset)

    need_eval = args.combine_train_eval or not args.skip_eval
    eval_frames: List[Dict] = []
    if need_eval:
        eval_dataset = pipeline.datamanager.eval_dataset
        eval_frames = _collect_frames(eval_dataset)

    frames_for_train = train_frames
    if args.combine_train_eval:
        reference_frames = None
        if args.reference_transforms is not None:
            reference_frames = _load_reference_frames(args.reference_transforms)
        frames_for_train = _merge_frames(train_frames, eval_frames, reference_frames)

    if not args.skip_train:
        if frames_for_train:
            train_path = output_dir / "transforms_train.json"
            with train_path.open("w", encoding="utf-8") as f:
                json.dump(frames_for_train, f, indent=2)
        else:
            print("Warning: train dataset empty, skipping transforms_train.json")

    if not args.skip_eval:
        if eval_frames:
            eval_path = output_dir / "transforms_eval.json"
            with eval_path.open("w", encoding="utf-8") as f:
                json.dump(eval_frames, f, indent=2)
        else:
            print("Warning: eval dataset empty, skipping transforms_eval.json")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
