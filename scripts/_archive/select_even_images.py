#!/usr/bin/env python3
"""Select evenly spaced images from a frame range and copy into target folder."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET_COUNT = 200
FRAME_START = 500  # inclusive
FRAME_END = 2500  # exclusive
SRC_DIR = Path("/mnt/h/RGBD/boardroom/depth")
DST_DIR = Path("/mnt/h/RGBD/depth_200")


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("count must be positive")
    if total <= 0:
        raise ValueError("total must be positive")
    if count == 1:
        return [0]
    raw_positions = [i * (total - 1) / (count - 1) for i in range(count)]
    indices: list[int] = []
    last = -1
    for i, value in enumerate(raw_positions):
        idx = int(round(value))
        if idx < 0:
            idx = 0
        if idx <= last:
            idx = last + 1
        max_allowed = total - (count - i)
        if idx > max_allowed:
            idx = max_allowed
        if idx >= total:
            idx = total - 1
        indices.append(idx)
        last = idx
    return indices


def frame_number(path: Path) -> int | None:
    stem = path.stem
    try:
        return int(stem)
    except ValueError:
        return None


def clear_destination(directory: Path) -> None:
    if not directory.exists():
        return
    for entry in directory.iterdir():
        if entry.is_file():
            entry.unlink()


def main() -> int:
    if not SRC_DIR.exists():
        print(f"Source directory {SRC_DIR} does not exist", file=sys.stderr)
        return 1

    all_images = sorted(
        [p for p in SRC_DIR.iterdir() if p.is_file()],
        key=lambda path: path.name,
    )

    filtered = []
    for image in all_images:
        number = frame_number(image)
        if number is None:
            continue
        if FRAME_START <= number < FRAME_END:
            filtered.append(image)

    total = len(filtered)
    if total == 0:
        print(
            f"No files found in {SRC_DIR} between frames {FRAME_START:06d}-{FRAME_END:06d}",
            file=sys.stderr,
        )
        return 1

    if total < TARGET_COUNT:
        print(
            f"Only found {total} files between frames {FRAME_START:06d}-{FRAME_END:06d}; requested {TARGET_COUNT}",
            file=sys.stderr,
        )
        return 1

    DST_DIR.mkdir(parents=True, exist_ok=True)
    clear_destination(DST_DIR)

    indices = evenly_spaced_indices(total, TARGET_COUNT)

    for idx in indices:
        source_path = filtered[idx]
        destination_path = DST_DIR / source_path.name
        shutil.copy2(source_path, destination_path)

    print(
        f"Copied {TARGET_COUNT} files from frames {FRAME_START:06d}-{FRAME_END:06d} "
        f"({total} available) into {DST_DIR}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
