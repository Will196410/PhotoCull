#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ExifTags
import imagehash


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".heic", ".heif"}
EXIF_TAGS = {v: k for k, v in ExifTags.TAGS.items()}


@dataclass
class PhotoInfo:
    path: Path
    relpath: str
    size: int
    width: int
    height: int
    mtime: datetime
    capture_time: datetime
    phash: Optional[imagehash.ImageHash]
    sha256: Optional[str] = None


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def iter_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and is_image(p)]


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_capture_time(path: Path) -> datetime:
    """
    Prefer EXIF DateTimeOriginal / DateTime.
    Fall back to filesystem mtime.
    """
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if exif:
                for field in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
                    tag_id = EXIF_TAGS.get(field)
                    if tag_id and tag_id in exif:
                        raw = exif.get(tag_id)
                        if raw:
                            # EXIF format: YYYY:MM:DD HH:MM:SS
                            try:
                                return datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S")
                            except ValueError:
                                pass
    except Exception:
        pass

    return datetime.fromtimestamp(path.stat().st_mtime)


def get_dimensions_and_phash(path: Path, hash_size: int = 16) -> Tuple[int, int, Optional[imagehash.ImageHash]]:
    try:
        with Image.open(path) as img:
            width, height = img.size
            ph = imagehash.phash(img, hash_size=hash_size)
            return width, height, ph
    except Exception:
        return 0, 0, None


def load_photo_info(root: Path, verbose: bool = False) -> List[PhotoInfo]:
    photos: List[PhotoInfo] = []
    files = iter_images(root)

    for idx, path in enumerate(files, start=1):
        if verbose and idx % 200 == 0:
            print(f"Scanned {idx}/{len(files)} files...")

        stat = path.stat()
        capture = get_capture_time(path)
        width, height, ph = get_dimensions_and_phash(path)

        photos.append(
            PhotoInfo(
                path=path,
                relpath=str(path.relative_to(root)),
                size=stat.st_size,
                width=width,
                height=height,
                mtime=datetime.fromtimestamp(stat.st_mtime),
                capture_time=capture,
                phash=ph,
            )
        )

    return photos


def score_photo(p: PhotoInfo) -> Tuple[int, int, int, str]:
    """
    Higher is better.
    Heuristics:
    - bigger resolution preferred
    - bigger file preferred
    - originals preferred over edited/export-ish names
    - stable tie-break by path
    """
    megapixels = p.width * p.height
    name = p.path.name.lower()

    penalty = 0
    bad_tokens = [
        "edited", "export", "copy", "duplicate", "small", "medium", "thumb",
        "preview", "instagram", "lightroom", "snapseed", "dxo"
    ]
    if any(tok in name for tok in bad_tokens):
        penalty -= 1

    # Favor camera originals a bit
    if any(name.startswith(prefix) for prefix in ("dsc", "img_", "p", "pxl_", "dji_")):
        penalty += 1

    return (megapixels, p.size, penalty, p.relpath)


def choose_keeper(group: List[PhotoInfo]) -> PhotoInfo:
    return max(group, key=score_photo)


def cluster_exact_duplicates(photos: List[PhotoInfo], verbose: bool = False) -> List[List[PhotoInfo]]:
    by_size: Dict[int, List[PhotoInfo]] = {}
    for p in photos:
        by_size.setdefault(p.size, []).append(p)

    exact_groups: List[List[PhotoInfo]] = []

    for size, candidates in by_size.items():
        if len(candidates) < 2:
            continue

        by_hash: Dict[str, List[PhotoInfo]] = {}
        for p in candidates:
            if p.sha256 is None:
                p.sha256 = file_sha256(p.path)
            by_hash.setdefault(p.sha256, []).append(p)

        for group in by_hash.values():
            if len(group) > 1:
                exact_groups.append(group)

    if verbose:
        print(f"Found {len(exact_groups)} exact duplicate groups")

    return exact_groups


class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def cluster_near_duplicates(
    photos: List[PhotoInfo],
    seconds_window: int,
    phash_threshold: int,
    verbose: bool = False,
) -> List[List[PhotoInfo]]:
    """
    Compares only images close in capture time.
    Intended for burst-like sequences.
    """
    valid = [p for p in photos if p.phash is not None]
    valid.sort(key=lambda p: p.capture_time)

    n = len(valid)
    dsu = DSU(n)
    window = timedelta(seconds=seconds_window)

    for i in range(n):
        j = i + 1
        while j < n and (valid[j].capture_time - valid[i].capture_time) <= window:
            dist = valid[i].phash - valid[j].phash
            if dist <= phash_threshold:
                dsu.union(i, j)
            j += 1

    groups_map: Dict[int, List[PhotoInfo]] = {}
    for i, p in enumerate(valid):
        root = dsu.find(i)
        groups_map.setdefault(root, []).append(p)

    groups = [g for g in groups_map.values() if len(g) > 1]

    # Exclude exact-duplicate-only groups from near groups later by path set logic if needed
    if verbose:
        print(f"Found {len(groups)} near-duplicate groups")

    return groups


def write_exact_csv(path: Path, groups: List[List[PhotoInfo]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "keeper", "relpath", "size", "width", "height", "capture_time", "sha256"])
        for idx, group in enumerate(groups, start=1):
            keeper = choose_keeper(group)
            for p in sorted(group, key=lambda x: x.relpath):
                w.writerow([
                    idx,
                    "yes" if p.relpath == keeper.relpath else "no",
                    p.relpath,
                    p.size,
                    p.width,
                    p.height,
                    p.capture_time.isoformat(sep=" "),
                    p.sha256 or "",
                ])


def write_near_csv(path: Path, groups: List[List[PhotoInfo]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "keeper", "relpath", "size", "width", "height", "capture_time", "phash"])
        for idx, group in enumerate(groups, start=1):
            keeper = choose_keeper(group)
            for p in sorted(group, key=lambda x: x.relpath):
                w.writerow([
                    idx,
                    "yes" if p.relpath == keeper.relpath else "no",
                    p.relpath,
                    p.size,
                    p.width,
                    p.height,
                    p.capture_time.isoformat(sep=" "),
                    str(p.phash) if p.phash is not None else "",
                ])


def normalise_groups(groups: List[List[PhotoInfo]]) -> List[List[PhotoInfo]]:
    seen = set()
    result = []

    for group in groups:
        key = tuple(sorted(p.relpath for p in group))
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        result.append(group)

    return result


def remove_exact_only_from_near(
    exact_groups: List[List[PhotoInfo]],
    near_groups: List[List[PhotoInfo]],
) -> List[List[PhotoInfo]]:
    exact_sets = {frozenset(p.relpath for p in g) for g in exact_groups}
    cleaned = []
    for g in near_groups:
        s = frozenset(p.relpath for p in g)
        if s not in exact_sets:
            cleaned.append(g)
    return cleaned


def build_excludes(
    exact_groups: List[List[PhotoInfo]],
    near_groups: List[List[PhotoInfo]],
) -> List[str]:
    excluded = set()

    for group in exact_groups + near_groups:
        keeper = choose_keeper(group)
        for p in group:
            if p.relpath != keeper.relpath:
                excluded.add(p.relpath)

    return sorted(excluded)


def write_excludes_txt(path: Path, excluded: List[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for relpath in excluded:
            f.write(relpath + "\n")


def write_summary_json(
    path: Path,
    photos: List[PhotoInfo],
    exact_groups: List[List[PhotoInfo]],
    near_groups: List[List[PhotoInfo]],
    excluded: List[str],
    args: argparse.Namespace,
) -> None:
    data = {
        "root": str(args.root),
        "generated_at": datetime.now().isoformat(),
        "photo_count": len(photos),
        "exact_group_count": len(exact_groups),
        "near_group_count": len(near_groups),
        "excluded_count": len(excluded),
        "seconds_window": args.seconds_window,
        "phash_threshold": args.phash_threshold,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Find exact and burst-like duplicate photos for gallery exclusion.")
    parser.add_argument("root", type=Path, help="Root of photo archive, e.g. /Volumes/All Photos/Photos")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("./dedupe_output"),
        help="Directory to write reports into",
    )
    parser.add_argument(
        "--seconds-window",
        type=int,
        default=8,
        help="Only compare photos taken within this many seconds",
    )
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=6,
        help="Perceptual hash distance threshold for near duplicates",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(f"Scanning: {root}")

    photos = load_photo_info(root, verbose=args.verbose)
    exact_groups = normalise_groups(cluster_exact_duplicates(photos, verbose=args.verbose))
    near_groups = normalise_groups(
        remove_exact_only_from_near(
            exact_groups,
            cluster_near_duplicates(
                photos,
                seconds_window=args.seconds_window,
                phash_threshold=args.phash_threshold,
                verbose=args.verbose,
            ),
        )
    )

    excluded = build_excludes(exact_groups, near_groups)

    write_exact_csv(outdir / "dedupe_exact.csv", exact_groups)
    write_near_csv(outdir / "dedupe_groups.csv", near_groups)
    write_excludes_txt(outdir / "gallery_excludes.txt", excluded)
    write_summary_json(outdir / "dedupe_summary.json", photos, exact_groups, near_groups, excluded, args)

    print(f"Photos scanned:        {len(photos)}")
    print(f"Exact duplicate groups:{len(exact_groups)}")
    print(f"Near duplicate groups: {len(near_groups)}")
    print(f"Excluded for gallery:  {len(excluded)}")
    print(f"Reports written to:    {outdir}")


if __name__ == "__main__":
    main()
