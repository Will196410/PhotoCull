#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imagehash
import rawpy
from PIL import ExifTags, Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 250_000_000

RAW_EXTENSIONS = {".dng", ".arw", ".cr2", ".nef", ".orf", ".rw2", ".raf"}
STANDARD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".heic", ".heif", ".bmp"}
IMAGE_EXTS = STANDARD_EXTENSIONS | RAW_EXTENSIONS

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


def is_raw(path: Path) -> bool:
    return path.suffix.lower() in RAW_EXTENSIONS


def load_image_for_hash(path: Path) -> Optional[Image.Image]:
    ext = path.suffix.lower()
    try:
        if ext in RAW_EXTENSIONS:
            with rawpy.imread(str(path)) as raw:
                try:
                    thumb = raw.extract_thumb()
                    if thumb.format == rawpy.ThumbFormat.JPEG:
                        return Image.open(io.BytesIO(thumb.data)).convert("RGB")
                except Exception:
                    pass

                rgb = raw.postprocess(
                    use_camera_wb=True,
                    half_size=True,
                    auto_bright=False,
                    output_bps=8,
                )
                return Image.fromarray(rgb).convert("RGB")

        return Image.open(path).convert("RGB")
    except Exception:
        return None


def get_capture_time(path: Path) -> datetime:
    """
    Prefer EXIF DateTimeOriginal / DateTime for standard image formats.
    RAW files often do not expose useful EXIF through Pillow, so fall back to mtime.
    """
    if path.suffix.lower() in STANDARD_EXTENSIONS:
        try:
            with Image.open(path) as img:
                exif = img.getexif()
                if exif:
                    for field in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
                        tag_id = EXIF_TAGS.get(field)
                        if tag_id and tag_id in exif:
                            raw = exif.get(tag_id)
                            if raw:
                                try:
                                    return datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S")
                                except ValueError:
                                    pass
        except Exception:
            pass

    return datetime.fromtimestamp(path.stat().st_mtime)


def get_dimensions_and_phash(path: Path, hash_size: int = 16) -> Tuple[int, int, Optional[imagehash.ImageHash]]:
    img = load_image_for_hash(path)
    if img is None:
        return 0, 0, None

    try:
        width, height = img.size
        ph = imagehash.phash(img, hash_size=hash_size)
        return width, height, ph
    except Exception:
        return 0, 0, None
    finally:
        try:
            img.close()
        except Exception:
            pass


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
                relpath=str(path.relative_to(root)).replace("\\", "/"),
                size=stat.st_size,
                width=width,
                height=height,
                mtime=datetime.fromtimestamp(stat.st_mtime),
                capture_time=capture,
                phash=ph,
            )
        )

    return photos


def normalise_stem(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"[\s_-]+", "", stem)
    return stem


def format_rank(path: Path) -> int:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return 5
    if ext in {".tif", ".tiff", ".png", ".webp", ".heic", ".heif", ".bmp"}:
        return 4
    if ext in RAW_EXTENSIONS:
        return 2
    return 1


def score_photo(p: PhotoInfo) -> Tuple[int, int, int, int, str]:
    """
    Higher is better for gallery use.
    Prefer rendered formats over RAW when they represent the same image.
    Then prefer resolution, size, and cleaner filenames.
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

    if any(name.startswith(prefix) for prefix in ("dsc", "img_", "p", "pxl_", "dji_")):
        penalty += 1

    return (format_rank(p.path), megapixels, p.size, penalty, p.relpath)


def choose_keeper(group: List[PhotoInfo]) -> PhotoInfo:
    return max(group, key=score_photo)


def cluster_exact_duplicates(photos: List[PhotoInfo], verbose: bool = False) -> List[List[PhotoInfo]]:
    by_size: Dict[int, List[PhotoInfo]] = {}
    for p in photos:
        by_size.setdefault(p.size, []).append(p)

    exact_groups: List[List[PhotoInfo]] = []

    for candidates in by_size.values():
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
    Compares images close in capture time, plus files that share the same
    base stem like DSC02483.ARW and DSC02483.JPG.
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

    stem_map: Dict[str, List[int]] = {}
    for i, p in enumerate(valid):
        stem_map.setdefault(normalise_stem(p.path), []).append(i)

    for indices in stem_map.values():
        if len(indices) < 2:
            continue

        for a_pos in range(len(indices)):
            for b_pos in range(a_pos + 1, len(indices)):
                a = indices[a_pos]
                b = indices[b_pos]
                dist = valid[a].phash - valid[b].phash
                if dist <= phash_threshold + 2:
                    dsu.union(a, b)

    groups_map: Dict[int, List[PhotoInfo]] = {}
    for i, p in enumerate(valid):
        root = dsu.find(i)
        groups_map.setdefault(root, []).append(p)

    groups = [g for g in groups_map.values() if len(g) > 1]

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

    root = args.root.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
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
