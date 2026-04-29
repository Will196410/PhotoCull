#!/usr/bin/env python3
"""
photo_themes_v2.py

Alpha upgrade of photo_themes.py.

What changed from the original:
- Same core outputs: {year}_images.csv, {year}_themes.csv, HTML gallery.
- Adds --model and --model-kind so you can test CLIP vs newer embedding models.
- Adds an embedding cache so clustering/prompt experiments do not always re-embed images.
- Adds a richer fallback prompt list for archive discovery.
- Keeps the existing thumbnail/gallery workflow.

Typical use:
    python photo_themes_v2.py "/Volumes/All Photos/Photos" --year 2023

Try SigLIP-style model, if your local transformers install supports it:
    python photo_themes_v2.py "/Volumes/All Photos/Photos" --year 2023 \
      --model google/siglip2-base-patch16-224 --model-kind auto

Force cache rebuild:
    python photo_themes_v2.py "/Volumes/All Photos/Photos" --year 2023 --force-rebuild-cache
"""

import argparse
import hashlib
import html
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rawpy
import torch
from PIL import Image, ImageFile
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True

RAW_EXTENSIONS = {".dng", ".arw", ".cr2", ".nef", ".orf", ".rw2", ".raf"}
STANDARD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".webp", ".bmp"}
SUPPORTED_EXTENSIONS = RAW_EXTENSIONS | STANDARD_EXTENSIONS

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "theme_output"
DEFAULT_PROMPTS_FILE = Path(__file__).resolve().parent / "theme_prompts.txt"
DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"

FALLBACK_THEME_PROMPTS = [
    # Core landscape / water
    "a coastal landscape photograph",
    "a cliff-top coastal landscape photograph",
    "a beach with sea and shoreline photograph",
    "a harbour or port scene with boats",
    "boats in a harbour photograph",
    "boats on a river photograph",
    "a waterside or river scene",
    "a countryside landscape photograph",
    "a woodland or forest scene",
    "a mountain, hill, or valley landscape photograph",

    # Nature and detail
    "a flower or plant close-up photograph",
    "a garden photograph",
    "a tree or foliage photograph",
    "a macro or texture detail photograph",
    "an abstract visual pattern photograph",

    # Animals
    "a bird or wildlife photograph",
    "wild animals in a natural setting photograph",
    "a farm animal photograph",
    "a herd of cows in a field photograph",
    "sheep grazing in a field photograph",
    "horses in a rural landscape photograph",
    "farm animals in pasture photograph",
    "a pet photograph",
    "a dog or cat photograph",

    # People
    "a people or group photograph",
    "a portrait of one person",
    "people walking in a town photograph",
    "people at an event photograph",
    "musicians or performers photograph",
    "workers or police officers photograph",

    # Place / travel / built environment
    "an old building or historic architecture photograph",
    "a village, town, or street scene photograph",
    "a travel photograph showing a place",
    "a city or urban street photograph",
    "a church or cathedral photograph",
    "a market or public square photograph",
    "a transport or vehicle photograph",
    "cars parked on a street photograph",
    "bicycles or cyclists photograph",
    "tents or camping equipment photograph",
    "an indoor scene photograph",
    "an indoor museum or gallery scene photograph",

    # Weather / light / seasonal
    "a dramatic sky photograph",
    "a sunset or sunrise photograph",
    "a misty or foggy landscape photograph",
    "a stormy weather photograph",
    "a moody atmospheric landscape photograph",
    "a photograph where light and weather create the mood",
    "a snow or winter landscape photograph",
    "a night street photograph",
    "a summer countryside photograph",
]


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def slugify(value: str, max_len: int = 80) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:max_len] or "model"


def short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]


def model_cache_key(model_name: str, model_kind: str) -> str:
    return f"{slugify(model_kind)}-{slugify(model_name)}-{short_hash(model_kind + '|' + model_name)}"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_theme_prompts(prompts_file: Optional[Path]) -> List[str]:
    if prompts_file is None:
        return FALLBACK_THEME_PROMPTS

    if not prompts_file.exists():
        print(f"Warning: prompts file not found: {prompts_file}")
        print("Falling back to built-in prompts.")
        return FALLBACK_THEME_PROMPTS

    lines = prompts_file.read_text(encoding="utf-8").splitlines()
    prompts = []
    for line in lines:
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        prompts.append(text)

    if not prompts:
        print(f"Warning: prompts file is empty after filtering comments/blanks: {prompts_file}")
        print("Falling back to built-in prompts.")
        return FALLBACK_THEME_PROMPTS

    print(f"Loaded {len(prompts)} theme prompts from {prompts_file}")
    return prompts


def is_ignored(path: Path) -> bool:
    s = str(path).lower()
    if path.name.startswith("."):
        return True
    ignored_fragments = ["metadata", "preview", "thumbnail", "sidecar"]
    return any(fragment in s for fragment in ignored_fragments)


def get_image_for_ai(path: Path) -> Optional[Image.Image]:
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


def make_thumbnail(image: Image.Image, thumb_path: Path, size=(360, 360)) -> None:
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    if thumb_path.exists():
        return
    thumb_img = image.copy()
    thumb_img.thumbnail(size)
    thumb_img.convert("RGB").save(thumb_path, "JPEG", quality=88)


def load_exclude_set(exclude_file: Optional[Path]) -> set:
    if exclude_file is None:
        return set()
    if not exclude_file.exists():
        print(f"Warning: exclude file not found: {exclude_file}")
        return set()
    lines = exclude_file.read_text(encoding="utf-8").splitlines()
    excluded = {line.strip().replace("\\", "/") for line in lines if line.strip()}
    print(f"Loaded {len(excluded)} excluded paths from {exclude_file}")
    return excluded


def iter_images(scan_root: Path, archive_root: Path, excluded_relative_paths=None):
    excluded_relative_paths = excluded_relative_paths or set()
    for p in scan_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if is_ignored(p):
            continue
        archive_rel = p.relative_to(archive_root).as_posix()
        if archive_rel in excluded_relative_paths:
            continue
        yield p


def count_candidate_images(scan_root: Path) -> int:
    count = 0
    for p in scan_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and not is_ignored(p):
            count += 1
    return count


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# ---------------------------------------------------------------------------
# Model adapter
# ---------------------------------------------------------------------------

class EmbeddingModel:
    def __init__(self, model_name: str, model_kind: str, device: str):
        self.model_name = model_name
        self.model_kind = model_kind
        self.device = device

        resolved_kind = model_kind
        if model_kind == "auto":
            lower = model_name.lower()
            if "clip" in lower and "siglip" not in lower:
                resolved_kind = "clip"
            else:
                resolved_kind = "auto"
        self.resolved_kind = resolved_kind

        print(f"Initializing Theme Judge on {device}...")
        print(f"Model: {model_name} ({model_kind})")

        if resolved_kind == "clip":
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).to(device)
        else:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)

        self.model.eval()

    def _move_inputs(self, inputs):
        return {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    def encode_text(self, prompts: List[str], batch_size: int = 64) -> np.ndarray:
        feats = []
        for batch in chunked(prompts, batch_size):
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            inputs = self._move_inputs(inputs)
            with torch.inference_mode():
                if hasattr(self.model, "get_text_features"):
                    out = self.model.get_text_features(**inputs)
                else:
                    result = self.model(**inputs)
                    out = result.text_embeds if hasattr(result, "text_embeds") else result.pooler_output
                out = out / out.norm(dim=-1, keepdim=True)
            feats.append(out.detach().cpu().float().numpy())
        return np.concatenate(feats, axis=0)

    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = self._move_inputs(inputs)
        with torch.inference_mode():
            if hasattr(self.model, "get_image_features"):
                out = self.model.get_image_features(**inputs)
            else:
                result = self.model(**inputs)
                out = result.image_embeds if hasattr(result, "image_embeds") else result.pooler_output
            out = out / out.norm(dim=-1, keepdim=True)
        return out.detach().cpu().float().numpy()


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def cache_paths(out_dir: Path, cache_key: str) -> Tuple[Path, Path, Path]:
    cache_dir = out_dir / "embedding_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return (
        cache_dir / f"{cache_key}_embeddings.npy",
        cache_dir / f"{cache_key}_manifest.csv",
        cache_dir / f"{cache_key}_meta.json",
    )


def manifest_for_paths(image_paths: List[Path], archive_root: Path, year_dir: Path) -> pd.DataFrame:
    rows = []
    for idx, path in enumerate(image_paths):
        try:
            stat = path.stat()
            size = stat.st_size
            mtime = int(stat.st_mtime)
        except Exception:
            size = -1
            mtime = -1
        rows.append({
            "row_index": idx,
            "path": str(path.resolve()),
            "relative_path": path.relative_to(year_dir).as_posix(),
            "archive_relative_path": path.relative_to(archive_root).as_posix(),
            "file": path.name,
            "size": size,
            "mtime": mtime,
        })
    return pd.DataFrame(rows)


def manifests_match(current: pd.DataFrame, cached: pd.DataFrame) -> bool:
    cols = ["path", "archive_relative_path", "size", "mtime"]
    if len(current) != len(cached):
        return False
    for col in cols:
        if col not in cached.columns or col not in current.columns:
            return False
        if current[col].astype(str).tolist() != cached[col].astype(str).tolist():
            return False
    return True


def try_load_cache(embeddings_path: Path, manifest_path: Path, current_manifest: pd.DataFrame, force: bool) -> Optional[np.ndarray]:
    if force:
        return None
    if not embeddings_path.exists() or not manifest_path.exists():
        return None
    try:
        cached_manifest = pd.read_csv(manifest_path)
        if not manifests_match(current_manifest, cached_manifest):
            print("Embedding cache exists, but manifest does not match current image set. Rebuilding.")
            return None
        embeddings = np.load(embeddings_path)
        if len(embeddings) != len(current_manifest):
            print("Embedding cache exists, but embedding count does not match. Rebuilding.")
            return None
        print(f"Loaded cached embeddings: {embeddings_path}")
        return embeddings.astype("float32")
    except Exception as exc:
        print(f"Could not load embedding cache: {exc}. Rebuilding.")
        return None


def save_cache(embeddings: np.ndarray, manifest: pd.DataFrame, embeddings_path: Path, manifest_path: Path, meta_path: Path, meta: Dict) -> None:
    np.save(embeddings_path, embeddings.astype("float32"))
    manifest.to_csv(manifest_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved embedding cache: {embeddings_path}")


# ---------------------------------------------------------------------------
# Theme labelling
# ---------------------------------------------------------------------------

def clean_prompt_label(label: str) -> str:
    label = label.strip()
    replacements = [
        ("a ", ""),
        ("an ", ""),
        (" photograph", ""),
        (" photo", ""),
        (" scene", ""),
        (" image", ""),
    ]
    for old, new in replacements:
        label = label.replace(old, new)
    return " ".join(label.split())


def get_dominant_subfolder(items) -> str:
    counts = Counter()
    for r in items:
        rel = Path(r["relative_path"])
        parts = rel.parts[:-1]
        counts[parts[0] if parts else "."] += 1
    return counts.most_common(1)[0][0] if counts else "."


def build_display_theme_name(top_labels, dominant_folder, cluster_id) -> str:
    primary = clean_prompt_label(top_labels[0])
    secondary = clean_prompt_label(top_labels[1]) if len(top_labels) > 1 else ""

    weak_seconds = {
        "travel showing a place",
        "travel photograph showing a place",
        "village, town, or street",
        "indoor",
    }

    parts = [primary]
    if secondary and secondary != primary and secondary not in weak_seconds:
        parts.append(secondary)
    if dominant_folder and dominant_folder != ".":
        parts.append(dominant_folder)
    parts.append(f"{int(cluster_id):02d}")
    return " • ".join(parts)


def choose_representative_indices(cluster_rows, embeddings, limit=3) -> List[int]:
    if len(cluster_rows) <= limit:
        return [r["row_index"] for r in cluster_rows]
    vecs = np.stack([embeddings[r["row_index"]] for r in cluster_rows], axis=0)
    centroid = vecs.mean(axis=0)
    centroid /= max((centroid @ centroid) ** 0.5, 1e-8)
    sims = vecs @ centroid
    ranked = sorted(zip(cluster_rows, sims), key=lambda x: x[1], reverse=True)
    return [item[0]["row_index"] for item in ranked[:limit]]


def make_clusterer(distance_threshold: float):
    try:
        return AgglomerativeClustering(
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
            n_clusters=None,
        )
    except TypeError:
        return AgglomerativeClustering(
            affinity="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
            n_clusters=None,
        )


# ---------------------------------------------------------------------------
# HTML gallery
# ---------------------------------------------------------------------------

def build_gallery_html(
    out_dir: Path,
    year: str,
    rows: List[Dict],
    grouped: Dict[int, List[Dict]],
    cluster_label_map: Dict[int, Dict],
    cluster_order: List[int],
    total_candidates_before_exclusions: int,
    excluded_count: int,
    model_name: str,
    cache_key: str,
) -> Path:
    toc_items = []
    for cid in cluster_order:
        items = grouped[cid]
        meta = cluster_label_map[cid]
        anchor_id = f"theme-{cid}"
        safe_name = html.escape(meta["display_theme_name"])
        toc_items.append(
            f'<li><a href="#{anchor_id}">{safe_name}</a> <span class="toc-count">({len(items)})</span></li>'
        )

    toc_html = f"""
        <div class="top-panel">
            <div class="stats">
                <div><strong>Year:</strong> {html.escape(year)}</div>
                <div><strong>Total candidates found:</strong> {total_candidates_before_exclusions}</div>
                <div><strong>Included after exclusions:</strong> {len(rows)}</div>
                <div><strong>Excluded before clustering:</strong> {excluded_count}</div>
                <div><strong>Theme clusters:</strong> {len(cluster_order)}</div>
                <div><strong>Model:</strong> {html.escape(model_name)}</div>
                <div><strong>Cache key:</strong> {html.escape(cache_key)}</div>
                <div><strong>Output folder:</strong> {html.escape(str(out_dir))}</div>
            </div>
            <div class="toc">
                <div class="toc-title">Jump to a theme</div>
                <ul>{''.join(toc_items)}</ul>
            </div>
        </div>
    """

    html_header = f"""
    <html>
    <head>
    <meta charset="utf-8">
    <title>Photo Themes {html.escape(year)}</title>
    <style>
        html {{ scroll-behavior: smooth; }}
        body {{ background:#111; color:white; font-family:system-ui,-apple-system,sans-serif; margin:0; }}
        .wrap {{ max-width:1600px; margin:0 auto; padding:20px; }}
        h1 {{ margin-top:8px; margin-bottom:8px; font-size:28px; }}
        .summary {{ color:#bbb; margin-bottom:22px; }}
        .top-panel {{ display:grid; grid-template-columns:minmax(280px,1fr) minmax(420px,2fr); gap:24px; margin-bottom:28px; padding:18px; background:#181818; border:1px solid #333; border-radius:12px; }}
        .stats {{ color:#ddd; font-size:14px; line-height:1.8; }}
        .toc-title {{ font-size:16px; font-weight:700; margin-bottom:10px; color:#fff; }}
        .toc ul {{ margin:0; padding-left:18px; columns:2; column-gap:24px; }}
        .toc li {{ margin-bottom:6px; break-inside:avoid; }}
        .toc a, .back-top {{ color:#9dd; text-decoration:none; }}
        .toc a:hover, .back-top:hover {{ text-decoration:underline; }}
        .toc-count {{ color:#888; font-size:12px; }}
        .theme-block {{ margin-bottom:34px; padding-bottom:18px; border-bottom:1px solid #333; scroll-margin-top:20px; }}
        .theme-head {{ display:flex; justify-content:space-between; align-items:baseline; gap:16px; margin-bottom:6px; }}
        .theme-title {{ font-size:22px; margin-bottom:0; }}
        .back-top {{ font-size:12px; white-space:nowrap; }}
        .theme-meta {{ color:#9dd; font-size:13px; margin-bottom:12px; }}
        .theme-labels {{ color:#aaa; font-size:12px; margin-bottom:14px; }}
        .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:14px; }}
        .card {{ background:#222; border:1px solid #333; border-radius:10px; overflow:hidden; }}
        .thumb-btn {{ display:block; width:100%; padding:0; margin:0; border:0; background:transparent; cursor:pointer; }}
        img {{ width:100%; aspect-ratio:1; object-fit:cover; display:block; background:#000; }}
        .info {{ padding:10px; font-size:12px; }}
        .file {{ color:#ddd; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
        .path {{ color:#777; font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-top:4px; }}
        .hint {{ color:#aaa; font-size:11px; margin-top:6px; }}
        .toast {{ position:fixed; bottom:20px; left:50%; transform:translateX(-50%); background:#00ffcc; color:#111; padding:10px 14px; border-radius:999px; font-size:13px; font-weight:600; box-shadow:0 6px 20px rgba(0,0,0,0.35); opacity:0; pointer-events:none; transition:opacity .2s ease; z-index:1000; }}
        .toast.show {{ opacity:1; }}
        @media (max-width:900px) {{ .top-panel {{ grid-template-columns:1fr; }} .toc ul {{ columns:1; }} }}
    </style>
    </head>
    <body>
    <div class="wrap" id="top">
        <h1>Photo Themes for {html.escape(year)}</h1>
        <div class="summary">
            {len(rows)} images grouped into {len(cluster_order)} theme clusters.
            Excluded before clustering: {excluded_count}.
            Total candidates in year folder: {total_candidates_before_exclusions}.
        </div>
        {toc_html}
    """

    script = """
    <script>
    async function copyText(text) {
        try {
            if (navigator.clipboard && window.isSecureContext) {
                await navigator.clipboard.writeText(text);
                return true;
            }
        } catch (e) {}
        try {
            const ta = document.createElement('textarea');
            ta.value = text;
            ta.setAttribute('readonly', '');
            ta.style.position = 'fixed';
            ta.style.left = '-9999px';
            ta.style.top = '0';
            document.body.appendChild(ta);
            ta.focus();
            ta.select();
            const ok = document.execCommand('copy');
            document.body.removeChild(ta);
            return ok;
        } catch (e) {
            return false;
        }
    }
    function showToast(message) {
        const toast = document.getElementById('toast');
        toast.textContent = message;
        toast.classList.add('show');
        clearTimeout(window.__toastTimer);
        window.__toastTimer = setTimeout(() => toast.classList.remove('show'), 1400);
    }
    async function copyPath(path) {
        const ok = await copyText(path);
        showToast(ok ? 'Copied path' : 'Could not copy path');
    }
    </script>
    """

    blocks = []
    for cid in cluster_order:
        items = grouped[cid]
        meta = cluster_label_map[cid]
        theme_name = html.escape(meta["display_theme_name"])
        top_labels = " • ".join(html.escape(clean_prompt_label(x)) for x in meta["top_labels"])
        dominant_folder = html.escape(meta["dominant_folder"])
        anchor_id = f"theme-{cid}"

        block = f"""
        <div class="theme-block" id="{anchor_id}">
            <div class="theme-head">
                <div class="theme-title">{theme_name}</div>
                <a class="back-top" href="#top">Back to top</a>
            </div>
            <div class="theme-meta">{len(items)} images • dominant folder: {dominant_folder}</div>
            <div class="theme-labels">{top_labels}</div>
            <div class="grid">
        """
        for r in sorted(items, key=lambda x: x["relative_path"]):
            safe_file = html.escape(r["file"])
            safe_rel = html.escape(r["relative_path"])
            js_path = json.dumps(r["path"])
            thumb_src = html.escape(r["thumb"])
            block += f"""
                <div class="card">
                    <button class="thumb-btn" onclick='copyPath({js_path})' title="Copy full path to clipboard">
                        <img src="{thumb_src}" loading="lazy" alt="{safe_file}">
                    </button>
                    <div class="info">
                        <div class="file">{safe_file}</div>
                        <div class="path">{safe_rel}</div>
                        <div class="hint">Click thumbnail to copy full path</div>
                    </div>
                </div>
            """
        block += "</div></div>"
        blocks.append(block)

    footer = """
        <div id="toast" class="toast"></div>
    </div>
    </body>
    </html>
    """

    gallery_html = out_dir / f"{year}_gallery.html"
    gallery_html.write_text(html_header + script + "".join(blocks) + footer, encoding="utf-8")
    return gallery_html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Discover photo themes for one year of a photo archive")
    parser.add_argument("root", help="Archive root, e.g. /Volumes/All Photos/Photos")
    parser.add_argument("--year", required=True, help="Year folder to process, e.g. 2008")
    parser.add_argument("--batch-size", type=int, default=16, help="Images per inference batch")
    parser.add_argument("--text-batch-size", type=int, default=64, help="Prompts per text inference batch")
    parser.add_argument("--distance-threshold", type=float, default=0.22, help="Lower = tighter clusters, higher = broader clusters")
    parser.add_argument("--min-cluster-size", type=int, default=6, help="Clusters smaller than this become Miscellaneous")
    parser.add_argument("--max-images", type=int, default=0, help="Optional cap for testing; 0 means no limit")
    parser.add_argument("--exclude-file", default="", help="Optional text file of relative paths to exclude, one per line")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Root folder for theme outputs")
    parser.add_argument("--prompts-file", default=str(DEFAULT_PROMPTS_FILE), help="Text file containing one theme prompt per line")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Embedding model name")
    parser.add_argument("--model-kind", default="auto", choices=["auto", "clip"], help="Model family. Use auto for SigLIP/OpenCLIP-style AutoModel loading.")
    parser.add_argument("--no-cache-embeddings", action="store_true", help="Disable embedding cache")
    parser.add_argument("--force-rebuild-cache", action="store_true", help="Recompute embeddings even if a matching cache exists")
    parser.add_argument("--skip-html", action="store_true", help="Skip HTML gallery creation")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    year_dir = root / str(args.year)
    if not year_dir.exists() or not year_dir.is_dir():
        print(f"Year folder not found: {year_dir}")
        return

    exclude_file = Path(args.exclude_file).expanduser().resolve() if args.exclude_file else None
    excluded_relative_paths = load_exclude_set(exclude_file)

    prompts_file = Path(args.prompts_file).expanduser().resolve() if args.prompts_file else None
    theme_prompts = load_theme_prompts(prompts_file)

    output_root = Path(args.output_root).expanduser().resolve()
    out_dir = output_root / str(args.year)
    thumbs_dir = out_dir / "thumbs"
    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {year_dir}...")
    print(f"Writing output to {out_dir}...")

    total_candidates_before_exclusions = count_candidate_images(year_dir)
    image_paths = list(iter_images(year_dir, root, excluded_relative_paths=excluded_relative_paths))
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    if not image_paths:
        print("No supported images found.")
        return

    print(f"Found {len(image_paths)} images.")

    cache_key = model_cache_key(args.model, args.model_kind)
    embeddings_path, manifest_path, meta_path = cache_paths(out_dir, cache_key)
    current_manifest = manifest_for_paths(image_paths, root, year_dir)

    embeddings = None
    if not args.no_cache_embeddings:
        embeddings = try_load_cache(embeddings_path, manifest_path, current_manifest, args.force_rebuild_cache)

    rows: List[Dict] = []
    failed = []

    if embeddings is None:
        device = pick_device()
        emb_model = EmbeddingModel(args.model, args.model_kind, device)
        print("Building text features...")
        theme_text_features = emb_model.encode_text(theme_prompts, batch_size=args.text_batch_size)

        batch_embeddings = []
        print("Loading images, embedding, and making thumbnails...")
        row_index = 0
        for batch_paths in tqdm(list(chunked(image_paths, args.batch_size)), desc="Embedding"):
            batch_images = []
            batch_indices = []

            for img_path in batch_paths:
                img = get_image_for_ai(img_path)
                if img is None:
                    failed.append(str(img_path))
                    continue

                rel_from_year = img_path.relative_to(year_dir)
                thumb_name = f"{row_index:06d}.jpg"
                thumb_path = thumbs_dir / thumb_name
                make_thumbnail(img, thumb_path)

                rows.append({
                    "row_index": row_index,
                    "file": img_path.name,
                    "path": str(img_path.resolve()),
                    "relative_path": rel_from_year.as_posix(),
                    "archive_relative_path": img_path.relative_to(root).as_posix(),
                    "folder": str(img_path.parent),
                    "thumb": f"thumbs/{thumb_name}",
                })

                batch_images.append(img)
                batch_indices.append(row_index)
                row_index += 1

            if batch_images:
                embs = emb_model.encode_images(batch_images)
                batch_embeddings.append(embs)

        if not rows:
            print("No images could be loaded.")
            return

        embeddings = np.concatenate(batch_embeddings, axis=0).astype("float32")
        if not args.no_cache_embeddings:
            successful_manifest = current_manifest.iloc[[r["row_index"] for r in rows]].copy().reset_index(drop=True)
            # If some files failed, row_index is compact in rows, so rebuild manifest from rows.
            successful_manifest = pd.DataFrame([
                {
                    "row_index": r["row_index"],
                    "path": r["path"],
                    "relative_path": r["relative_path"],
                    "archive_relative_path": r["archive_relative_path"],
                    "file": r["file"],
                    "size": Path(r["path"]).stat().st_size if Path(r["path"]).exists() else -1,
                    "mtime": int(Path(r["path"]).stat().st_mtime) if Path(r["path"]).exists() else -1,
                }
                for r in rows
            ])
            save_cache(
                embeddings,
                successful_manifest,
                embeddings_path,
                manifest_path,
                meta_path,
                {
                    "model": args.model,
                    "model_kind": args.model_kind,
                    "cache_key": cache_key,
                    "year": args.year,
                    "prompt_count": len(theme_prompts),
                    "image_count": len(rows),
                    "failed_count": len(failed),
                },
            )
    else:
        # Cached embeddings: rows come from current manifest, and thumbnails are ensured if missing.
        print("Preparing rows and checking thumbnails from cache manifest...")
        rows = []
        for row_index, img_path in enumerate(tqdm(image_paths, desc="Thumbnails")):
            rel_from_year = img_path.relative_to(year_dir)
            thumb_name = f"{row_index:06d}.jpg"
            thumb_path = thumbs_dir / thumb_name
            if not thumb_path.exists():
                img = get_image_for_ai(img_path)
                if img is None:
                    failed.append(str(img_path))
                    continue
                make_thumbnail(img, thumb_path)
            rows.append({
                "row_index": row_index,
                "file": img_path.name,
                "path": str(img_path.resolve()),
                "relative_path": rel_from_year.as_posix(),
                "archive_relative_path": img_path.relative_to(root).as_posix(),
                "folder": str(img_path.parent),
                "thumb": f"thumbs/{thumb_name}",
            })
        # Text features still depend on prompts/model, but are cheap.
        device = pick_device()
        emb_model = EmbeddingModel(args.model, args.model_kind, device)
        print("Building text features...")
        theme_text_features = emb_model.encode_text(theme_prompts, batch_size=args.text_batch_size)

    # If failures happened during cached thumbnail creation, keep embeddings/rows aligned by rebuilding.
    if len(rows) != len(embeddings):
        raise RuntimeError(
            "Rows and embeddings are out of alignment. Run again with --force-rebuild-cache."
        )

    print("Clustering images into themes...")
    if len(rows) == 1:
        labels = np.array([0])
    else:
        clusterer = make_clusterer(args.distance_threshold)
        labels = clusterer.fit_predict(embeddings)

    for r, label in zip(rows, labels):
        r["cluster_id"] = int(label)

    cluster_to_rows = defaultdict(list)
    for r in rows:
        cluster_to_rows[r["cluster_id"]].append(r)

    small_clusters = {cid for cid, items in cluster_to_rows.items() if len(items) < args.min_cluster_size}
    if small_clusters:
        misc_id = max(cluster_to_rows.keys(), default=-1) + 1
        for r in rows:
            if r["cluster_id"] in small_clusters:
                r["cluster_id"] = misc_id
        cluster_to_rows = defaultdict(list)
        for r in rows:
            cluster_to_rows[r["cluster_id"]].append(r)

    print("Labelling themes...")
    theme_rows = []
    cluster_label_map = {}

    for cluster_id, items in sorted(cluster_to_rows.items(), key=lambda kv: len(kv[1]), reverse=True):
        idxs = [r["row_index"] for r in items]
        cluster_vecs = embeddings[idxs]
        centroid = cluster_vecs.mean(axis=0)
        centroid /= max((centroid @ centroid) ** 0.5, 1e-8)

        sims = theme_text_features @ centroid
        best_indices = sims.argsort()[::-1][:3]
        top_labels = [theme_prompts[i] for i in best_indices]
        top_scores = [float(sims[i]) for i in best_indices]

        dominant_folder = get_dominant_subfolder(items)

        if len(items) < args.min_cluster_size:
            theme_name = "miscellaneous mixed subjects"
            display_theme_name = f"miscellaneous mixed subjects • {int(cluster_id):02d}"
        else:
            theme_name = clean_prompt_label(top_labels[0])
            display_theme_name = build_display_theme_name(top_labels, dominant_folder, cluster_id)

        cluster_label_map[cluster_id] = {
            "theme_name": theme_name,
            "display_theme_name": display_theme_name,
            "top_labels": top_labels,
            "top_scores": top_scores,
            "dominant_folder": dominant_folder,
        }

        reps = choose_representative_indices(items, embeddings, limit=3)
        rep_files = [rows[i]["relative_path"] for i in reps]

        theme_rows.append({
            "cluster_id": cluster_id,
            "theme_name": theme_name,
            "display_theme_name": display_theme_name,
            "image_count": len(items),
            "top_label_1": top_labels[0],
            "top_label_2": top_labels[1],
            "top_label_3": top_labels[2],
            "top_score_1": round(top_scores[0], 4),
            "top_score_2": round(top_scores[1], 4),
            "top_score_3": round(top_scores[2], 4),
            "dominant_folder": dominant_folder,
            "representative_1": rep_files[0] if len(rep_files) > 0 else "",
            "representative_2": rep_files[1] if len(rep_files) > 1 else "",
            "representative_3": rep_files[2] if len(rep_files) > 2 else "",
            "model": args.model,
            "cache_key": cache_key,
        })

    for r in rows:
        meta = cluster_label_map[r["cluster_id"]]
        r["theme_name"] = meta["theme_name"]
        r["display_theme_name"] = meta["display_theme_name"]
        r["theme_top_label_1"] = meta["top_labels"][0]
        r["theme_top_label_2"] = meta["top_labels"][1]
        r["theme_top_label_3"] = meta["top_labels"][2]
        r["theme_top_score_1"] = round(meta["top_scores"][0], 4)
        r["theme_top_score_2"] = round(meta["top_scores"][1], 4)
        r["theme_top_score_3"] = round(meta["top_scores"][2], 4)
        r["dominant_folder"] = meta["dominant_folder"]
        r["model"] = args.model
        r["cache_key"] = cache_key

    image_df = pd.DataFrame([
        {
            "cluster_id": r["cluster_id"],
            "theme_name": r["theme_name"],
            "display_theme_name": r["display_theme_name"],
            "file": r["file"],
            "path": r["path"],
            "relative_path": r["relative_path"],
            "archive_relative_path": r["archive_relative_path"],
            "folder": r["folder"],
            "thumb": r["thumb"],
            "dominant_folder": r["dominant_folder"],
            "theme_top_label_1": r["theme_top_label_1"],
            "theme_top_label_2": r["theme_top_label_2"],
            "theme_top_label_3": r["theme_top_label_3"],
            "theme_top_score_1": r["theme_top_score_1"],
            "theme_top_score_2": r["theme_top_score_2"],
            "theme_top_score_3": r["theme_top_score_3"],
            "model": r["model"],
            "cache_key": r["cache_key"],
        }
        for r in rows
    ]).sort_values(by=["display_theme_name", "relative_path"])

    image_csv = out_dir / f"{args.year}_images.csv"
    image_df.to_csv(image_csv, index=False)

    theme_df = pd.DataFrame(theme_rows).sort_values(by=["image_count", "display_theme_name"], ascending=[False, True])
    theme_csv = out_dir / f"{args.year}_themes.csv"
    theme_df.to_csv(theme_csv, index=False)

    grouped = defaultdict(list)
    for r in rows:
        grouped[r["cluster_id"]].append(r)
    cluster_order = sorted(grouped.keys(), key=lambda cid: len(grouped[cid]), reverse=True)

    gallery_html = None
    if not args.skip_html:
        print("Building HTML gallery...")
        excluded_count = max(total_candidates_before_exclusions - len(rows), 0)
        gallery_html = build_gallery_html(
            out_dir=out_dir,
            year=str(args.year),
            rows=rows,
            grouped=grouped,
            cluster_label_map=cluster_label_map,
            cluster_order=cluster_order,
            total_candidates_before_exclusions=total_candidates_before_exclusions,
            excluded_count=excluded_count,
            model_name=args.model,
            cache_key=cache_key,
        )

    diagnostics = {
        "year": str(args.year),
        "model": args.model,
        "model_kind": args.model_kind,
        "cache_key": cache_key,
        "image_count": int(len(rows)),
        "cluster_count": int(len(cluster_order)),
        "distance_threshold": args.distance_threshold,
        "min_cluster_size": args.min_cluster_size,
        "prompt_count": len(theme_prompts),
        "failed_count": len(failed),
        "outputs": {
            "images_csv": str(image_csv),
            "themes_csv": str(theme_csv),
            "gallery_html": str(gallery_html) if gallery_html else "",
        },
    }
    diagnostics_path = out_dir / f"{args.year}_theme_run_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")

    if failed:
        failed_file = out_dir / f"{args.year}_failed.txt"
        failed_file.write_text("\n".join(failed), encoding="utf-8")
        print(f"{len(failed)} files failed to load. Logged to {failed_file}")

    print()
    print("Done.")
    print(f"Images CSV:       {image_csv}")
    print(f"Themes CSV:       {theme_csv}")
    if gallery_html:
        print(f"Gallery:          {gallery_html}")
    print(f"Diagnostics JSON: {diagnostics_path}")
    if not args.no_cache_embeddings:
        print(f"Embedding cache:  {embeddings_path}")


if __name__ == "__main__":
    main()
