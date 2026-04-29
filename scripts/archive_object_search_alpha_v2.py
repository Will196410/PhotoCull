#!/usr/bin/env python3
"""
archive_object_search_alpha_v2.py

Simple CLIP-based object/condition search across your photo archive.

Default input:
    theme_output/master_gallery/master_gallery_images.csv

Default output:
    theme_output/archive_object_search_alpha/
        index.html
        search_results.csv
        cache/image_embeddings.npy
        cache/image_manifest.csv
        queries/*.html

Examples:
    python archive_object_search_alpha.py . --queries "cars,bicycles,men,women,tents"
    python archive_object_search_alpha.py . --queries "night,winter,summer,boats,dogs" --top-k 200
    python archive_object_search_alpha.py . --queries-file object_queries.txt --top-k 300

Notes:
- This does not use your master categories.
- It uses CLIP similarity, so searches are approximate discovery searches, not exact object detection.
- The first run builds an image embedding cache and can take a while.
- Later runs reuse the cache unless you pass --rebuild-cache.
"""

import argparse
import html
import io
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

RAW_EXTENSIONS = {".dng", ".arw", ".cr2", ".nef", ".orf", ".rw2", ".raf"}
STANDARD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".webp", ".bmp"}
SUPPORTED_EXTENSIONS = RAW_EXTENSIONS | STANDARD_EXTENSIONS

DEFAULT_INPUT = "theme_output/master_gallery/master_gallery_images.csv"
DEFAULT_OUTPUT = "theme_output/archive_object_search_alpha"
DEFAULT_MODEL = "openai/clip-vit-base-patch32"


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def safe_str(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def slugify(value: str) -> str:
    text = safe_str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "query"


def split_queries(value: str) -> List[str]:
    if not value:
        return []
    return [q.strip() for q in value.split(",") if q.strip()]


def resolve_path(project_root: Path, value: str) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = project_root / p
    return p.resolve()


def choose_device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Image loading
# -----------------------------------------------------------------------------

def load_image(path: Path) -> Optional[Image.Image]:
    suffix = path.suffix.lower()
    try:
        if suffix in RAW_EXTENSIONS:
            return load_raw_image(path)
        img = Image.open(path)
        return img.convert("RGB")
    except Exception:
        return None


def load_raw_image(path: Path) -> Optional[Image.Image]:
    try:
        import rawpy
    except Exception:
        return None

    try:
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
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Manifest and cache
# -----------------------------------------------------------------------------

def load_source_dataframe(input_csv: Path, limit: int = 0) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if "path" not in df.columns:
        raise RuntimeError("Input CSV must contain a 'path' column with archive paths.")

    df = df.copy()
    df["path"] = df["path"].apply(safe_str)
    df = df[df["path"].str.len() > 0].copy()
    df = df.drop_duplicates(subset=["path"], keep="first")

    if limit and limit > 0:
        df = df.head(limit).copy()

    df["_path_exists"] = df["path"].apply(lambda p: Path(p).exists())
    missing = int((~df["_path_exists"]).sum())
    if missing:
        print(f"Warning: {missing} paths in the CSV do not currently exist and will be skipped.")
    df = df[df["_path_exists"]].copy()

    df["_suffix_ok"] = df["path"].apply(lambda p: Path(p).suffix.lower() in SUPPORTED_EXTENSIONS)
    df = df[df["_suffix_ok"]].copy()

    if df.empty:
        raise RuntimeError("No readable image paths found in the input CSV.")

    return df.reset_index(drop=True)


def cache_is_compatible(manifest_csv: Path, embeddings_npy: Path, source_df: pd.DataFrame) -> bool:
    if not manifest_csv.exists() or not embeddings_npy.exists():
        return False

    try:
        manifest = pd.read_csv(manifest_csv)
        emb = np.load(embeddings_npy, mmap_mode="r")
    except Exception:
        return False

    if len(manifest) != len(source_df):
        return False
    if emb.shape[0] != len(source_df):
        return False

    old_paths = manifest["path"].astype(str).tolist() if "path" in manifest.columns else []
    new_paths = source_df["path"].astype(str).tolist()
    return old_paths == new_paths


def build_or_load_cache(
    source_df: pd.DataFrame,
    cache_dir: Path,
    model_name: str,
    batch_size: int,
    rebuild_cache: bool,
) -> Tuple[pd.DataFrame, np.ndarray, object, object, object]:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = cache_dir / "image_manifest.csv"
    embeddings_npy = cache_dir / "image_embeddings.npy"

    device = choose_device()
    print(f"Using device: {device}")
    print(f"Loading model: {model_name}")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    if not rebuild_cache and cache_is_compatible(manifest_csv, embeddings_npy, source_df):
        print(f"Using existing image embedding cache: {embeddings_npy}")
        manifest = pd.read_csv(manifest_csv)
        embeddings = np.load(embeddings_npy)
        return manifest, embeddings, model, processor, device

    print("Building image embedding cache...")
    rows: List[dict] = []
    embeddings: List[np.ndarray] = []

    batch_images: List[Image.Image] = []
    batch_rows: List[dict] = []

    def flush_batch():
        nonlocal batch_images, batch_rows, embeddings, rows
        if not batch_images:
            return
        inputs = processor(images=batch_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        arr = feats.detach().cpu().numpy().astype("float32")
        embeddings.extend(list(arr))
        rows.extend(batch_rows)
        batch_images = []
        batch_rows = []

    for _, row in tqdm(source_df.iterrows(), total=len(source_df), desc="Embedding images"):
        path = Path(safe_str(row.get("path", "")))
        img = load_image(path)
        if img is None:
            continue

        row_dict = row.to_dict()
        batch_images.append(img)
        batch_rows.append(row_dict)

        if len(batch_images) >= batch_size:
            flush_batch()

    flush_batch()

    if not embeddings:
        raise RuntimeError("No image embeddings could be created.")

    manifest = pd.DataFrame(rows).reset_index(drop=True)
    embedding_array = np.vstack(embeddings).astype("float32")

    manifest.to_csv(manifest_csv, index=False)
    np.save(embeddings_npy, embedding_array)

    print(f"Cached manifest:   {manifest_csv}")
    print(f"Cached embeddings: {embeddings_npy}")
    print(f"Embedded images:   {len(manifest)}")

    return manifest, embedding_array, model, processor, device


# -----------------------------------------------------------------------------
# Query scoring
# -----------------------------------------------------------------------------

def make_prompt(query: str) -> str:
    q = query.strip()
    lower = q.lower()

    # Mild prompt shaping for common condition searches.
    condition_phrases = {
        "night": "a photograph taken at night",
        "day": "a photograph taken during the day",
        "daytime": "a daytime photograph",
        "winter": "a winter photograph with cold seasonal conditions",
        "summer": "a summer photograph with warm seasonal conditions",
        "spring": "a spring photograph",
        "autumn": "an autumn photograph",
        "fall": "an autumn photograph",
        "snow": "a photograph with snow",
        "rain": "a photograph with rain",
        "fog": "a foggy photograph",
        "mist": "a misty photograph",
        "sunset": "a sunset photograph",
        "sunrise": "a sunrise photograph",
    }
    if lower in condition_phrases:
        return condition_phrases[lower]

    return f"a photograph of {q}"


def score_queries(
    queries: List[str],
    embeddings: np.ndarray,
    model,
    processor,
    device,
) -> Dict[str, np.ndarray]:
    import torch

    prompts = [make_prompt(q) for q in queries]
    inputs = processor(text=prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        text_feats = model.get_text_features(**inputs)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    text_arr = text_feats.detach().cpu().numpy().astype("float32")

    scores = embeddings @ text_arr.T
    return {queries[i]: scores[:, i] for i in range(len(queries))}


# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------

def make_thumb_src(row: pd.Series, thumb_prefix: str = "../") -> str:
    year = safe_str(row.get("year", "")).strip()
    thumb = safe_str(row.get("thumb", "")).strip()
    if year and thumb:
        # The object-search output sits under theme_output/archive_object_search_alpha/.
        # Dashboard-level pages need ../YEAR/thumbs/file.jpg.
        # Query pages under queries/ need ../../YEAR/thumbs/file.jpg.
        return f"{thumb_prefix}{html.escape(year)}/{html.escape(thumb)}"
    return ""


def card_html(row: pd.Series, score: float, thumb_prefix: str = "../") -> str:
    file_name = html.escape(safe_str(row.get("file", "")) or Path(safe_str(row.get("path", ""))).name)
    rel_path = html.escape(safe_str(row.get("archive_relative_path") or row.get("relative_path") or row.get("path") or ""))
    full_path = safe_str(row.get("path", ""))
    path_attr = html.escape(full_path, quote=True)
    primary = html.escape(safe_str(row.get("primary_master_category", "")))
    theme = html.escape(safe_str(row.get("display_theme_name") or row.get("theme_name") or ""))
    year = html.escape(safe_str(row.get("year", "")))
    score_text = html.escape(f"{score:.4f}")

    thumb_src = make_thumb_src(row, thumb_prefix=thumb_prefix)
    if thumb_src:
        thumb = f'<img src="{thumb_src}" alt="{file_name}" loading="lazy">'
    else:
        thumb = f'<div class="thumb-fallback">{year or "No thumb"}</div>'

    return f"""
    <article class="card" data-path="{path_attr}">
        <button class="thumb-btn" data-copy-path="{path_attr}" title="Copy archive path">
            {thumb}
        </button>
        <div class="card-body">
            <div class="card-title">{file_name}</div>
            <div class="path" title="{rel_path}">{rel_path}</div>
            <div class="chips">
                <span class="chip score">score {score_text}</span>
                <span class="chip">{primary or "uncategorised"}</span>
                <span class="chip">{year}</span>
            </div>
            <div class="meta"><strong>Theme:</strong> {theme or "—"}</div>
        </div>
    </article>
    """


def common_css() -> str:
    return """
:root { --bg:#101214; --panel:#181b1f; --panel2:#20242a; --text:#f2f2f2; --muted:#b8bec8; --line:#3b414b; --accent:#8ee7ff; --warn:#ffd166; }
* { box-sizing:border-box; }
html { scroll-behavior:smooth; }
body { margin:0; background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
.wrap { max-width:1800px; margin:0 auto; padding:20px; }
.top-panel { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:18px; margin-bottom:24px; }
h1 { margin:0 0 8px; font-size:30px; }
a { color:var(--accent); text-decoration:none; }
a:hover { text-decoration:underline; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(245px,1fr)); gap:14px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:14px; overflow:hidden; box-shadow:0 8px 18px rgba(0,0,0,0.2); }
.thumb-btn { width:100%; aspect-ratio:1; display:block; border:0; padding:0; margin:0; background:#050607; cursor:pointer; }
.thumb-btn img { width:100%; height:100%; display:block; object-fit:cover; }
.thumb-fallback { width:100%; height:100%; display:flex; align-items:center; justify-content:center; color:var(--muted); font-size:22px; }
.card-body { padding:11px; font-size:12px; }
.card-title { font-weight:800; font-size:13px; margin-bottom:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.path { color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-bottom:7px; }
.chips { display:flex; flex-wrap:wrap; gap:5px; margin-bottom:8px; }
.chip { border:1px solid var(--line); background:#11161b; color:var(--muted); border-radius:999px; padding:3px 7px; font-size:11px; }
.chip.score { color:#111; background:var(--accent); border-color:var(--accent); font-weight:800; }
.meta { color:var(--muted); margin-top:5px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.summary-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:10px; margin:16px 0; }
.summary-card { background:var(--panel2); border:1px solid var(--line); border-radius:12px; padding:12px; }
.summary-card .num { font-size:24px; font-weight:800; }
.summary-card .label { color:var(--muted); font-size:13px; }
.query-list { columns:2; padding-left:20px; }
.query-list li { break-inside:avoid; margin:6px 0; }
.toolbar { display:flex; flex-wrap:wrap; gap:10px; margin:14px 0; }
.toolbar button { border:1px solid var(--line); background:#272c33; color:var(--text); border-radius:10px; padding:9px 11px; cursor:pointer; font-weight:700; }
textarea.export { width:100%; min-height:90px; background:#08090a; color:var(--text); border:1px solid var(--line); border-radius:12px; padding:12px; margin-top:12px; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12px; }
.toast { position:fixed; left:50%; bottom:22px; transform:translateX(-50%); background:var(--accent); color:#111; border-radius:999px; padding:11px 16px; font-weight:800; opacity:0; pointer-events:none; transition:opacity .18s ease; z-index:9999; }
.toast.show { opacity:1; }
"""


def common_js() -> str:
    return """
(function () {
    "use strict";
    function showToast(message) {
        const toast = document.getElementById("toast");
        if (!toast) return;
        toast.textContent = message;
        toast.classList.add("show");
        clearTimeout(window.__objectSearchToastTimer);
        window.__objectSearchToastTimer = setTimeout(() => toast.classList.remove("show"), 1500);
    }
    async function copyText(text) {
        const value = String(text || "");
        try {
            if (navigator.clipboard && window.isSecureContext) {
                await navigator.clipboard.writeText(value);
                return true;
            }
        } catch (e) {}
        try {
            const ta = document.createElement("textarea");
            ta.value = value;
            ta.setAttribute("readonly", "");
            ta.style.position = "fixed";
            ta.style.left = "-9999px";
            ta.style.top = "0";
            document.body.appendChild(ta);
            ta.focus();
            ta.select();
            ta.setSelectionRange(0, ta.value.length);
            const ok = document.execCommand("copy");
            document.body.removeChild(ta);
            return !!ok;
        } catch (e) {
            return false;
        }
    }
    function bindCopyButtons() {
        document.querySelectorAll("[data-copy-path]").forEach(btn => {
            btn.addEventListener("click", function () {
                const path = btn.getAttribute("data-copy-path") || "";
                copyText(path).then(ok => {
                    const box = document.getElementById("exportBox");
                    if (!ok && box) box.value = path;
                    showToast(ok ? "Copied archive path" : "Path shown in export box");
                });
            });
        });
    }
    function bindCopyPagePaths() {
        const btn = document.getElementById("copyPagePathsBtn");
        if (!btn) return;
        btn.addEventListener("click", function () {
            const paths = Array.from(document.querySelectorAll(".card"))
                .map(card => card.getAttribute("data-path") || "")
                .filter(Boolean);
            const text = paths.join("\n");
            const box = document.getElementById("exportBox");
            if (box) box.value = text;
            copyText(text).then(ok => showToast(ok ? "Copied " + paths.length + " paths" : "Paths shown in export box"));
        });
    }
    bindCopyButtons();
    bindCopyPagePaths();
    showToast("Search page ready");
}());
"""


def write_query_page(
    output_path: Path,
    query: str,
    result_df: pd.DataFrame,
    score_col: str,
    top_k: int,
) -> None:
    cards = []
    for _, row in result_df.iterrows():
        cards.append(card_html(row, float(row[score_col]), thumb_prefix="../../"))

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(query)} — Archive Object Search</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{common_css()}</style>
</head>
<body>
<div class="wrap">
    <div class="top-panel">
        <h1>{html.escape(query)}</h1>
        <p><a href="../index.html">Back to object search dashboard</a></p>
        <p>Top {len(result_df)} results. Query prompt: <strong>{html.escape(make_prompt(query))}</strong></p>
        <div class="toolbar"><button id="copyPagePathsBtn">Copy paths on this page</button></div>
        <textarea class="export" id="exportBox" placeholder="Copied paths will appear here if clipboard is blocked."></textarea>
    </div>
    <div class="grid">{''.join(cards)}</div>
</div>
<div id="toast" class="toast"></div>
<script>{common_js()}</script>
</body>
</html>
"""
    output_path.write_text(page, encoding="utf-8")


def write_dashboard(
    output_path: Path,
    queries: List[str],
    result_counts: Dict[str, int],
    total_images: int,
    top_k: int,
) -> None:
    items = []
    for q in queries:
        slug = slugify(q)
        items.append(
            f'<li><a href="queries/{slug}.html">{html.escape(q)}</a> '
            f'<span style="color:var(--muted)">({result_counts.get(q, 0)} shown)</span></li>'
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Archive Object Search Alpha</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{common_css()}</style>
</head>
<body>
<div class="wrap">
    <div class="top-panel">
        <h1>Archive Object Search <span style="background:var(--warn); color:#111; border-radius:999px; padding:3px 8px; font-size:13px;">ALPHA</span></h1>
        <p>Simple CLIP search across the archive. No master categories required.</p>
        <div class="summary-grid">
            <div class="summary-card"><div class="num">{total_images}</div><div class="label">images searched</div></div>
            <div class="summary-card"><div class="num">{len(queries)}</div><div class="label">queries</div></div>
            <div class="summary-card"><div class="num">{top_k}</div><div class="label">max results per query</div></div>
        </div>
        <h2>Queries</h2>
        <ul class="query-list">{''.join(items)}</ul>
        <p>Full combined results: <code>search_results.csv</code></p>
    </div>
</div>
<div id="toast" class="toast"></div>
<script>{common_js()}</script>
</body>
</html>
"""
    output_path.write_text(page, encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CLIP object/condition search across your photo archive")
    parser.add_argument("project_root", help="Project root containing theme_output/")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input master_gallery_images.csv path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output folder")
    parser.add_argument("--queries", default="", help="Comma-separated queries, e.g. cars,bicycles,men,women,tents")
    parser.add_argument("--queries-file", default="", help="Optional text file, one query per line")
    parser.add_argument("--top-k", type=int, default=200, help="Top results per query")
    parser.add_argument("--min-score", type=float, default=0.0, help="Optional minimum CLIP similarity score")
    parser.add_argument("--batch-size", type=int, default=16, help="Image embedding batch size")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="CLIP model name")
    parser.add_argument("--limit", type=int, default=0, help="Limit images for testing")
    parser.add_argument("--rebuild-cache", action="store_true", help="Rebuild image embedding cache")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    input_csv = resolve_path(project_root, args.input)
    output_root = resolve_path(project_root, args.output)
    cache_dir = output_root / "cache"
    query_dir = output_root / "queries"
    output_root.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)

    queries = split_queries(args.queries)
    if args.queries_file:
        qfile = resolve_path(project_root, args.queries_file)
        lines = qfile.read_text(encoding="utf-8").splitlines()
        queries.extend([line.strip() for line in lines if line.strip() and not line.strip().startswith("#")])

    # A useful default starter set.
    if not queries:
        queries = [
            "cars", "bicycles", "men", "women", "children", "dogs", "cats",
            "tents", "boats", "birds", "cows", "horses", "night", "winter", "summer",
        ]

    # Deduplicate while preserving order.
    seen = set()
    queries = [q for q in queries if not (q.lower() in seen or seen.add(q.lower()))]

    print(f"Input CSV:  {input_csv}")
    print(f"Output:     {output_root}")
    print(f"Queries:    {', '.join(queries)}")

    source_df = load_source_dataframe(input_csv, limit=args.limit)
    manifest, embeddings, model, processor, device = build_or_load_cache(
        source_df=source_df,
        cache_dir=cache_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        rebuild_cache=args.rebuild_cache,
    )

    score_map = score_queries(queries, embeddings, model, processor, device)

    all_rows = []
    result_counts: Dict[str, int] = {}

    for query in queries:
        scores = score_map[query]
        order = np.argsort(-scores)
        if args.min_score > 0:
            order = [int(i) for i in order if float(scores[i]) >= args.min_score]
        else:
            order = [int(i) for i in order]
        order = order[: args.top_k]

        score_col = f"score_{slugify(query)}"
        result_df = manifest.iloc[order].copy()
        result_df[score_col] = [float(scores[i]) for i in order]
        result_df.insert(0, "search_query", query)
        result_df.insert(1, "search_prompt", make_prompt(query))
        result_df.insert(2, "search_score", result_df[score_col])

        result_counts[query] = len(result_df)
        all_rows.append(result_df)

        write_query_page(
            output_path=query_dir / f"{slugify(query)}.html",
            query=query,
            result_df=result_df,
            score_col=score_col,
            top_k=args.top_k,
        )

    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    combined_csv = output_root / "search_results.csv"
    combined.to_csv(combined_csv, index=False)

    metrics = {
        "total_images_searched": int(len(manifest)),
        "queries": queries,
        "top_k": int(args.top_k),
        "min_score": float(args.min_score),
        "result_counts": result_counts,
    }
    (output_root / "object_search_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    write_dashboard(output_root / "index.html", queries, result_counts, total_images=len(manifest), top_k=args.top_k)

    print()
    print("Archive Object Search Alpha built.")
    print(f"Dashboard:       {output_root / 'index.html'}")
    print(f"Combined CSV:    {combined_csv}")
    print(f"Query pages:     {query_dir}")
    print()
    print("Open the dashboard in your browser:")
    print(f"open {output_root / 'index.html'}")
    print()
    print("For best thumbnail loading, serve from theme_output, e.g.:")
    print("cd theme_output && python -m http.server 8000")
    print("Then open http://localhost:8000/archive_object_search_alpha/")


if __name__ == "__main__":
    main()
