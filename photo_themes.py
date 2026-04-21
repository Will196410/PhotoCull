import argparse
import io
import json
import html
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import rawpy
import torch
from PIL import Image, ImageFile
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True

RAW_EXTENSIONS = {".dng", ".arw", ".cr2", ".nef", ".orf", ".rw2", ".raf"}
STANDARD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".webp", ".bmp"}
SUPPORTED_EXTENSIONS = RAW_EXTENSIONS | STANDARD_EXTENSIONS

THEME_PROMPTS = [
    "a coastal landscape photograph",
    "a harbour or port scene with boats",
    "a countryside landscape photograph",
    "a woodland or forest scene",
    "a flower or plant close-up photograph",
    "a bird or wildlife photograph",
    "a farm animal photograph",
    "a pet photograph",
    "an old building or historic architecture photograph",
    "a village, town, or street scene photograph",
    "a travel snapshot of a place",
    "a sky, cloud, or weather photograph",
    "a macro or texture detail photograph",
    "a waterside or river scene",
    "a beach or shoreline photograph",
    "a people or group photograph",
    "a portrait of one person",
    "a garden photograph",
    "a tree or foliage photograph",
    "a transport or vehicle photograph",
    "an indoor scene photograph",
    "an abstract visual pattern photograph",
]


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = pick_device()

print(f"Initializing Theme Judge on {DEVICE}...")
MODEL_NAME = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
processor = CLIPProcessor.from_pretrained(MODEL_NAME)


def get_image_for_ai(path: Path):
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


def is_ignored(path: Path):
    s = str(path).lower()
    if path.name.startswith("."):
        return True
    ignored_fragments = ["metadata", "preview", "thumbnail", "sidecar"]
    return any(fragment in s for fragment in ignored_fragments)


def load_exclude_set(exclude_file: Path | None):
    if exclude_file is None:
        return set()

    if not exclude_file.exists():
        print(f"Warning: exclude file not found: {exclude_file}")
        return set()

    lines = exclude_file.read_text(encoding="utf-8").splitlines()
    excluded = {line.strip().replace("\\", "/") for line in lines if line.strip()}
    print(f"Loaded {len(excluded)} excluded paths from {exclude_file}")
    return excluded


def iter_images(root: Path, excluded_relative_paths=None):
    excluded_relative_paths = excluded_relative_paths or set()

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if is_ignored(p):
            continue

        rel = p.relative_to(root).as_posix()
        if rel in excluded_relative_paths:
            continue

        yield p


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def build_text_features(prompts):
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
    with torch.inference_mode():
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def embed_images(images):
    inputs = processor(images=images, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu()


def make_thumbnail(image: Image.Image, thumb_path: Path, size=(360, 360)):
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    if thumb_path.exists():
        return
    thumb_img = image.copy()
    thumb_img.thumbnail(size)
    thumb_img.convert("RGB").save(thumb_path, "JPEG", quality=88)


def choose_representative_indices(cluster_rows, embeddings, limit=3):
    if len(cluster_rows) <= limit:
        return [r["row_index"] for r in cluster_rows]

    import numpy as np

    vecs = np.stack([embeddings[r["row_index"]] for r in cluster_rows], axis=0)
    centroid = vecs.mean(axis=0)
    centroid /= max((centroid @ centroid) ** 0.5, 1e-8)
    sims = vecs @ centroid
    ranked = sorted(zip(cluster_rows, sims), key=lambda x: x[1], reverse=True)
    return [item[0]["row_index"] for item in ranked[:limit]]


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


def get_dominant_subfolder(items):
    counts = Counter()
    for r in items:
        rel = Path(r["relative_path"])
        parts = rel.parts[:-1]
        if parts:
            counts[parts[0]] += 1
        else:
            counts["."] += 1
    return counts.most_common(1)[0][0] if counts else "."


def build_display_theme_name(top_labels, dominant_folder, cluster_id):
    primary = clean_prompt_label(top_labels[0])
    secondary = clean_prompt_label(top_labels[1])

    weak_seconds = {
        "travel snapshot of a place",
        "village, town, or street",
        "indoor",
    }

    parts = [primary]

    if secondary != primary and secondary not in weak_seconds:
        parts.append(secondary)

    if dominant_folder and dominant_folder != ".":
        parts.append(dominant_folder)

    parts.append(f"{int(cluster_id):02d}")

    return " • ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Discover photo themes for one year of a photo archive")
    parser.add_argument("root", help="Archive root, e.g. /Volumes/All Photos/Photos")
    parser.add_argument("--year", required=True, help="Year folder to process, e.g. 2008")
    parser.add_argument("--batch-size", type=int, default=16, help="Images per inference batch")
    parser.add_argument("--distance-threshold", type=float, default=0.22, help="Lower = tighter clusters, higher = broader clusters")
    parser.add_argument("--min-cluster-size", type=int, default=6, help="Clusters smaller than this become Miscellaneous")
    parser.add_argument("--max-images", type=int, default=0, help="Optional cap for testing; 0 means no limit")
    parser.add_argument("--exclude-file", default="", help="Optional text file of relative paths to exclude, one per line")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    year_dir = root / str(args.year)

    if not year_dir.exists() or not year_dir.is_dir():
        print(f"Year folder not found: {year_dir}")
        return

    exclude_file = Path(args.exclude_file).expanduser().resolve() if args.exclude_file else None
    excluded_relative_paths = load_exclude_set(exclude_file)

    out_dir = Path("theme_output") / str(args.year)
    thumbs_dir = out_dir / "thumbs"
    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {year_dir}...")
    image_paths = list(iter_images(year_dir, excluded_relative_paths=excluded_relative_paths))
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    if not image_paths:
        print("No supported images found.")
        return

    print(f"Found {len(image_paths)} images.")

    theme_text_features = build_text_features(THEME_PROMPTS).cpu().numpy()

    rows = []
    failed = []

    print("Loading images, embedding, and making thumbnails...")
    row_index = 0
    for batch_paths in tqdm(list(chunked(image_paths, args.batch_size)), desc="Embedding"):
        batch_images = []
        batch_meta = []

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
            batch_meta.append(row_index)
            row_index += 1

        if not batch_images:
            continue

        batch_embeddings = embed_images(batch_images).numpy()
        for idx, emb in zip(batch_meta, batch_embeddings):
            rows[idx]["embedding"] = emb

    if not rows:
        print("No images could be loaded.")
        return

    import numpy as np

    embeddings = np.stack([r["embedding"] for r in rows], axis=0)

    print("Clustering images into themes...")
    if len(rows) == 1:
        labels = np.array([0])
    else:
        clusterer = AgglomerativeClustering(
            metric="cosine",
            linkage="average",
            distance_threshold=args.distance_threshold,
            n_clusters=None,
        )
        labels = clusterer.fit_predict(embeddings)

    for r, label in zip(rows, labels):
        r["cluster_id"] = int(label)

    cluster_to_rows = defaultdict(list)
    for r in rows:
        cluster_to_rows[r["cluster_id"]].append(r)

    small_clusters = {cid for cid, items in cluster_to_rows.items() if len(items) < args.min_cluster_size}
    next_misc_cluster = max(cluster_to_rows.keys(), default=-1) + 1

    if small_clusters:
        misc_id = next_misc_cluster
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
        top_labels = [THEME_PROMPTS[i] for i in best_indices]

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
            "dominant_folder": dominant_folder,
            "representative_1": rep_files[0] if len(rep_files) > 0 else "",
            "representative_2": rep_files[1] if len(rep_files) > 1 else "",
            "representative_3": rep_files[2] if len(rep_files) > 2 else "",
        })

    for r in rows:
        meta = cluster_label_map[r["cluster_id"]]
        r["theme_name"] = meta["theme_name"]
        r["display_theme_name"] = meta["display_theme_name"]
        r["theme_top_label_1"] = meta["top_labels"][0]
        r["theme_top_label_2"] = meta["top_labels"][1]
        r["theme_top_label_3"] = meta["top_labels"][2]
        r["dominant_folder"] = meta["dominant_folder"]

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
        }
        for r in rows
    ]).sort_values(by=["display_theme_name", "relative_path"])

    image_csv = out_dir / f"{args.year}_images.csv"
    image_df.to_csv(image_csv, index=False)

    theme_df = pd.DataFrame(theme_rows).sort_values(by=["image_count", "display_theme_name"], ascending=[False, True])
    theme_csv = out_dir / f"{args.year}_themes.csv"
    theme_df.to_csv(theme_csv, index=False)

    print("Building HTML gallery...")
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["cluster_id"]].append(r)

    cluster_order = sorted(grouped.keys(), key=lambda cid: len(grouped[cid]), reverse=True)

    excluded_count = len(excluded_relative_paths)
    html_header = f"""
    <html>
    <head>
    <meta charset="utf-8">
    <title>Photo Themes {args.year}</title>
    <style>
        body {{
            background: #111;
            color: white;
            font-family: system-ui, -apple-system, sans-serif;
            margin: 0;
        }}
        .wrap {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1 {{
            margin-top: 8px;
            margin-bottom: 8px;
            font-size: 28px;
        }}
        .summary {{
            color: #bbb;
            margin-bottom: 22px;
        }}
        .theme-block {{
            margin-bottom: 34px;
            padding-bottom: 18px;
            border-bottom: 1px solid #333;
        }}
        .theme-title {{
            font-size: 22px;
            margin-bottom: 6px;
        }}
        .theme-meta {{
            color: #9dd;
            font-size: 13px;
            margin-bottom: 12px;
        }}
        .theme-labels {{
            color: #aaa;
            font-size: 12px;
            margin-bottom: 14px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 14px;
        }}
        .card {{
            background: #222;
            border: 1px solid #333;
            border-radius: 10px;
            overflow: hidden;
        }}
        .thumb-btn {{
            display: block;
            width: 100%;
            padding: 0;
            margin: 0;
            border: 0;
            background: transparent;
            cursor: pointer;
        }}
        img {{
            width: 100%;
            aspect-ratio: 1;
            object-fit: cover;
            display: block;
            background: #000;
        }}
        .info {{
            padding: 10px;
            font-size: 12px;
        }}
        .file {{
            color: #ddd;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .path {{
            color: #777;
            font-size: 11px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            margin-top: 4px;
        }}
        .hint {{
            color: #aaa;
            font-size: 11px;
            margin-top: 6px;
        }}
        .toast {{
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: #00ffcc;
            color: #111;
            padding: 10px 14px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 600;
            box-shadow: 0 6px 20px rgba(0,0,0,0.35);
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.2s ease;
            z-index: 1000;
        }}
        .toast.show {{
            opacity: 1;
        }}
    </style>
    </head>
    <body>
    <div class="wrap">
        <h1>Photo Themes for {args.year}</h1>
        <div class="summary">{len(rows)} images grouped into {len(cluster_order)} theme clusters. Excluded before clustering: {excluded_count}.</div>
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
        if (ok) {
            showToast('Copied path');
        } else {
            showToast('Could not copy path');
        }
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

        block = f"""
        <div class="theme-block">
            <div class="theme-title">{theme_name}</div>
            <div class="theme-meta">{len(items)} images • dominant folder: {dominant_folder}</div>
            <div class="theme-labels">{top_labels}</div>
            <div class="grid">
        """

        for r in sorted(items, key=lambda x: x["relative_path"]):
            safe_file = html.escape(r["file"])
            safe_rel = html.escape(r["relative_path"])
            js_path = json.dumps(r["path"])
            thumb_src = r["thumb"]

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

    gallery_html = out_dir / f"{args.year}_gallery.html"
    with open(gallery_html, "w", encoding="utf-8") as f:
        f.write(html_header + script + "".join(blocks) + footer)

    if failed:
        failed_file = out_dir / f"{args.year}_failed.txt"
        failed_file.write_text("\n".join(failed), encoding="utf-8")
        print(f"{len(failed)} files failed to load. Logged to {failed_file}")

    print()
    print("Done.")
    print(f"Images CSV:  {image_csv}")
    print(f"Themes CSV:  {theme_csv}")
    print(f"Gallery:     {gallery_html}")


if __name__ == "__main__":
    main()
