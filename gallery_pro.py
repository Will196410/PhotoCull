import pandas as pd
import imagehash
from PIL import Image
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import html
import json

# Create a thumbnails folder
THUMB_DIR = Path("gallery_thumbs")
THUMB_DIR.mkdir(exist_ok=True)

def get_hash_and_thumb(img_info):
    path_str, score, file_name = img_info
    path = Path(path_str)
    try:
        resolved_path = path.resolve()
        with Image.open(resolved_path) as img:
            h = imagehash.phash(img)
            thumb_path = THUMB_DIR / f"{h}.jpg"
            if not thumb_path.exists():
                thumb_img = img.copy()
                thumb_img.thumbnail((400, 400))
                thumb_img.convert("RGB").save(thumb_path, "JPEG", quality=90)
            return {
                "hash": h,
                "path": str(resolved_path),
                "score": score,
                "file": file_name,
                "thumb": thumb_path.as_posix()
            }
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Input CSV file")
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--limit", type=int, default=1000, help="Top N images to process")
    args = parser.parse_args()

    print(f"Reading {args.csv}...")
    df = pd.read_csv(args.csv)

    if 'score' in df.columns:
        df = df.sort_values(by="score", ascending=False)

    df = df.head(args.limit)
    data_to_process = list(zip(df['path'], df['score'], df['file']))

    print(f"🚀 Processing top {len(data_to_process)} images using multiprocessing...")

    processed_data = []
    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(get_hash_and_thumb, data_to_process), total=len(data_to_process)):
            if result:
                processed_data.append(result)

    unique_images = []
    seen_hashes = []

    print("💎 Deduping...")
    for item in processed_data:
        is_dup = False
        for h in seen_hashes:
            if item['hash'] - h <= args.threshold:
                is_dup = True
                break
        if not is_dup:
            seen_hashes.append(item['hash'])
            unique_images.append(item)

    html_header = """
    <html>
    <head>
    <meta charset="utf-8">
    <title>Photo Gallery</title>
    <style>
        body { background: #111; color: white; font-family: system-ui; margin: 0; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px; padding: 20px; }
        .card { background: #222; border-radius: 10px; overflow: hidden; transition: 0.2s; border: 1px solid #333; }
        .card:hover { border-color: #00ffcc; transform: scale(1.02); }
        .thumb-btn {
            display: block;
            width: 100%;
            padding: 0;
            margin: 0;
            border: 0;
            background: transparent;
            cursor: pointer;
        }
        img { width: 100%; aspect-ratio: 1; object-fit: cover; display: block; background: #000; }
        .info { padding: 10px; font-size: 12px; }
        .score { color: #00ffcc; font-weight: bold; }
        .file { color:#888; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .path { color: #666; font-size: 11px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: 4px; }
        .hint { color: #aaa; font-size: 11px; margin-top: 6px; }
        .toast {
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
        }
        .toast.show { opacity: 1; }
    </style>
    </head>
    <body>
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

    title = f"<h2 style='text-align:center; padding-top:20px;'>AI Results: {len(unique_images)} Unique Images</h2>"
    grid_start = '<div class="grid">'

    cards = ""
    for item in unique_images:
        thumb_src = item['thumb']
        safe_file = html.escape(str(item['file']))
        safe_path = html.escape(str(item['path']))
        js_path = json.dumps(str(item['path']))

        cards += f"""
        <div class="card">
            <button class="thumb-btn" onclick='copyPath({js_path})' title="Copy full path to clipboard">
                <img src="{thumb_src}" loading="lazy" alt="{safe_file}">
            </button>
            <div class="info">
                <div class="score">Score: {item['score']}</div>
                <div class="file">{safe_file}</div>
                <div class="path">{safe_path}</div>
                <div class="hint">Click thumbnail to copy full path</div>
            </div>
        </div>"""

    footer = '<div id="toast" class="toast"></div></body></html>'

    with open("gallery.html", "w", encoding="utf-8") as f:
        f.write(html_header + script + title + grid_start + cards + "</div>" + footer)

    print(f"✅ Gallery ready: {len(unique_images)} images. Run 'open gallery.html'")

if __name__ == "__main__":
    main()
