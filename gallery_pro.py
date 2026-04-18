import pandas as pd
import imagehash
from PIL import Image
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import html

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
            thumb_path = (THUMB_DIR / f"{h}.jpg").resolve()
            if not thumb_path.exists():
                thumb_img = img.copy()
                thumb_img.thumbnail((400, 400))
                thumb_img.convert("RGB").save(thumb_path, "JPEG", quality=90)
            return {
                "hash": h,
                "path": str(resolved_path),
                "score": score,
                "file": file_name,
                "thumb": str(thumb_path)
            }
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Input CSV file")
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--limit", type=int, default=1000, help="Top N images to process")
    args = parser.parse_args()

    # 1. Load, Sort, and Limit based on your parameter
    print(f"Reading {args.csv}...")
    df = pd.read_csv(args.csv)

    # Ensure we sort by score so the 'limit' takes the best ones
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

    # 2. Deduping Logic
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

    # 3. Generate HTML
    html_header = """
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        body { background: #111; color: white; font-family: system-ui; margin: 0; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px; padding: 20px; }
        .card { background: #222; border-radius: 10px; overflow: hidden; transition: 0.2s; border: 1px solid #333; }
        .card:hover { border-color: #00ffcc; transform: scale(1.02); }
        img { width: 100%; aspect-ratio: 1; object-fit: cover; display: block; background: #000; }
        .info { padding: 10px; font-size: 12px; }
        .score { color: #00ffcc; font-weight: bold; }
        .path { color: #666; font-size: 11px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: 4px; }
    </style>
    </head>
    <body>
    """

    title = f"<h2 style='text-align:center; padding-top:20px;'>AI Results: {len(unique_images)} Unique Images</h2>"
    grid_start = '<div class="grid">'

    cards = ""
    for item in unique_images:
        orig_uri = Path(item['path']).as_uri()
        thumb_uri = Path(item['thumb']).as_uri()
        safe_file = html.escape(str(item['file']))
        safe_path = html.escape(str(item['path']))

        cards += f"""
        <div class="card">
            <a href="{orig_uri}" target="_blank">
                <img src="{thumb_uri}" loading="lazy" alt="{safe_file}">
            </a>
            <div class="info">
                <div class="score">Score: {item['score']}</div>
                <div style="color:#888; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{safe_file}</div>
                <div class="path">{safe_path}</div>
            </div>
        </div>"""

    footer = "</div></body></html>"

    with open("gallery.html", "w", encoding="utf-8") as f:
        f.write(html_header + title + grid_start + cards + footer)

    print(f"✅ Gallery ready: {len(unique_images)} images. Run 'open gallery.html'")

if __name__ == "__main__":
    main()
