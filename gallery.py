import pandas as pd
import imagehash
from PIL import Image
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Create a thumbnails folder to keep the gallery light
THUMB_DIR = Path("gallery_thumbs")
THUMB_DIR.mkdir(exist_ok=True)

def get_hash_and_thumb(img_info):
    """Worker function to process one image."""
    path_str, score, file_name = img_info
    path = Path(path_str)
    try:
        with Image.open(path) as img:
            # 1. Generate Hash
            h = imagehash.phash(img)
            
            # 2. Create small thumbnail for the HTML
            thumb_path = THUMB_DIR / f"{h}.jpg"
            if not thumb_path.exists():
                img.thumbnail((400, 400))
                img.convert("RGB").save(thumb_path, "JPEG")
            
            return {"hash": h, "path": path_str, "score": score, "file": file_name, "thumb": str(thumb_path)}
    except:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Input CSV from the cull script")
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--limit", type=int, default=1000, help="Top N images to consider")
    args = parser.parse_args()

    df = pd.read_csv(args.csv).head(args.limit)
    data_to_process = list(zip(df['path'], df['score'], df['file']))

    print(f"🚀 Processing top {args.limit} images using multiprocessing...")
    
    processed_data = []
    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(get_hash_and_thumb, data_to_process), total=len(data_to_process)):
            if result:
                processed_data.append(result)

    # Deduping Logic
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

    # Generate HTML with Lazy Loading
    html_start = """
    <html><head><style>
        body {{ background: #111; color: white; font-family: system-ui; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px; padding: 20px; }}
        .card {{ background: #222; border-radius: 10px; overflow: hidden; transition: 0.2s; border: 1px solid #333; }}
        .card:hover {{ border-color: #00ffcc; transform: scale(1.02); }}
        img {{ width: 100%; aspect-ratio: 1; object-fit: cover; display: block; }}
        .info {{ padding: 10px; font-size: 12px; }}
        .score {{ color: #00ffcc; font-weight: bold; }}
    </style></head><body>
    <h2 style='text-align:center'>AI Culling Results: {count} Unique Images</h2>
    <div class="grid">
    """
    
    cards = ""
    for item in unique_images:
        cards += f"""
        <div class="card">
            <a href="file://{item['path']}" target="_blank">
                <img src="{item['thumb']}" loading="lazy">
            </a>
            <div class="info">
                <div class="score">Score: {item['score']}</div>
                <div style="color:#888; overflow:hidden; text-overflow:ellipsis;">{item['file']}</div>
            </div>
        </div>"""

    with open("gallery.html", "w") as f:
        f.write(html_start.format(count=len(unique_images)) + cards + "</div></body></html>")

    print(f"✅ Gallery ready: {len(unique_images)} images. Run 'open gallery.html'")

if __name__ == "__main__":
    main()