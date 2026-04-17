import pandas as pd
import imagehash
from PIL import Image
import shutil
from pathlib import Path
from tqdm import tqdm

def is_too_similar(new_hash, existing_hashes, threshold=10):
    """
    Checks if the new_hash is within the 'similarity' distance of any existing hash.
    Threshold Guide:
    0-2: Exact same photo
    5-10: Near-duplicate (burst shot, slightly different pose)
    12-20: Different photos, similar composition
    """
    for h in existing_hashes:
        if (new_hash - h) <= threshold:
            return True
    return False

def process_duplicates(csv_path, output_dir="./best_shots", top_n=50):
    df = pd.read_csv(csv_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Sort by score so the AI's favorites get checked (and saved) first
    df = df.sort_values(by="score", ascending=False)
    
    unique_hashes = []
    copied_count = 0

    print(f"Filtering {csv_path} for unique compositions...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        if copied_count >= top_n:
            break
            
        path = row['path']
        try:
            with Image.open(path) as img:
                current_hash = imagehash.phash(img)
            
            # Check against our 'Winners' list
            if not is_too_similar(current_hash, unique_hashes, threshold=10):
                unique_hashes.append(current_hash)
                
                # Copy the winner
                src = Path(path)
                dest = Path(output_dir) / src.name
                shutil.copy2(src, dest)
                copied_count += 1
        except Exception:
            continue

    print(f"\n✅ Done! Copied {copied_count} unique candidates to {output_dir}")

if __name__ == "__main__":
    # Use 'art_candidates.csv' or 'stock_candidates.csv'
    process_duplicates("art_candidates.csv", top_n=500)
