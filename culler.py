import torch
import rawpy
import io
import argparse
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import pandas as pd

# 1. Setup Apple Silicon Acceleration
device = "mps" if torch.backends.mps.is_available() else "cpu"

# 2. Load the AI Model
print(f"Initializing AI Judge on {device}...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_for_ai(path):
    """Handles DNG, RAW, and Standard files efficiently."""
    ext = path.suffix.lower()
    try:
        if ext in ['.dng', '.arw', '.cr2', '.nef', '.orf']:
            with rawpy.imread(str(path)) as raw:
                try:
                    thumb = raw.extract_thumb()
                    if thumb.format == rawpy.ThumbFormat.JPEG:
                        return Image.open(io.BytesIO(thumb.data))
                except:
                    pass
                return Image.fromarray(raw.postprocess(use_camera_wb=True, half_size=True, bright=1.2))
        return Image.open(path).convert("RGB")
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="AI Photo Culler for Fine Art Prints")
    parser.add_argument("path", help="Path to your SSD/Folder")
    # Added 'animal' and 'landscape' to the choices
    parser.add_argument("--mode", choices=['art', 'stock', 'animal', 'landscape'], required=True, help="Selection criteria")
    args = parser.parse_args()

    # Define the 'Positive' and 'Negative' prompts based on mode
    if args.mode == 'art':
        prompts = ["a masterpiece fine art photograph with moody lighting and composition", 
                   "a blurry low quality accidental snapshot"]
        outfile = "art_candidates.csv"
    elif args.mode == 'stock':
        prompts = ["a clean commercial stock photo with copy space and sharp focus", 
                   "a grainy snapshot with distracting background and poor framing"]
        outfile = "stock_candidates.csv"
    elif args.mode == 'animal':
        prompts = ["a majestic wildlife portrait of an animal, sharp eyes, national geographic style", 
                   "a blurry, distant animal, cluttered cage bars, any text, or domestic mess"]
        outfile = "animal_candidates.csv"
    elif args.mode == 'landscape':
        prompts = ["a breathtaking landscape photograph, epic scale, golden hour, fine art print", 
                   "a boring gray sky, flat lighting, window reflection, or cluttered power lines"]
        outfile = "landscape_candidates.csv"

    # Scan for files
    extensions = {'.jpg', '.jpeg', '.png', '.heic', '.dng', '.arw', '.cr2', '.nef'}
    image_paths = [p for p in Path(args.path).rglob('*') if p.suffix.lower() in extensions]
    
    if not image_paths:
        print("No images found. Check your path and permissions.")
        return

    print(f"Mode: {args.mode.upper()} | Found {len(image_paths)} images. Starting scan...")
    
    results = []
    for img_path in tqdm(image_paths):
        if img_path.name.startswith('.') or 'metadata' in str(img_path).lower():
            continue
            
        img = get_image_for_ai(img_path)
        if img is None: continue

        inputs = processor(text=prompts, images=img, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Higher probability for index 0 (the positive prompt) means a higher score
            probs = outputs.logits_per_image.softmax(dim=1)
            score = probs[0][0].item()

        results.append({
            "file": img_path.name,
            "path": str(img_path),
            "score": round(score, 4)
        })

    # Save and sort
    df = pd.DataFrame(results).sort_values(by="score", ascending=False)
    df.to_csv(outfile, index=False)
    print(f"\n✅ Done! Top 10 {args.mode} candidates:")
    print(df[['file', 'score']].head(10))

if __name__ == "__main__":
    main()
