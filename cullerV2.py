import torch
import rawpy
import io
import argparse
from pathlib import Path
from PIL import Image, ImageFile
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 1. Setup acceleration
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# 2. Load model
print(f"Initializing AI Judge on {device}...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model.eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

RAW_EXTENSIONS = {'.dng', '.arw', '.cr2', '.nef', '.orf', '.rw2', '.raf'}
STANDARD_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.tif', '.tiff', '.webp', '.bmp'}
SUPPORTED_EXTENSIONS = RAW_EXTENSIONS | STANDARD_EXTENSIONS


def get_image_for_ai(path):
    """Load RAW or standard image, returning RGB PIL image or None."""
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
                    output_bps=8
                )
                return Image.fromarray(rgb).convert("RGB")

        return Image.open(path).convert("RGB")
    except Exception:
        return None


def is_ignored(path):
    s = str(path).lower()
    if path.name.startswith('.'):
        return True
    ignored_fragments = ['metadata', 'preview', 'thumbnail', 'sidecar']
    return any(fragment in s for fragment in ignored_fragments)


def iter_images(root):
    for p in Path(root).rglob('*'):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if is_ignored(p):
            continue
        yield p


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def build_text_features(prompts):
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def score_images(images, text_features):
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ text_features.T
    return logits


def soft_pair_score(logits_row, pos_idx, neg_idx):
    pair = torch.tensor([logits_row[pos_idx].item(), logits_row[neg_idx].item()])
    probs = torch.softmax(pair, dim=0)
    return probs[0].item()


def main():
    parser = argparse.ArgumentParser(description="AI Photo Culler for Fine Art Prints / Stock / Wildlife / Landscape")
    parser.add_argument("path", help="Path to SSD/folder")
    parser.add_argument("--mode", choices=['art', 'stock', 'animal', 'group', 'landscape'], required=True, help="Selection criteria")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--top", type=int, default=10, help="How many top results to print")
    args = parser.parse_args()

    if args.mode == 'stock':
        prompts = [
            # Positive prompts
            "a professional commercial stock photo suitable for advertising",
            "a clean modern stock photograph with sharp main subject and usable copy space",
            "a commercially useful image with simple composition and broad business or editorial appeal",
            "a polished high quality photograph with clean background and natural color",
            "a licensable stock image with no obvious branding or legal problems",

            # Negative prompts
            "a blurry low quality snapshot with poor focus and technical defects",
            "a cluttered distracting photo with messy background and weak composition",
            "an image with logos, trademarks, brand names, labels, or copyrighted artwork",
            "a private or restricted property image likely to need a property release",
            "a photo with harsh noise, oversharpening, artifacts, or bad exposure"
        ]
        outfile = "stock_candidates.csv"

        pairs = {
            "commercial_appeal": (0, 5),
            "copyspace_cleanliness": (1, 6),
            "broad_usability": (2, 6),
            "technical_quality": (3, 9),
            "release_friendliness": (4, 7)
        }

    elif args.mode == 'art':
        prompts = [
            "a strong fine art photograph with mood, atmosphere, and compelling composition",
            "a gallery-worthy photographic print with emotional impact and elegant light",
            "a photograph with distinctive artistic vision and print-worthy tonal quality",

            "a dull accidental snapshot with weak composition",
            "a technically poor blurry image with no artistic impact",
            "a cluttered literal record shot with no atmosphere"
        ]
        outfile = "art_candidates.csv"

        pairs = {
            "artistic_strength": (0, 3),
            "print_appeal": (1, 4),
            "distinctiveness": (2, 5)
        }

    elif args.mode == 'animal':
        prompts = [
            "a strong wildlife photograph with sharp eyes and clear animal subject",
            "a close, detailed, professional animal portrait",
            "a nature photograph with excellent animal subject isolation",

            "a blurry distant animal photograph",
            "an animal obscured by clutter, fences, cage bars, or background distractions",
            "a domestic pet snapshot with weak composition"
        ]
        outfile = "animal_candidates.csv"

        pairs = {
            "subject_strength": (0, 3),
            "detail_closeness": (1, 3),
            "clean_isolation": (2, 4)
        }

    elif args.mode == 'group':
        prompts = [
            "a well-composed group photograph with multiple people clearly as the main subject",
            "a professional group portrait with good balance and clear subjects",

            "a single person portrait",
            "a photo with no people",
            "a chaotic crowd scene with no clear group subject"
        ]
        outfile = "group_candidates.csv"

        pairs = {
            "group_presence": (0, 2),
            "group_clarity": (1, 4)
        }

    elif args.mode == 'landscape':
        prompts = [
            "a striking landscape photograph with dramatic light and strong composition",
            "a print-worthy scenic landscape with depth, atmosphere, and visual impact",
            "a beautiful natural landscape image with clean horizon and no distractions",

            "a dull flat landscape with weak light",
            "a cluttered landscape with wires, poles, reflections, or distractions",
            "a casual travel snapshot with no strong scenic impact"
        ]
        outfile = "landscape_candidates.csv"

        pairs = {
            "visual_impact": (0, 3),
            "print_quality": (1, 5),
            "clean_scenery": (2, 4)
        }

    image_paths = list(iter_images(args.path))

    if not image_paths:
        print("No images found. Check your path and permissions.")
        return

    print(f"Mode: {args.mode.upper()} | Found {len(image_paths)} images. Starting scan...")

    text_features = build_text_features(prompts)
    results = []
    failures = []

    for batch_paths in tqdm(list(chunked(image_paths, args.batch_size)), desc="Scoring"):
        batch_images = []
        valid_paths = []

        for img_path in batch_paths:
            img = get_image_for_ai(img_path)
            if img is None:
                failures.append(str(img_path))
                continue
            batch_images.append(img)
            valid_paths.append(img_path)

        if not batch_images:
            continue

        try:
            logits = score_images(batch_images, text_features)
        except Exception:
            failures.extend(str(p) for p in valid_paths)
            continue

        for img_path, row in zip(valid_paths, logits):
            row_scores = {}
            for name, (pos_idx, neg_idx) in pairs.items():
                row_scores[name] = soft_pair_score(row, pos_idx, neg_idx)

            # Weighted final score
            if args.mode == 'stock':
                final_score = (
                    row_scores["commercial_appeal"] * 0.28 +
                    row_scores["copyspace_cleanliness"] * 0.20 +
                    row_scores["broad_usability"] * 0.17 +
                    row_scores["technical_quality"] * 0.23 +
                    row_scores["release_friendliness"] * 0.12
                )
            else:
                final_score = sum(row_scores.values()) / len(row_scores)

            results.append({
                "file": img_path.name,
                "path": str(img_path),
                "score": round(final_score, 4),
                **{k: round(v, 4) for k, v in row_scores.items()}
            })

    if not results:
        print("No results scored successfully.")
        return

    df = pd.DataFrame(results).sort_values(by="score", ascending=False)
    df.to_csv(outfile, index=False)

    print(f"\n✅ Done! Top {args.top} {args.mode} candidates:")
    print(df.head(args.top).to_string(index=False))

    if failures:
        failed_file = Path(outfile).with_suffix(".failed.txt")
        failed_file.write_text("\n".join(failures), encoding="utf-8")
        print(f"\n⚠️ {len(failures)} files failed to load. Logged to {failed_file}")


if __name__ == "__main__":
    main()
