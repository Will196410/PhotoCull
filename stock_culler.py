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
print(f"Initializing Stock Judge on {device}...")
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


def clamp01(x):
    return max(0.0, min(1.0, x))


def summarize_scores(scores):
    notes = []

    if scores["commercial_usefulness"] >= 0.72:
        notes.append("strong commercial potential")
    elif scores["commercial_usefulness"] < 0.45:
        notes.append("weak commercial concept")

    if scores["technical_quality"] >= 0.72:
        notes.append("looks technically solid")
    elif scores["technical_quality"] < 0.45:
        notes.append("possible technical weakness")

    if scores["clean_background"] >= 0.70:
        notes.append("clean composition")
    elif scores["clean_background"] < 0.42:
        notes.append("distracting background")

    if scores["copy_space"] >= 0.70:
        notes.append("usable copy space")
    elif scores["copy_space"] < 0.42:
        notes.append("limited copy space")

    if scores["generic_stock_fit"] >= 0.70:
        notes.append("broad stock usability")
    elif scores["generic_stock_fit"] < 0.42:
        notes.append("narrow or personal image")

    if scores["branding_penalty"] >= 0.60:
        notes.append("possible branding/text risk")

    if scores["release_risk_penalty"] >= 0.60:
        notes.append("possible people/property release risk")

    if scores["editorial_bias_penalty"] >= 0.60:
        notes.append("more editorial than commercial")

    return "; ".join(notes) if notes else "mixed stock signals"


def main():
    parser = argparse.ArgumentParser(description="Dedicated AI stock photo culler")
    parser.add_argument("path", help="Path to SSD/folder")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--top", type=int, default=20, help="How many top results to print")
    parser.add_argument("--min-score", type=float, default=0.35, help="Discard images below this final score")
    parser.add_argument("--output", default="stock_candidates.csv", help="Output CSV filename")
    args = parser.parse_args()

    # Positive prompts
    prompts = [
        "a professional commercial stock photo suitable for advertising or marketing",
        "a clean stock image with simple composition, sharp main subject, and broad commercial appeal",
        "a licensable stock photograph with generic everyday usefulness and no obvious legal problems",
        "a polished image with clean background, natural color, and strong technical quality",
        "a versatile stock photo with useful copy space and clear visual concept",
        "a commercially useful image that could illustrate a website, brochure, or article",

        # Negative prompts
        "a blurry low quality snapshot with missed focus or technical flaws",
        "a cluttered messy image with distracting background and weak composition",
        "an image with visible logos, trademarks, packaging, labels, signs, or readable text",
        "a recognizable person or private property image likely to need a release",
        "an editorial news style image rather than a commercial stock photo",
        "a personal family snapshot or travel memory with little stock value",
        "a badly exposed noisy oversharpened or artifact-damaged photograph",
        "a visually confusing image with no clear subject or concept"
    ]

    # Pairs: (positive_idx, negative_idx)
    pairs = {
        "commercial_usefulness": (0, 11),
        "clean_background": (1, 7),
        "generic_stock_fit": (2, 10),
        "technical_quality": (3, 12),
        "copy_space": (4, 13),
        "concept_clarity": (5, 13),

        # Penalties
        "branding_penalty": (8, 2),
        "release_risk_penalty": (9, 2),
        "editorial_bias_penalty": (10, 0),
        "snapshot_penalty": (11, 0),
        "technical_penalty": (12, 3),
        "clutter_penalty": (7, 1),
    }

    image_paths = list(iter_images(args.path))

    if not image_paths:
        print("No images found. Check your path and permissions.")
        return

    print(f"Found {len(image_paths)} images. Starting stock scan...")

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
            raw_scores = {}
            for name, (a_idx, b_idx) in pairs.items():
                raw_scores[name] = soft_pair_score(row, a_idx, b_idx)

            # Positive signals
            commercial_usefulness = raw_scores["commercial_usefulness"]
            clean_background = raw_scores["clean_background"]
            generic_stock_fit = raw_scores["generic_stock_fit"]
            technical_quality = raw_scores["technical_quality"]
            copy_space = raw_scores["copy_space"]
            concept_clarity = raw_scores["concept_clarity"]

            # Penalty signals: higher means more risky / less suitable
            branding_penalty = raw_scores["branding_penalty"]
            release_risk_penalty = raw_scores["release_risk_penalty"]
            editorial_bias_penalty = raw_scores["editorial_bias_penalty"]
            snapshot_penalty = raw_scores["snapshot_penalty"]
            technical_penalty = raw_scores["technical_penalty"]
            clutter_penalty = raw_scores["clutter_penalty"]

            # Weighted final score
            positive_score = (
                commercial_usefulness * 0.23 +
                technical_quality * 0.24 +
                clean_background * 0.16 +
                copy_space * 0.11 +
                generic_stock_fit * 0.16 +
                concept_clarity * 0.10
            )

            penalty_score = (
                branding_penalty * 0.22 +
                release_risk_penalty * 0.15 +
                editorial_bias_penalty * 0.12 +
                snapshot_penalty * 0.13 +
                technical_penalty * 0.23 +
                clutter_penalty * 0.15
            )

            final_score = clamp01((positive_score * 0.78) + ((1.0 - penalty_score) * 0.22))

            scores = {
                "commercial_usefulness": round(commercial_usefulness, 4),
                "technical_quality": round(technical_quality, 4),
                "clean_background": round(clean_background, 4),
                "copy_space": round(copy_space, 4),
                "generic_stock_fit": round(generic_stock_fit, 4),
                "concept_clarity": round(concept_clarity, 4),
                "branding_penalty": round(branding_penalty, 4),
                "release_risk_penalty": round(release_risk_penalty, 4),
                "editorial_bias_penalty": round(editorial_bias_penalty, 4),
                "snapshot_penalty": round(snapshot_penalty, 4),
                "technical_penalty": round(technical_penalty, 4),
                "clutter_penalty": round(clutter_penalty, 4),
            }

            reason = summarize_scores({
                "commercial_usefulness": commercial_usefulness,
                "technical_quality": technical_quality,
                "clean_background": clean_background,
                "copy_space": copy_space,
                "generic_stock_fit": generic_stock_fit,
                "branding_penalty": branding_penalty,
                "release_risk_penalty": release_risk_penalty,
                "editorial_bias_penalty": editorial_bias_penalty,
            })

            if final_score >= args.min_score:
                results.append({
                    "file": img_path.name,
                    "path": str(img_path),
                    "score": round(final_score, 4),
                    "reason": reason,
                    **scores
                })

    if not results:
        print("No results scored successfully above the threshold.")
        return

    df = pd.DataFrame(results).sort_values(
        by=["score", "technical_quality", "commercial_usefulness"],
        ascending=False
    )
    df.to_csv(args.output, index=False)

    print(f"\n✅ Done! Top {args.top} stock candidates:")
    print(df[[
        "file",
        "score",
        "technical_quality",
        "commercial_usefulness",
        "clean_background",
        "copy_space",
        "branding_penalty",
        "release_risk_penalty",
        "reason"
    ]].head(args.top).to_string(index=False))

    if failures:
        failed_file = Path(args.output).with_suffix(".failed.txt")
        failed_file.write_text("\n".join(failures), encoding="utf-8")
        print(f"\n⚠️ {len(failures)} files failed to load. Logged to {failed_file}")


if __name__ == "__main__":
    main()
