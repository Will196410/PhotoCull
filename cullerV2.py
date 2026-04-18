import argparse
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import rawpy
import torch
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_NAME = "openai/clip-vit-base-patch32"
RAW_EXTENSIONS = {".dng", ".arw", ".cr2", ".nef", ".orf", ".rw2", ".raf"}
STD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp", ".heic"}
SUPPORTED_EXTENSIONS = RAW_EXTENSIONS | STD_EXTENSIONS


@dataclass(frozen=True)
class ModeConfig:
    positive_prompt: str
    negative_prompt: str
    outfile: str


MODE_CONFIGS = {
    "art": ModeConfig(
        positive_prompt="a masterpiece fine art photograph with moody lighting and strong composition",
        negative_prompt="a blurry low quality accidental snapshot",
        outfile="art_candidates.csv",
    ),
    "stock": ModeConfig(
        positive_prompt="a clean commercial stock photo with copy space, sharp focus, and professional composition",
        negative_prompt="a grainy snapshot with distracting background and poor framing",
        outfile="stock_candidates.csv",
    ),
    "animal": ModeConfig(
        positive_prompt="a striking wildlife photograph of an animal with sharp eyes and strong subject isolation",
        negative_prompt="a blurry or distant animal photo with clutter, cage bars, text, or domestic mess",
        outfile="animal_candidates.csv",
    ),
    "group": ModeConfig(
        positive_prompt="a well-composed group photograph featuring multiple people as the primary subject",
        negative_prompt="a single-person portrait or a photograph with no people",
        outfile="group_candidates.csv",
    ),
    "landscape": ModeConfig(
        positive_prompt="a breathtaking landscape photograph with epic scale, fine light, and strong print quality",
        negative_prompt="a dull landscape with flat light, reflections, clutter, or distracting power lines",
        outfile="landscape_candidates.csv",
    ),
}


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = pick_device()


def load_model_and_processor():
    print(f"Initializing CLIP on {DEVICE}...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor


def is_hidden_or_ignored(path: Path) -> bool:
    parts = path.parts
    if any(part.startswith(".") for part in parts):
        return True
    lowered = str(path).lower()
    ignored_fragments = [
        "/metadata",
        "\\metadata",
        "/thumbnails",
        "\\thumbnails",
        "/previews",
        "\\previews",
        "/sidecar",
        "\\sidecar",
    ]
    return any(fragment in lowered for fragment in ignored_fragments)


def iter_image_paths(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if is_hidden_or_ignored(p):
            continue
        yield p


def safe_open_standard_image(path: Path) -> Optional[Image.Image]:
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except Exception:
        return None


def safe_open_raw_image(path: Path) -> Optional[Image.Image]:
    try:
        with rawpy.imread(str(path)) as raw:
            try:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    with Image.open(io.BytesIO(thumb.data)) as img:
                        return img.convert("RGB")
                return Image.fromarray(thumb.data).convert("RGB")
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


def get_image_for_ai(path: Path) -> Optional[Image.Image]:
    ext = path.suffix.lower()
    if ext in RAW_EXTENSIONS:
        return safe_open_raw_image(path)
    return safe_open_standard_image(path)


def chunked(items: List[Path], size: int) -> Iterable[List[Path]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def build_prompt_embeddings(model, processor, prompts: List[str]) -> torch.Tensor:
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
    with torch.inference_mode():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def score_batch(
    model,
    processor,
    images: List[Image.Image],
    text_features: torch.Tensor,
) -> torch.Tensor:
    image_inputs = processor(images=images, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        image_features = model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ text_features.T
        probs = logits.softmax(dim=1)
    return probs


def main():
    parser = argparse.ArgumentParser(description="AI Photo Culler for fine art prints and stock photography")
    parser.add_argument("path", help="Path to folder or drive root")
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_CONFIGS.keys()),
        required=True,
        help="Selection criteria",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of images to score per batch (default: 16)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many top results to print at the end (default: 10)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Discard results below this positive-prompt probability",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional custom CSV filename",
    )
    args = parser.parse_args()

    root = Path(args.path).expanduser().resolve()
    if not root.exists():
        print(f"Path does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    cfg = MODE_CONFIGS[args.mode]
    outfile = args.output or cfg.outfile
    prompts = [cfg.positive_prompt, cfg.negative_prompt]

    model, processor = load_model_and_processor()
    text_features = build_prompt_embeddings(model, processor, prompts)

    image_paths = list(iter_image_paths(root))
    if not image_paths:
        print("No supported images found. Check your path, extensions, and permissions.")
        return

    print(f"Mode: {args.mode.upper()} | Found {len(image_paths)} images")
    print(f"Writing results to: {outfile}")

    results = []
    failed = []

    for batch_paths in tqdm(list(chunked(image_paths, args.batch_size)), desc="Scoring"):
        batch_images = []
        batch_valid_paths = []

        for path in batch_paths:
            img = get_image_for_ai(path)
            if img is None:
                failed.append(str(path))
                continue
            batch_images.append(img)
            batch_valid_paths.append(path)

        if not batch_images:
            continue

        try:
            probs = score_batch(model, processor, batch_images, text_features)
        except Exception:
            failed.extend(str(p) for p in batch_valid_paths)
            continue

        for path, prob_row in zip(batch_valid_paths, probs):
            positive_score = prob_row[0].item()
            negative_score = prob_row[1].item()

            if positive_score < args.min_score:
                continue

            results.append(
                {
                    "file": path.name,
                    "path": str(path),
                    "parent": str(path.parent),
                    "mode": args.mode,
                    "score": round(positive_score, 6),
                    "negative_score": round(negative_score, 6),
                    "extension": path.suffix.lower(),
                }
            )

    if not results:
        print("No results passed your filter.")
        if failed:
            print(f"Failed to read/score: {len(failed)} files")
        return

    df = pd.DataFrame(results).sort_values(by="score", ascending=False)
    df.to_csv(outfile, index=False)

    print(f"\nDone. Saved {len(df)} scored images to {outfile}")
    print(f"Top {min(args.top, len(df))} {args.mode} candidates:\n")
    print(df[["file", "score", "negative_score", "parent"]].head(args.top).to_string(index=False))

    if failed:
        failfile = Path(outfile).with_suffix(".failed.txt")
        failfile.write_text("\n".join(failed), encoding="utf-8")
        print(f"\nFailed to read/score {len(failed)} files. Logged to {failfile}")


if __name__ == "__main__":
    main()
