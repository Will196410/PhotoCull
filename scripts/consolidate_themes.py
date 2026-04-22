import argparse
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple

import pandas as pd

MASTER_CATEGORIES = [
    "Landscape",
    "Waterside and Harbour",
    "Nature Detail",
    "Wildlife",
    "Farm Animals",
    "People and Human Presence",
    "Place and Travel",
    "Rural Life and Working Country",
    "Weather, Light, and Atmosphere",
    "Other / Uncertain",
]

DEFAULT_THEME_OUTPUT_DIRNAME = "theme_output"
DEFAULT_MASTER_GALLERY_DIRNAME = "master_gallery"

EXACT_PRIMARY_MAP = {
    "coastal landscape": "Landscape",
    "harbour or port with boats": "Waterside and Harbour",
    "harbour or port": "Waterside and Harbour",
    "waterside or river": "Waterside and Harbour",
    "beach or shoreline": "Landscape",
    "countryside landscape": "Landscape",
    "woodland or forest": "Landscape",
    "flower or plant close-up": "Nature Detail",
    "macro or texture detail": "Nature Detail",
    "tree or foliage": "Nature Detail",
    "garden": "Nature Detail",
    "bird or wildlife": "Wildlife",
    "farm animal": "Farm Animals",
    "pet": "Other / Uncertain",
    "people or group": "People and Human Presence",
    "portrait of one person": "People and Human Presence",
    "village, town, or street": "Place and Travel",
    "travel snapshot of a place": "Place and Travel",
    "old building or historic architecture": "Place and Travel",
    "transport or vehicle": "Other / Uncertain",
    "indoor": "Other / Uncertain",
    "sky, cloud, or weather": "Weather, Light, and Atmosphere",
    "abstract visual pattern": "Nature Detail",
}

SECONDARY_HINTS = {
    "Landscape": ["Weather, Light, and Atmosphere"],
    "Waterside and Harbour": ["Landscape"],
    "Nature Detail": [],
    "Wildlife": [],
    "Farm Animals": ["Rural Life and Working Country"],
    "People and Human Presence": [],
    "Place and Travel": [],
    "Rural Life and Working Country": ["Landscape"],
    "Weather, Light, and Atmosphere": ["Landscape"],
    "Other / Uncertain": [],
}

WATERSIDE_KEYWORDS = {
    "harbour", "harbor", "port", "boat", "boats", "pier", "quay", "jetty", "marina",
    "river", "waterside", "shore", "shoreline", "fishing", "dock", "docks"
}
WILDLIFE_KEYWORDS = {
    "wildlife", "bird", "birds", "bear", "bears", "deer", "fox", "foxes", "seal", "seals",
    "squirrel", "squirrels", "raptor", "raptors", "owl", "owls", "eagle", "eagles",
    "hawk", "hawks", "heron", "gull", "gulls", "otter", "otters"
}
FARM_KEYWORDS = {
    "farm", "sheep", "cow", "cows", "goat", "goats", "pig", "pigs", "chicken",
    "chickens", "tractor", "barn", "barns", "grazing", "livestock", "calf", "calves"
}
PET_KEYWORDS = {
    "pet", "pets", "dog", "dogs", "cat", "cats", "puppy", "puppies", "kitten", "kittens"
}
PEOPLE_KEYWORDS = {
    "people", "person", "portrait", "group", "crowd", "musician", "musicians",
    "performer", "performers", "judge", "judges", "police", "officer", "officers",
    "worker", "workers", "vendor", "vendors", "tourist", "tourists", "child",
    "children", "couple", "man", "woman", "women", "men", "family", "families"
}
RURAL_KEYWORDS = {
    "rural", "field", "fields", "farmland", "farm", "tractor", "barn", "barns",
    "hedgerow", "country", "countryside", "pasture"
}
ATMOSPHERE_KEYWORDS = {
    "weather", "mist", "misty", "fog", "foggy", "storm", "stormy", "sunset", "sunrise",
    "dramatic", "rain", "rainy", "frost", "cloud", "clouds", "sky", "moody", "gloom",
    "golden", "dusk", "dawn"
}
NATURE_DETAIL_KEYWORDS = {
    "flower", "flowers", "plant", "plants", "leaf", "leaves", "foliage", "macro",
    "texture", "detail", "bark", "fungi", "mushroom", "mushrooms", "garden"
}
LANDSCAPE_KEYWORDS = {
    "landscape", "coast", "coastal", "beach", "shoreline", "woodland", "forest",
    "countryside", "scenic", "hill", "hills", "valley", "cliff", "moor"
}
PLACE_TRAVEL_KEYWORDS = {
    "travel", "village", "town", "street", "architecture", "building", "historic",
    "city", "market", "place", "urban", "square", "plaza", "church", "cathedral"
}


def normalize_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = text.replace("&", " and ")
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9\s,/-]", " ", text)
    text = re.sub(r"\b(a|an)\b", " ", text)
    for phrase in ["photograph", "photo", "scene", "image"]:
        text = text.replace(phrase, " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> set:
    return set(normalize_text(text).split())


def split_csvish(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def guess_year_folders(theme_output_root: Path) -> List[Path]:
    year_dirs = []
    for child in theme_output_root.iterdir():
        if child.is_dir() and child.name.isdigit():
            year_dirs.append(child)
    return sorted(year_dirs, key=lambda p: p.name)


def find_year_files(year_dir: Path, year: str) -> Tuple[Path, Path]:
    images_csv = year_dir / f"{year}_images.csv"
    themes_csv = year_dir / f"{year}_themes.csv"
    return images_csv, themes_csv


def keyword_score(tokens: set, keywords: set) -> int:
    return len(tokens & keywords)


def collect_text_fields(row: pd.Series) -> str:
    parts = [
        row.get("theme_name", ""),
        row.get("display_theme_name", ""),
        row.get("theme_top_label_1", ""),
        row.get("theme_top_label_2", ""),
        row.get("theme_top_label_3", ""),
        row.get("relative_path", ""),
        row.get("folder", ""),
        row.get("dominant_folder", ""),
    ]
    return " | ".join(str(p) for p in parts if pd.notna(p) and str(p).strip())


def has_any_keyword(tokens: set, keywords: set) -> bool:
    return keyword_score(tokens, keywords) > 0


def map_primary_category(row: pd.Series) -> Tuple[str, float, List[str], dict]:
    review_flags = []
    raw_theme = normalize_text(row.get("theme_name", ""))
    top_labels = [
        normalize_text(row.get("theme_top_label_1", "")),
        normalize_text(row.get("theme_top_label_2", "")),
        normalize_text(row.get("theme_top_label_3", "")),
    ]
    full_text = normalize_text(collect_text_fields(row))
    tokens = tokenize(full_text)

    evidence = Counter()

    pet_hits = keyword_score(tokens, PET_KEYWORDS)
    people_hits = keyword_score(tokens, PEOPLE_KEYWORDS)
    wildlife_hits = keyword_score(tokens, WILDLIFE_KEYWORDS)
    farm_hits = keyword_score(tokens, FARM_KEYWORDS)
    waterside_hits = keyword_score(tokens, WATERSIDE_KEYWORDS)
    rural_hits = keyword_score(tokens, RURAL_KEYWORDS)
    atmosphere_hits = keyword_score(tokens, ATMOSPHERE_KEYWORDS)
    nature_hits = keyword_score(tokens, NATURE_DETAIL_KEYWORDS)
    landscape_hits = keyword_score(tokens, LANDSCAPE_KEYWORDS)
    place_hits = keyword_score(tokens, PLACE_TRAVEL_KEYWORDS)

    if raw_theme in EXACT_PRIMARY_MAP:
        evidence[EXACT_PRIMARY_MAP[raw_theme]] += 8

    for label in top_labels:
        if label in EXACT_PRIMARY_MAP:
            evidence[EXACT_PRIMARY_MAP[label]] += 5

    evidence["Waterside and Harbour"] += waterside_hits * 2
    evidence["Wildlife"] += wildlife_hits * 2
    evidence["People and Human Presence"] += people_hits * 2
    evidence["Rural Life and Working Country"] += rural_hits * 2
    evidence["Weather, Light, and Atmosphere"] += atmosphere_hits * 3
    evidence["Nature Detail"] += nature_hits * 2
    evidence["Landscape"] += landscape_hits * 2
    evidence["Place and Travel"] += place_hits * 2
    evidence["Other / Uncertain"] += pet_hits * 2

    # Farm Animals should be stricter.
    if raw_theme == "farm animal":
        evidence["Farm Animals"] += 6
    else:
        if farm_hits >= 2:
            evidence["Farm Animals"] += farm_hits * 2
        elif farm_hits == 1:
            evidence["Farm Animals"] += 1

    # Pets soften animal classification rather than strengthen farm/wild.
    if pet_hits > 0:
        evidence["Farm Animals"] = max(evidence["Farm Animals"] - 2, 0)
        if wildlife_hits == 0:
            evidence["Wildlife"] = max(evidence["Wildlife"] - 2, 0)
        if people_hits > 0:
            evidence["People and Human Presence"] += 1

    # Indoor/transport should not dominate Place and Travel.
    if raw_theme in {"indoor", "transport or vehicle"}:
        evidence["Place and Travel"] = max(evidence["Place and Travel"] - 4, 0)
        evidence["Other / Uncertain"] += 3

    # Transport should not leak into Farm Animals unless farm evidence is genuinely strong.
    if raw_theme == "transport or vehicle" and farm_hits < 2:
        evidence["Farm Animals"] = 0

    # Abstract visual pattern should not drift to travel or farm.
    if raw_theme == "abstract visual pattern":
        evidence["Nature Detail"] += 2
        evidence["Place and Travel"] = max(evidence["Place and Travel"] - 2, 0)
        if farm_hits < 2:
            evidence["Farm Animals"] = 0

    # Macro/detail should beat weak farm contamination.
    if raw_theme == "macro or texture detail" and farm_hits < 2:
        evidence["Farm Animals"] = 0

    # Farm Animals should not win from weak stray evidence.
    farm_theme_exact = (raw_theme == "farm animal")
    strong_farm_evidence = farm_hits >= 2
    if not farm_theme_exact and not strong_farm_evidence:
        evidence["Farm Animals"] = min(evidence["Farm Animals"], 2)

    # Atmosphere deserves a real chance to become primary.
    if raw_theme == "sky, cloud, or weather":
        evidence["Weather, Light, and Atmosphere"] += 4
    if atmosphere_hits >= 2:
        evidence["Weather, Light, and Atmosphere"] += 2

    # Travel snapshot is too broad; trim some of its dominance when stronger evidence exists.
    if raw_theme == "travel snapshot of place":
        if landscape_hits >= 2:
            evidence["Landscape"] += 2
        if waterside_hits >= 2:
            evidence["Waterside and Harbour"] += 2
        if people_hits >= 2:
            evidence["People and Human Presence"] += 2
        if atmosphere_hits >= 2:
            evidence["Weather, Light, and Atmosphere"] += 2
        if nature_hits >= 2:
            evidence["Nature Detail"] += 2

    wildlife_conflict_strength = min(evidence["Wildlife"], evidence["Farm Animals"])
    if wildlife_conflict_strength >= 4:
        review_flags.append("possible_wildlife_farm_conflict")
        if evidence["Wildlife"] >= evidence["Farm Animals"]:
            evidence["Wildlife"] += 2
        else:
            evidence["Farm Animals"] += 1

    people_place_other = max(
        evidence["Place and Travel"],
        evidence["Landscape"],
        evidence["Waterside and Harbour"],
    )
    if evidence["People and Human Presence"] >= 4 and people_place_other >= 4:
        review_flags.append("possible_people_place_conflict")

    atmosphere_other = max(
        evidence["Landscape"],
        evidence["Waterside and Harbour"],
        evidence["Rural Life and Working Country"],
    )
    if evidence["Weather, Light, and Atmosphere"] >= 4 and atmosphere_other >= 4:
        review_flags.append("possible_atmosphere_primary_conflict")

    nonzero_evidence = Counter({k: v for k, v in evidence.items() if v > 0})
    if not nonzero_evidence:
        return "Other / Uncertain", 0.2, ["low_mapping_confidence"], {
            "full_text": full_text,
            "evidence": {},
        }

    ranked = nonzero_evidence.most_common()
    primary, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0

    # Preference rules.
    if evidence["People and Human Presence"] >= 5 and evidence["People and Human Presence"] >= top_score - 1:
        primary = "People and Human Presence"
        top_score = evidence[primary]

    if evidence["Weather, Light, and Atmosphere"] >= 6 and evidence["Weather, Light, and Atmosphere"] >= top_score - 1:
        primary = "Weather, Light, and Atmosphere"
        top_score = evidence[primary]

    if evidence["Wildlife"] >= 5 and evidence["Wildlife"] >= evidence["Farm Animals"]:
        primary = "Wildlife"
        top_score = evidence[primary]

    # Nature Detail should beat ambiguous farm cases unless the theme is explicitly farm animal.
    if raw_theme != "farm animal":
        if evidence["Nature Detail"] >= max(evidence["Farm Animals"] - 1, 4):
            if evidence["Nature Detail"] >= top_score - 1:
                primary = "Nature Detail"
                top_score = evidence[primary]

    if raw_theme == "farm animal" and evidence["Farm Animals"] >= 6 and evidence["Farm Animals"] > evidence["Wildlife"]:
        primary = "Farm Animals"
        top_score = evidence[primary]
    elif evidence["Farm Animals"] >= 6 and evidence["Farm Animals"] > evidence["Wildlife"] + 1:
        primary = "Farm Animals"
        top_score = evidence[primary]

    if primary == "Place and Travel" and evidence["Landscape"] >= 5:
        primary = "Landscape"
        top_score = evidence[primary]

    # If uncertainty remains high, prefer honesty over bad precision.
    if top_score <= 3:
        primary = "Other / Uncertain"
        top_score = evidence[primary]

    confidence = 0.5
    if top_score >= 10 and (top_score - second_score) >= 4:
        confidence = 0.96
    elif top_score >= 8 and (top_score - second_score) >= 3:
        confidence = 0.88
    elif top_score >= 6 and (top_score - second_score) >= 2:
        confidence = 0.78
    elif top_score >= 4:
        confidence = 0.64

    if primary == "Other / Uncertain":
        confidence = min(confidence, 0.45)

    # Low-confidence farm results should not pretend to be reliable.
    if primary == "Farm Animals" and confidence < 0.7:
        primary = "Other / Uncertain"
        confidence = 0.45
        review_flags.append("reassigned_from_farm_animals_low_confidence")

    if confidence < 0.7:
        review_flags.append("low_mapping_confidence")

    return primary, confidence, sorted(set(review_flags)), {
        "full_text": full_text,
        "evidence": dict(nonzero_evidence),
    }


def derive_secondary_categories(primary: str, row: pd.Series, evidence: dict) -> List[str]:
    secondaries = set(SECONDARY_HINTS.get(primary, []))
    tokens = tokenize(collect_text_fields(row))

    if primary != "People and Human Presence" and keyword_score(tokens, PEOPLE_KEYWORDS) >= 2:
        secondaries.add("People and Human Presence")
    if primary != "Waterside and Harbour" and keyword_score(tokens, WATERSIDE_KEYWORDS) >= 1:
        secondaries.add("Waterside and Harbour")
    if primary != "Landscape" and keyword_score(tokens, LANDSCAPE_KEYWORDS) >= 2:
        secondaries.add("Landscape")
    if primary != "Weather, Light, and Atmosphere" and keyword_score(tokens, ATMOSPHERE_KEYWORDS) >= 2:
        secondaries.add("Weather, Light, and Atmosphere")
    if primary != "Rural Life and Working Country" and keyword_score(tokens, RURAL_KEYWORDS) >= 2:
        secondaries.add("Rural Life and Working Country")
    if primary != "Nature Detail" and keyword_score(tokens, NATURE_DETAIL_KEYWORDS) >= 2:
        secondaries.add("Nature Detail")
    if primary != "Place and Travel" and keyword_score(tokens, PLACE_TRAVEL_KEYWORDS) >= 2:
        secondaries.add("Place and Travel")

    if primary == "Wildlife":
        secondaries.discard("Farm Animals")
    if primary == "Farm Animals":
        secondaries.discard("Wildlife")

    secondaries.discard("Other / Uncertain")
    secondaries.discard(primary)
    return [c for c in MASTER_CATEGORIES if c in secondaries]


def build_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category, group in df.groupby("primary_master_category"):
        years_present = sorted({str(y) for y in group["year"].dropna().tolist()})
        top_source_themes = Counter(group["theme_name"].fillna("").tolist()).most_common(5)
        example_images = group["archive_relative_path"].fillna(group["path"]).head(5).tolist()
        rows.append(
            {
                "master_category": category,
                "image_count": len(group),
                "years_present": ", ".join(years_present),
                "top_source_themes": json.dumps(top_source_themes, ensure_ascii=False),
                "example_images": json.dumps(example_images, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows).sort_values(["image_count", "master_category"], ascending=[False, True])


def build_html_gallery(df: pd.DataFrame, output_path: Path, title: str = "Master Gallery") -> None:
    grouped = defaultdict(list)
    for _, row in df.sort_values(["primary_master_category", "year", "relative_path"], na_position="last").iterrows():
        grouped[row["primary_master_category"]].append(row)

    toc_items = []
    blocks = []
    for category in MASTER_CATEGORIES:
        items = grouped.get(category, [])
        if not items:
            continue
        anchor = f"cat-{re.sub(r'[^a-z0-9]+', '-', category.lower()).strip('-')}"
        toc_items.append(f'<li><a href="#{anchor}">{html.escape(category)}</a> <span class="toc-count">({len(items)})</span></li>')
        block = [
            f'<div class="category-block" id="{anchor}">',
            '<div class="category-head">',
            f'<div class="category-title">{html.escape(category)}</div>',
            '<a class="back-top" href="#top">Back to top</a>',
            '</div>',
            f'<div class="category-meta">{len(items)} images</div>',
            '<div class="grid">',
        ]
        for row in items:
            safe_file = html.escape(str(row.get("file", "")))
            safe_rel = html.escape(str(row.get("archive_relative_path") or row.get("relative_path") or ""))
            safe_year = html.escape(str(row.get("year", "")))
            safe_theme = html.escape(str(row.get("display_theme_name", "")))
            secondary = html.escape(str(row.get("secondary_master_categories", "")))
            path_json = json.dumps(str(row.get("path", "")))
            block.append(
                f'''<div class="card">
                    <button class="thumb-btn" onclick='copyPath({path_json})' title="Copy full path to clipboard">
                        <div class="thumb-fallback">{safe_year}</div>
                    </button>
                    <div class="info">
                        <div class="file">{safe_file}</div>
                        <div class="path">{safe_rel}</div>
                        <div class="meta">{safe_theme}</div>
                        <div class="meta">Secondary: {secondary or '—'}</div>
                        <div class="hint">Click tile to copy full path</div>
                    </div>
                </div>'''
            )
        block.append('</div></div>')
        blocks.append("\n".join(block))

    page = f"""
    <html>
    <head>
    <meta charset="utf-8">
    <title>{html.escape(title)}</title>
    <style>
        html {{ scroll-behavior: smooth; }}
        body {{ background:#111; color:#fff; font-family:system-ui,-apple-system,sans-serif; margin:0; }}
        .wrap {{ max-width:1600px; margin:0 auto; padding:20px; }}
        .top-panel {{ background:#181818; border:1px solid #333; border-radius:12px; padding:18px; margin-bottom:28px; }}
        .toc ul {{ margin:0; padding-left:18px; columns:2; column-gap:24px; }}
        .toc li {{ margin-bottom:6px; break-inside:avoid; }}
        .toc a, .back-top {{ color:#9dd; text-decoration:none; }}
        .toc a:hover, .back-top:hover {{ text-decoration:underline; }}
        .toc-count {{ color:#888; font-size:12px; }}
        .category-block {{ margin-bottom:34px; padding-bottom:18px; border-bottom:1px solid #333; }}
        .category-head {{ display:flex; justify-content:space-between; gap:16px; align-items:baseline; }}
        .category-title {{ font-size:24px; }}
        .category-meta, .meta {{ color:#aaa; font-size:12px; margin-top:6px; }}
        .grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(220px, 1fr)); gap:14px; margin-top:14px; }}
        .card {{ background:#222; border:1px solid #333; border-radius:10px; overflow:hidden; }}
        .thumb-btn {{ width:100%; border:0; padding:0; margin:0; background:transparent; cursor:pointer; }}
        .thumb-fallback {{ aspect-ratio:1; display:flex; align-items:center; justify-content:center; background:#000; color:#bbb; font-size:22px; }}
        .info {{ padding:10px; font-size:12px; }}
        .file, .path {{ white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
        .hint {{ color:#888; font-size:11px; margin-top:6px; }}
        .toast {{ position:fixed; bottom:20px; left:50%; transform:translateX(-50%); background:#00ffcc; color:#111; padding:10px 14px; border-radius:999px; font-size:13px; font-weight:600; opacity:0; pointer-events:none; transition:opacity .2s ease; z-index:1000; }}
        .toast.show {{ opacity:1; }}
        @media (max-width:900px) {{ .toc ul {{ columns:1; }} }}
    </style>
    </head>
    <body>
        <div class="wrap" id="top">
            <h1>{html.escape(title)}</h1>
            <div class="top-panel">
                <div><strong>Total images:</strong> {len(df)}</div>
                <div><strong>Categories present:</strong> {df['primary_master_category'].nunique()}</div>
                <div><strong>Years present:</strong> {', '.join(sorted({str(y) for y in df['year'].dropna().tolist()}))}</div>
                <div class="toc">
                    <div style="font-weight:700; margin:12px 0 10px;">Jump to a category</div>
                    <ul>{''.join(toc_items)}</ul>
                </div>
            </div>
            {''.join(blocks)}
            <div id="toast" class="toast"></div>
        </div>
        <script>
        async function copyText(text) {{
            try {{
                if (navigator.clipboard && window.isSecureContext) {{
                    await navigator.clipboard.writeText(text);
                    return true;
                }}
            }} catch (e) {{}}
            try {{
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
            }} catch (e) {{
                return false;
            }}
        }}
        function showToast(message) {{
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.classList.add('show');
            clearTimeout(window.__toastTimer);
            window.__toastTimer = setTimeout(() => toast.classList.remove('show'), 1400);
        }}
        async function copyPath(path) {{
            const ok = await copyText(path);
            showToast(ok ? 'Copied path' : 'Could not copy path');
        }}
        </script>
    </body>
    </html>
    """
    output_path.write_text(page, encoding="utf-8")


def process_year(year_dir: Path, strict: bool = False) -> pd.DataFrame:
    year = year_dir.name
    images_csv, themes_csv = find_year_files(year_dir, year)

    missing = [str(p.name) for p in [images_csv, themes_csv] if not p.exists()]
    if missing:
        message = f"Skipping {year}: missing {', '.join(missing)}"
        if strict:
            raise FileNotFoundError(message)
        print(f"Warning: {message}")
        return pd.DataFrame()

    images_df = pd.read_csv(images_csv)
    _themes_df = pd.read_csv(themes_csv)

    images_df["year"] = year
    return images_df


def main():
    parser = argparse.ArgumentParser(description="Consolidate annual theme outputs into a master gallery view")
    parser.add_argument("project_root", help="Project root containing theme_output/")
    parser.add_argument(
        "--theme-output-root",
        default="",
        help="Optional override for the theme_output root. Defaults to <project_root>/theme_output",
    )
    parser.add_argument(
        "--output-root",
        default="",
        help="Optional override for consolidated outputs. Defaults to <theme_output_root>/master_gallery",
    )
    parser.add_argument(
        "--years",
        default="",
        help="Optional comma-separated list of years to process, e.g. 2008,2010,2015",
    )
    parser.add_argument("--include-html", action="store_true", help="Build master_gallery.html")
    parser.add_argument("--strict", action="store_true", help="Fail if a year folder is missing required files")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    theme_output_root = (
        Path(args.theme_output_root).expanduser().resolve()
        if args.theme_output_root
        else project_root / DEFAULT_THEME_OUTPUT_DIRNAME
    )
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else theme_output_root / DEFAULT_MASTER_GALLERY_DIRNAME
    )

    if not theme_output_root.exists() or not theme_output_root.is_dir():
        raise FileNotFoundError(f"Theme output root not found: {theme_output_root}")

    requested_years = {y.strip() for y in split_csvish(args.years)} if args.years else set()
    year_dirs = guess_year_folders(theme_output_root)
    if requested_years:
        year_dirs = [p for p in year_dirs if p.name in requested_years]

    if not year_dirs:
        raise RuntimeError(f"No year folders found under {theme_output_root}")

    print(f"Reading annual outputs from {theme_output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    frames = []
    for year_dir in year_dirs:
        print(f"Processing {year_dir.name}...")
        df = process_year(year_dir, strict=args.strict)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError("No valid annual data found.")

    combined = pd.concat(frames, ignore_index=True)

    primary_categories = []
    secondary_categories = []
    confidences = []
    review_flags = []
    evidence_json = []

    for _, row in combined.iterrows():
        primary, confidence, flags, evidence = map_primary_category(row)
        secondary = derive_secondary_categories(primary, row, evidence)
        primary_categories.append(primary)
        secondary_categories.append(", ".join(secondary))
        confidences.append(round(float(confidence), 2))
        review_flags.append(", ".join(flags))
        evidence_json.append(json.dumps(evidence["evidence"], ensure_ascii=False, sort_keys=True))

    combined["primary_master_category"] = primary_categories
    combined["secondary_master_categories"] = secondary_categories
    combined["mapping_confidence"] = confidences
    combined["review_flags"] = review_flags
    combined["mapping_evidence"] = evidence_json

    desired_columns = [
        "year",
        "cluster_id",
        "file",
        "path",
        "archive_relative_path",
        "relative_path",
        "theme_name",
        "display_theme_name",
        "theme_top_label_1",
        "theme_top_label_2",
        "theme_top_label_3",
        "primary_master_category",
        "secondary_master_categories",
        "mapping_confidence",
        "review_flags",
        "thumb",
        "mapping_evidence",
    ]
    ordered_columns = [c for c in desired_columns if c in combined.columns] + [c for c in combined.columns if c not in desired_columns]
    combined = combined[ordered_columns].sort_values(
        ["primary_master_category", "year", "display_theme_name", "relative_path"],
        na_position="last",
    )

    images_csv = output_root / "master_gallery_images.csv"
    combined.to_csv(images_csv, index=False)

    categories_df = build_category_summary(combined)
    categories_csv = output_root / "master_gallery_categories.csv"
    categories_df.to_csv(categories_csv, index=False)

    flags_df = combined[combined["review_flags"].fillna("") != ""].copy()
    flags_columns = [
        c for c in [
            "year",
            "file",
            "path",
            "display_theme_name",
            "primary_master_category",
            "review_flags",
            "mapping_confidence",
            "theme_name",
            "archive_relative_path",
        ] if c in flags_df.columns
    ]
    flags_df = flags_df[flags_columns].sort_values(["year", "primary_master_category", "file"], na_position="last")
    flags_csv = output_root / "master_gallery_review_flags.csv"
    flags_df.to_csv(flags_csv, index=False)

    if args.include_html:
        html_path = output_root / "master_gallery.html"
        build_html_gallery(combined, html_path)
        print(f"HTML gallery:              {html_path}")

    print()
    print("Done.")
    print(f"Master images CSV:         {images_csv}")
    print(f"Master categories CSV:     {categories_csv}")
    print(f"Review flags CSV:          {flags_csv}")
    print(f"Total images consolidated: {len(combined)}")
    print(f"Categories present:        {combined['primary_master_category'].nunique()}")


if __name__ == "__main__":
    main()
