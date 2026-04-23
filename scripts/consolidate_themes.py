import argparse
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

# ============================================================================
# CONSTANTS AND DEFAULTS
# ============================================================================

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
DEFAULT_PROMPTS_FILE = Path(__file__).resolve().parent / "theme_prompts.txt"
DEFAULT_RULES_FILE = Path(__file__).resolve().parent / "theme_mapping_rules.json"

FALLBACK_THEME_PROMPTS = [
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
    "a travel photograph showing a place",
    "a dramatic sky photograph",
    "a sunset or sunrise photograph",
    "a misty or foggy landscape photograph",
    "a stormy weather photograph",
    "a moody atmospheric landscape photograph",
    "a photograph where light and weather create the mood",
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

FALLBACK_RULES = {
    "secondary_hints": {
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
    },
    "aliases": {
        "travel snapshot of a place": "Place and Travel",
        "sky, cloud, or weather": "Weather, Light, and Atmosphere",
    },
    "keywords": {
        "waterside": [
            "harbour", "harbor", "port", "boat", "boats", "pier", "quay", "jetty", "marina",
            "river", "waterside", "shore", "shoreline", "fishing", "dock", "docks"
        ],
        "wildlife": [
            "wildlife", "bird", "birds", "bear", "bears", "deer", "fox", "foxes", "seal", "seals",
            "squirrel", "squirrels", "raptor", "raptors", "owl", "owls", "eagle", "eagles",
            "hawk", "hawks", "heron", "gull", "gulls", "otter", "otters"
        ],
        "farm": [
            "farm", "sheep", "cow", "cows", "goat", "goats", "pig", "pigs", "chicken",
            "chickens", "tractor", "barn", "barns", "grazing", "livestock", "calf", "calves"
        ],
        "pet": [
            "pet", "pets", "dog", "dogs", "cat", "cats", "puppy", "puppies", "kitten", "kittens"
        ],
        "people": [
            "people", "person", "portrait", "group", "crowd", "musician", "musicians",
            "performer", "performers", "judge", "judges", "police", "officer", "officers",
            "worker", "workers", "vendor", "vendors", "tourist", "tourists", "child",
            "children", "couple", "man", "woman", "women", "men", "family", "families"
        ],
        "rural": [
            "rural", "field", "fields", "farmland", "farm", "tractor", "barn", "barns",
            "hedgerow", "country", "countryside", "pasture"
        ],
        "atmosphere": [
            "weather", "mist", "misty", "fog", "foggy", "storm", "stormy", "sunset", "sunrise",
            "dramatic", "rain", "rainy", "frost", "cloud", "clouds", "sky", "moody", "gloom",
            "golden", "dusk", "dawn", "atmospheric", "light"
        ],
        "nature_detail": [
            "flower", "flowers", "plant", "plants", "leaf", "leaves", "foliage", "macro",
            "texture", "detail", "bark", "fungi", "mushroom", "mushrooms", "garden"
        ],
        "landscape": [
            "landscape", "coast", "coastal", "beach", "shoreline", "woodland", "forest",
            "countryside", "scenic", "hill", "hills", "valley", "cliff", "moor"
        ],
        "place_travel": [
            "travel", "village", "town", "street", "architecture", "building", "historic",
            "city", "market", "place", "urban", "square", "plaza", "church", "cathedral"
        ],
    },
    "thresholds": {
        "farm_exact_bonus": 6,
        "farm_keyword_bonus_multiplier": 2,
        "farm_single_keyword_bonus": 1,
        "pet_soften_farm_by": 2,
        "pet_soften_wildlife_by": 2,
        "pet_people_bonus": 1,
        "indoor_transport_place_penalty": 4,
        "indoor_transport_uncertain_bonus": 3,
        "abstract_nature_bonus": 2,
        "abstract_place_penalty": 2,
        "farm_cap_without_strong_evidence": 2,
        "atmosphere_theme_bonus": 6,
        "atmosphere_hits_bonus_threshold": 2,
        "atmosphere_hits_bonus": 2,
        "travel_bonus_threshold": 2,
        "travel_bonus_amount": 2,
        "wildlife_farm_conflict_threshold": 4,
        "people_place_conflict_threshold": 4,
        "atmosphere_conflict_threshold": 6,
        "atmosphere_conflict_max_gap": 2,
        "people_override_min": 5,
        "weather_override_min": 6,
        "wildlife_override_min": 5,
        "nature_override_floor": 4,
        "farm_override_min": 6,
        "farm_override_wildlife_gap": 1,
        "indoor_override_min": 5,
        "travel_waterside_override_min": 5,
        "travel_landscape_override_min": 6,
        "place_to_landscape_fallback_min": 6,
        "low_evidence_primary_max": 3,
        "confidence_high_score": 10,
        "confidence_high_gap": 4,
        "confidence_mid_score": 8,
        "confidence_mid_gap": 3,
        "confidence_low_score": 6,
        "confidence_low_gap": 2,
        "confidence_floor_score": 4,
        "uncertain_confidence_cap": 0.45,
    },
}

# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

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

# ============================================================================
# CONFIG FILE LOADING
# ============================================================================

def load_theme_prompts(prompts_file: Optional[Path]) -> List[str]:
    if prompts_file is None:
        return FALLBACK_THEME_PROMPTS

    if not prompts_file.exists():
        print(f"Warning: prompts file not found: {prompts_file}")
        print("Falling back to built-in prompts.")
        return FALLBACK_THEME_PROMPTS

    lines = prompts_file.read_text(encoding="utf-8").splitlines()
    prompts = []
    for line in lines:
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        prompts.append(text)

    if not prompts:
        print(f"Warning: prompts file is empty after filtering comments/blanks: {prompts_file}")
        print("Falling back to built-in prompts.")
        return FALLBACK_THEME_PROMPTS

    print(f"Loaded {len(prompts)} theme prompts from {prompts_file}")
    return prompts


def load_mapping_rules(rules_file: Optional[Path]) -> dict:
    if rules_file is None:
        return FALLBACK_RULES

    if not rules_file.exists():
        print(f"Warning: rules file not found: {rules_file}")
        print("Falling back to built-in mapping rules.")
        return FALLBACK_RULES

    try:
        rules = json.loads(rules_file.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Warning: could not parse rules file: {rules_file}")
        print(f"Reason: {exc}")
        print("Falling back to built-in mapping rules.")
        return FALLBACK_RULES

    merged = json.loads(json.dumps(FALLBACK_RULES))
    for key, value in rules.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value

    print(f"Loaded mapping rules from {rules_file}")
    return merged

# ============================================================================
# PROMPT INFERENCE
# ============================================================================

def infer_category_from_prompt(prompt: str) -> str:
    label = normalize_text(prompt)
    tokens = tokenize(label)

    if "farm animal" in label:
        return "Farm Animals"
    if "pet" in label:
        return "Other / Uncertain"
    if "portrait" in label or "people or group" in label:
        return "People and Human Presence"
    if any(word in tokens for word in {"sunset", "sunrise", "misty", "foggy", "stormy", "dramatic", "moody", "atmospheric", "weather", "sky"}):
        return "Weather, Light, and Atmosphere"
    if any(word in tokens for word in {"bird", "wildlife"}):
        return "Wildlife"
    if any(word in tokens for word in {"flower", "plant", "macro", "texture", "garden", "foliage"}):
        return "Nature Detail"
    if any(word in tokens for word in {"harbour", "harbor", "port", "waterside", "river"}):
        return "Waterside and Harbour"
    if any(word in tokens for word in {"village", "town", "street", "travel", "architecture", "historic"}):
        return "Place and Travel"
    if any(word in tokens for word in {"transport", "vehicle", "indoor"}):
        return "Other / Uncertain"
    if any(word in tokens for word in {"coastal", "countryside", "woodland", "forest", "beach", "shoreline", "landscape"}):
        return "Landscape"
    if "tree" in tokens:
        return "Nature Detail"
    if "abstract visual pattern" in label:
        return "Nature Detail"
    return "Other / Uncertain"


def build_exact_primary_map(theme_prompts: List[str], aliases: dict) -> dict:
    mapping = {}
    for prompt in theme_prompts:
        mapping[normalize_text(prompt)] = infer_category_from_prompt(prompt)
    mapping.update(aliases)
    return mapping


def build_atmosphere_theme_names(theme_prompts: List[str], aliases: dict) -> set:
    names = set()
    for prompt in theme_prompts:
        label = normalize_text(prompt)
        if infer_category_from_prompt(prompt) == "Weather, Light, and Atmosphere":
            names.add(label)
    if "sky, cloud, or weather" in aliases:
        names.add("sky, cloud, or weather")
    return names

# ============================================================================
# FILE DISCOVERY
# ============================================================================

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

# ============================================================================
# EVIDENCE HELPERS
# ============================================================================

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

# ============================================================================
# PRIMARY CATEGORY MAPPING
# ============================================================================

def map_primary_category(
    row: pd.Series,
    exact_primary_map: dict,
    atmosphere_theme_names: set,
    rules: dict,
) -> Tuple[str, float, List[str], dict]:
    # ------------------------------------------------------------------------
    # EXTRACT INPUT
    # ------------------------------------------------------------------------
    review_flags = []
    raw_theme = normalize_text(row.get("theme_name", ""))
    top_labels = [
        normalize_text(row.get("theme_top_label_1", "")),
        normalize_text(row.get("theme_top_label_2", "")),
        normalize_text(row.get("theme_top_label_3", "")),
    ]
    full_text = normalize_text(collect_text_fields(row))
    tokens = tokenize(full_text)

    secondary_hints = rules["secondary_hints"]
    _ = secondary_hints  # reserved for consistency/debugging
    keywords = {k: set(v) for k, v in rules["keywords"].items()}
    t = rules["thresholds"]

    # ------------------------------------------------------------------------
    # COLLECT KEYWORD HITS
    # ------------------------------------------------------------------------
    pet_hits = keyword_score(tokens, keywords["pet"])
    people_hits = keyword_score(tokens, keywords["people"])
    wildlife_hits = keyword_score(tokens, keywords["wildlife"])
    farm_hits = keyword_score(tokens, keywords["farm"])
    waterside_hits = keyword_score(tokens, keywords["waterside"])
    rural_hits = keyword_score(tokens, keywords["rural"])
    atmosphere_hits = keyword_score(tokens, keywords["atmosphere"])
    nature_hits = keyword_score(tokens, keywords["nature_detail"])
    landscape_hits = keyword_score(tokens, keywords["landscape"])
    place_hits = keyword_score(tokens, keywords["place_travel"])

    # ------------------------------------------------------------------------
    # BUILD EVIDENCE
    # ------------------------------------------------------------------------
    evidence = Counter()

    if raw_theme in exact_primary_map:
        evidence[exact_primary_map[raw_theme]] += 8

    for label in top_labels:
        if label in exact_primary_map:
            evidence[exact_primary_map[label]] += 5

    evidence["Waterside and Harbour"] += waterside_hits * 2
    evidence["Wildlife"] += wildlife_hits * 2
    evidence["People and Human Presence"] += people_hits * 2
    evidence["Rural Life and Working Country"] += rural_hits * 2
    evidence["Weather, Light, and Atmosphere"] += atmosphere_hits * 2
    evidence["Nature Detail"] += nature_hits * 2
    evidence["Landscape"] += landscape_hits * 2
    evidence["Place and Travel"] += place_hits * 2
    evidence["Other / Uncertain"] += pet_hits * 2

    if raw_theme == "farm animal":
        evidence["Farm Animals"] += t["farm_exact_bonus"]
    else:
        if farm_hits >= 2:
            evidence["Farm Animals"] += farm_hits * t["farm_keyword_bonus_multiplier"]
        elif farm_hits == 1:
            evidence["Farm Animals"] += t["farm_single_keyword_bonus"]

    if pet_hits > 0:
        evidence["Farm Animals"] = max(evidence["Farm Animals"] - t["pet_soften_farm_by"], 0)
        if wildlife_hits == 0:
            evidence["Wildlife"] = max(evidence["Wildlife"] - t["pet_soften_wildlife_by"], 0)
        if people_hits > 0:
            evidence["People and Human Presence"] += t["pet_people_bonus"]

    if raw_theme in {"indoor", "transport or vehicle"}:
        evidence["Place and Travel"] = max(evidence["Place and Travel"] - t["indoor_transport_place_penalty"], 0)
        evidence["Other / Uncertain"] += t["indoor_transport_uncertain_bonus"]

    if raw_theme == "transport or vehicle" and farm_hits < 2:
        evidence["Farm Animals"] = 0

    if raw_theme == "abstract visual pattern":
        evidence["Nature Detail"] += t["abstract_nature_bonus"]
        evidence["Place and Travel"] = max(evidence["Place and Travel"] - t["abstract_place_penalty"], 0)
        if farm_hits < 2:
            evidence["Farm Animals"] = 0

    if raw_theme == "macro or texture detail" and farm_hits < 2:
        evidence["Farm Animals"] = 0

    farm_theme_exact = (raw_theme == "farm animal")
    strong_farm_evidence = farm_hits >= 2
    if not farm_theme_exact and not strong_farm_evidence:
        evidence["Farm Animals"] = min(evidence["Farm Animals"], t["farm_cap_without_strong_evidence"])

    if raw_theme in atmosphere_theme_names:
        evidence["Weather, Light, and Atmosphere"] += t["atmosphere_theme_bonus"]
    if atmosphere_hits >= t["atmosphere_hits_bonus_threshold"]:
        evidence["Weather, Light, and Atmosphere"] += t["atmosphere_hits_bonus"]

    if raw_theme in {"travel snapshot of a place", "travel photograph showing a place", "travel showing place"}:
        if landscape_hits >= t["travel_bonus_threshold"]:
            evidence["Landscape"] += t["travel_bonus_amount"]
        if waterside_hits >= t["travel_bonus_threshold"]:
            evidence["Waterside and Harbour"] += t["travel_bonus_amount"]
        if people_hits >= t["travel_bonus_threshold"]:
            evidence["People and Human Presence"] += t["travel_bonus_amount"]
        if atmosphere_hits >= t["travel_bonus_threshold"]:
            evidence["Weather, Light, and Atmosphere"] += t["travel_bonus_amount"]
        if nature_hits >= t["travel_bonus_threshold"]:
            evidence["Nature Detail"] += t["travel_bonus_amount"]

    # ------------------------------------------------------------------------
    # REVIEW FLAGS
    # ------------------------------------------------------------------------
    wildlife_farm_conflict_strength = min(evidence["Wildlife"], evidence["Farm Animals"])
    if wildlife_farm_conflict_strength >= t["wildlife_farm_conflict_threshold"]:
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
    if evidence["People and Human Presence"] >= t["people_place_conflict_threshold"] and people_place_other >= t["people_place_conflict_threshold"]:
        review_flags.append("possible_people_place_conflict")

    atmosphere_other = max(
        evidence["Landscape"],
        evidence["Waterside and Harbour"],
        evidence["Rural Life and Working Country"],
    )
    if (
        evidence["Weather, Light, and Atmosphere"] >= t["atmosphere_conflict_threshold"]
        and atmosphere_other >= t["atmosphere_conflict_threshold"]
        and abs(evidence["Weather, Light, and Atmosphere"] - atmosphere_other) <= t["atmosphere_conflict_max_gap"]
    ):
        review_flags.append("possible_atmosphere_primary_conflict")

    # ------------------------------------------------------------------------
    # INITIAL RANKING
    # ------------------------------------------------------------------------
    nonzero_evidence = Counter({k: v for k, v in evidence.items() if v > 0})
    if not nonzero_evidence:
        return "Other / Uncertain", 0.2, ["low_mapping_confidence"], {
            "full_text": full_text,
            "evidence": {},
        }

    ranked = nonzero_evidence.most_common()
    primary, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0

    # ------------------------------------------------------------------------
    # DECISION-STAGE OVERRIDES
    # ------------------------------------------------------------------------
    if raw_theme in atmosphere_theme_names:
        primary = "Weather, Light, and Atmosphere"
        top_score = evidence["Weather, Light, and Atmosphere"]

    if evidence["People and Human Presence"] >= t["people_override_min"] and evidence["People and Human Presence"] >= top_score - 1:
        primary = "People and Human Presence"
        top_score = evidence["People and Human Presence"]

    if raw_theme not in atmosphere_theme_names:
        if evidence["Weather, Light, and Atmosphere"] >= t["weather_override_min"] and evidence["Weather, Light, and Atmosphere"] >= top_score - 1:
            primary = "Weather, Light, and Atmosphere"
            top_score = evidence["Weather, Light, and Atmosphere"]

    if evidence["Wildlife"] >= t["wildlife_override_min"] and evidence["Wildlife"] >= evidence["Farm Animals"]:
        primary = "Wildlife"
        top_score = evidence["Wildlife"]

    if raw_theme != "farm animal":
        if evidence["Nature Detail"] >= max(evidence["Farm Animals"] - 1, t["nature_override_floor"]):
            if evidence["Nature Detail"] >= top_score - 1:
                primary = "Nature Detail"
                top_score = evidence["Nature Detail"]

    if raw_theme == "farm animal" and evidence["Farm Animals"] >= t["farm_override_min"] and evidence["Farm Animals"] > evidence["Wildlife"]:
        primary = "Farm Animals"
        top_score = evidence["Farm Animals"]
    elif evidence["Farm Animals"] >= t["farm_override_min"] and evidence["Farm Animals"] > evidence["Wildlife"] + t["farm_override_wildlife_gap"]:
        primary = "Farm Animals"
        top_score = evidence["Farm Animals"]

    if raw_theme == "indoor":
        if evidence["People and Human Presence"] >= t["indoor_override_min"] and evidence["People and Human Presence"] >= top_score - 1:
            primary = "People and Human Presence"
            top_score = evidence["People and Human Presence"]
        elif evidence["Place and Travel"] >= t["indoor_override_min"] and evidence["Place and Travel"] >= top_score - 1:
            primary = "Place and Travel"
            top_score = evidence["Place and Travel"]
        elif evidence["Waterside and Harbour"] >= t["indoor_override_min"] and evidence["Waterside and Harbour"] >= top_score - 1:
            primary = "Waterside and Harbour"
            top_score = evidence["Waterside and Harbour"]

    if raw_theme in {"travel snapshot of a place", "travel photograph showing a place", "travel showing place"}:
        if evidence["Waterside and Harbour"] >= t["travel_waterside_override_min"] and evidence["Waterside and Harbour"] >= evidence["Place and Travel"] - 1:
            primary = "Waterside and Harbour"
            top_score = evidence["Waterside and Harbour"]
        elif evidence["Landscape"] >= t["travel_landscape_override_min"] and evidence["Landscape"] >= evidence["Place and Travel"] - 1:
            primary = "Landscape"
            top_score = evidence["Landscape"]

    if primary == "Place and Travel" and evidence["Landscape"] >= t["place_to_landscape_fallback_min"]:
        primary = "Landscape"
        top_score = evidence["Landscape"]

    # ------------------------------------------------------------------------
    # WEAK-EVIDENCE FALLBACK
    # ------------------------------------------------------------------------
    if top_score <= t["low_evidence_primary_max"]:
        primary = "Other / Uncertain"
        top_score = evidence["Other / Uncertain"]

    # ------------------------------------------------------------------------
    # CONFIDENCE
    # ------------------------------------------------------------------------
    confidence = 0.5
    if top_score >= t["confidence_high_score"] and (top_score - second_score) >= t["confidence_high_gap"]:
        confidence = 0.96
    elif top_score >= t["confidence_mid_score"] and (top_score - second_score) >= t["confidence_mid_gap"]:
        confidence = 0.88
    elif top_score >= t["confidence_low_score"] and (top_score - second_score) >= t["confidence_low_gap"]:
        confidence = 0.78
    elif top_score >= t["confidence_floor_score"]:
        confidence = 0.64

    if primary == "Other / Uncertain":
        confidence = min(confidence, t["uncertain_confidence_cap"])

    # ------------------------------------------------------------------------
    # LOW-CONFIDENCE REASSIGNMENTS
    # ------------------------------------------------------------------------
    if primary == "Farm Animals" and confidence < 0.7:
        primary = "Other / Uncertain"
        confidence = t["uncertain_confidence_cap"]
        review_flags.append("reassigned_from_farm_animals_low_confidence")

    if primary == "Nature Detail" and confidence < 0.7:
        if raw_theme in {
            "travel snapshot of a place",
            "travel photograph showing a place",
            "travel showing place",
            "transport or vehicle",
            "indoor",
        }:
            primary = "Other / Uncertain"
            confidence = t["uncertain_confidence_cap"]
            review_flags.append("reassigned_from_nature_detail_low_confidence")

    # ------------------------------------------------------------------------
    # FINAL REVIEW FLAGS
    # ------------------------------------------------------------------------
    if raw_theme in {"travel snapshot of a place", "travel photograph showing a place", "travel showing place"} and confidence < 0.7:
        review_flags.append("generic_travel_theme_low_confidence")

    if confidence < 0.7:
        review_flags.append("low_mapping_confidence")

    return primary, confidence, sorted(set(review_flags)), {
        "full_text": full_text,
        "evidence": dict(nonzero_evidence),
    }

# ============================================================================
# SECONDARY CATEGORY DERIVATION
# ============================================================================

def derive_secondary_categories(primary: str, row: pd.Series, rules: dict) -> List[str]:
    secondaries = set(rules["secondary_hints"].get(primary, []))
    tokens = tokenize(collect_text_fields(row))
    keywords = {k: set(v) for k, v in rules["keywords"].items()}

    if primary != "People and Human Presence" and keyword_score(tokens, keywords["people"]) >= 2:
        secondaries.add("People and Human Presence")
    if primary != "Waterside and Harbour" and keyword_score(tokens, keywords["waterside"]) >= 1:
        secondaries.add("Waterside and Harbour")
    if primary != "Landscape" and keyword_score(tokens, keywords["landscape"]) >= 2:
        secondaries.add("Landscape")
    if primary != "Weather, Light, and Atmosphere" and keyword_score(tokens, keywords["atmosphere"]) >= 2:
        secondaries.add("Weather, Light, and Atmosphere")
    if primary != "Rural Life and Working Country" and keyword_score(tokens, keywords["rural"]) >= 2:
        secondaries.add("Rural Life and Working Country")
    if primary != "Nature Detail" and keyword_score(tokens, keywords["nature_detail"]) >= 2:
        secondaries.add("Nature Detail")
    if primary != "Place and Travel" and keyword_score(tokens, keywords["place_travel"]) >= 2:
        secondaries.add("Place and Travel")

    if primary == "Wildlife":
        secondaries.discard("Farm Animals")
    if primary == "Farm Animals":
        secondaries.discard("Wildlife")

    secondaries.discard("Other / Uncertain")
    secondaries.discard(primary)
    return [c for c in MASTER_CATEGORIES if c in secondaries]

# ============================================================================
# OUTPUT BUILDERS
# ============================================================================

def build_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category, group in df.groupby("primary_master_category"):
        years_present = sorted({str(y) for y in group["year"].dropna().tolist()})
        top_source_themes = Counter(group["theme_name"].fillna("").tolist()).most_common(5)
        example_images = group["archive_relative_path"].fillna(group["path"]).head(5).tolist()
        rows.append({
            "master_category": category,
            "image_count": len(group),
            "years_present": ", ".join(years_present),
            "top_source_themes": json.dumps(top_source_themes, ensure_ascii=False),
            "example_images": json.dumps(example_images, ensure_ascii=False),
        })
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

# ============================================================================
# DATA LOADING
# ============================================================================

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

# ============================================================================
# MAIN
# ============================================================================

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
    parser.add_argument(
        "--prompts-file",
        default=str(DEFAULT_PROMPTS_FILE),
        help="Text file containing one theme prompt per line",
    )
    parser.add_argument(
        "--rules-file",
        default=str(DEFAULT_RULES_FILE),
        help="JSON file containing keyword sets, aliases, hints, and thresholds",
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

    prompts_file = Path(args.prompts_file).expanduser().resolve() if args.prompts_file else None
    rules_file = Path(args.rules_file).expanduser().resolve() if args.rules_file else None

    theme_prompts = load_theme_prompts(prompts_file)
    rules = load_mapping_rules(rules_file)
    exact_primary_map = build_exact_primary_map(theme_prompts, rules["aliases"])
    atmosphere_theme_names = build_atmosphere_theme_names(theme_prompts, rules["aliases"])

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
        primary, confidence, flags, evidence = map_primary_category(
            row,
            exact_primary_map,
            atmosphere_theme_names,
            rules,
        )
        secondary = derive_secondary_categories(primary, row, rules)
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
