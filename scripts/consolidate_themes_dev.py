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
            "river", "waterside", "shore", "shoreline", "coast", "coastal", "beach",
            "fishing", "dock", "docks"
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
    evidence["Weather, Light, and Atmosphere"] += atmosphere_hits
    evidence["Nature Detail"] += nature_hits * 2
    evidence["Landscape"] += landscape_hits * 1.5
    evidence["Place and Travel"] += place_hits * 2
    evidence["Other / Uncertain"] += pet_hits * 2

    if waterside_hits >= 2:
        evidence["Waterside and Harbour"] += 2

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
        evidence["Place and Travel"] = max(
            evidence["Place and Travel"] - t["indoor_transport_place_penalty"], 0
        )
        evidence["Other / Uncertain"] += t["indoor_transport_uncertain_bonus"]

    if raw_theme == "transport or vehicle" and farm_hits < 2:
        evidence["Farm Animals"] = 0

    if raw_theme == "abstract visual pattern":
        evidence["Nature Detail"] += t["abstract_nature_bonus"]
        evidence["Place and Travel"] = max(
            evidence["Place and Travel"] - t["abstract_place_penalty"], 0
        )
        if farm_hits < 2:
            evidence["Farm Animals"] = 0

    if raw_theme == "macro or texture detail" and farm_hits < 2:
        evidence["Farm Animals"] = 0

    farm_theme_exact = raw_theme == "farm animal"
    strong_farm_evidence = farm_hits >= 2
    if not farm_theme_exact and not strong_farm_evidence:
        evidence["Farm Animals"] = min(
            evidence["Farm Animals"], t["farm_cap_without_strong_evidence"]
        )

    if raw_theme in atmosphere_theme_names:
        if landscape_hits >= 2 or waterside_hits >= 2:
            evidence["Weather, Light, and Atmosphere"] += 1
        else:
            evidence["Weather, Light, and Atmosphere"] += 2

    if atmosphere_hits >= 3:
        evidence["Weather, Light, and Atmosphere"] += 1

    if raw_theme in {"travel snapshot of a place", "travel photograph showing a place", "travel showing place"}:
        if landscape_hits >= t["travel_bonus_threshold"]:
            evidence["Landscape"] += t["travel_bonus_amount"]
        if waterside_hits >= t["travel_bonus_threshold"]:
            evidence["Waterside and Harbour"] += t["travel_bonus_amount"]
        if people_hits >= t["travel_bonus_threshold"]:
            evidence["People and Human Presence"] += t["travel_bonus_amount"]
        if atmosphere_hits >= t["travel_bonus_threshold"]:
            evidence["Weather, Light, and Atmosphere"] += 1
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
    if (
        raw_theme != "indoor"
        and evidence["People and Human Presence"] >= t["people_place_conflict_threshold"]
        and people_place_other >= t["people_place_conflict_threshold"]
    ):
        review_flags.append("possible_people_place_conflict")

    atmosphere_other = max(
        evidence["Landscape"],
        evidence["Waterside and Harbour"],
        evidence["Rural Life and Working Country"],
    )
    if (
        raw_theme in atmosphere_theme_names
        and evidence["Weather, Light, and Atmosphere"] >= 6
        and atmosphere_other >= 6
        and abs(evidence["Weather, Light, and Atmosphere"] - atmosphere_other) <= 2
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
    
    # ------------------------------------------------------------------------
    # COASTAL LANDSCAPE CORRECTION
    # ------------------------------------------------------------------------
    # CLIP often treats coastal/beach/shore scenes as generic landscapes.
    # This correction only fires when Landscape is currently winning and the
    # text contains explicit coastal/waterside language, without strong evidence
    # that the subject is really people, architecture/place, rural land, or nature detail.
    coastal_subject_tokens = {
        "coast", "coastal", "beach", "shore", "shoreline",
        "harbour", "harbor", "port", "pier", "quay", "jetty",
        "marina", "dock", "docks", "river", "waterside", "boat", "boats"
    }

    rural_land_tokens = {
        "countryside", "woodland", "forest", "field", "fields",
        "farmland", "pasture", "hill", "hills", "valley", "moor"
    }

    has_coastal_subject = bool(tokens & coastal_subject_tokens)
    has_rural_land_subject = bool(tokens & rural_land_tokens)

    if (
        primary == "Landscape"
        and has_coastal_subject
        and not (has_rural_land_subject and evidence["Landscape"] >= evidence["Waterside and Harbour"] + 4)
        and waterside_hits >= 1
        and evidence["Waterside and Harbour"] >= 4
        and evidence["People and Human Presence"] < 5
        and evidence["Place and Travel"] < 5
        and evidence["Nature Detail"] < 5
        and raw_theme != "indoor"
    ):
        primary = "Waterside and Harbour"
        top_score = evidence["Waterside and Harbour"]
        review_flags.append("coastal_landscape_corrected_to_waterside")

    # ------------------------------------------------------------------------
    # WATERSIDE PROTECTION
    # ------------------------------------------------------------------------
    # Protect genuinely waterside scenes from being swallowed by generic Landscape
    # or Weather, but do not steal People / Place / Nature / Wildlife scenes.
    if (
        primary in {"Landscape", "Weather, Light, and Atmosphere"}
        and waterside_hits >= 2
        and evidence["Waterside and Harbour"] >= 6
        and evidence["People and Human Presence"] < 5
        and evidence["Place and Travel"] < 5
        and evidence["Nature Detail"] < 5
        and evidence["Wildlife"] < 5
        and raw_theme != "indoor"
    ):
        primary = "Waterside and Harbour"
        top_score = evidence["Waterside and Harbour"]
        review_flags.append("waterside_protected_from_landscape_or_weather")

    if raw_theme in atmosphere_theme_names:
    if (
        waterside_hits >= 1
        or place_hits >= 2
        or people_hits >= 2
        or landscape_hits >= 3
        or evidence["Waterside and Harbour"] >= 5
        or evidence["Place and Travel"] >= 5
    ):
        review_flags.append("weather_override_suppressed_by_subject")
    else:
        primary = "Weather, Light, and Atmosphere"
        top_score = evidence["Weather, Light, and Atmosphere"]

    if raw_theme != "indoor":
        if (
            evidence["People and Human Presence"] >= t["people_override_min"]
            and evidence["People and Human Presence"] >= top_score - 1
        ):
            primary = "People and Human Presence"
            top_score = evidence["People and Human Presence"]

    if raw_theme not in atmosphere_theme_names:
        if (
            evidence["Weather, Light, and Atmosphere"] >= t["weather_override_min"]
            and atmosphere_hits >= 2
            and evidence["Weather, Light, and Atmosphere"] >= top_score + 1
        ):
            primary = "Weather, Light, and Atmosphere"
            top_score = evidence["Weather, Light, and Atmosphere"]

    if evidence["Wildlife"] >= t["wildlife_override_min"] and evidence["Wildlife"] >= evidence["Farm Animals"]:
        primary = "Wildlife"
        top_score = evidence["Wildlife"]

    # ------------------------------------------------------------------------
    # STORMY WEATHER PROTECTION
    # ------------------------------------------------------------------------
    if raw_theme == "stormy weather":
        if evidence["Weather, Light, and Atmosphere"] >= evidence["Wildlife"] - 1:
            primary = "Weather, Light, and Atmosphere"
            top_score = evidence["Weather, Light, and Atmosphere"]
            review_flags.append("stormy_weather_protected_from_wildlife")

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

    # ------------------------------------------------------------------------
    # INDOOR CONSERVATIVE OVERRIDES
    # ------------------------------------------------------------------------
    if raw_theme == "indoor":
        has_portrait_label = "portrait of one person" in top_labels
        has_people_group_label = "people or group" in top_labels
        has_architecture_label = "old building or historic architecture" in top_labels
        has_waterside_label = "waterside or river" in top_labels
        has_weather_mood_label = (
            "photograph where light and weather create the mood" in top_labels
            or "light and weather create the mood" in full_text
        )

        strong_portrait_support = (
            has_portrait_label
            and evidence["People and Human Presence"] >= 6
            and people_hits >= 1
        )

        strong_people_group_support = (
            has_people_group_label
            and evidence["People and Human Presence"] >= 8
            and people_hits >= 2
        )

        strong_architecture_support = (
            has_architecture_label
            and evidence["Place and Travel"] >= 7
            and place_hits >= 2
        )

        strong_waterside_support = (
            has_waterside_label
            and evidence["Waterside and Harbour"] >= 8
            and waterside_hits >= 2
        )

        strong_weather_mood_support = (
            has_weather_mood_label
            and evidence["Weather, Light, and Atmosphere"] >= 9
            and atmosphere_hits >= 3
        )

        if strong_portrait_support or strong_people_group_support:
            primary = "People and Human Presence"
            top_score = evidence["People and Human Presence"]
        elif strong_architecture_support:
            primary = "Place and Travel"
            top_score = evidence["Place and Travel"]
        elif strong_waterside_support:
            primary = "Waterside and Harbour"
            top_score = evidence["Waterside and Harbour"]
        elif strong_weather_mood_support:
            primary = "Weather, Light, and Atmosphere"
            top_score = evidence["Weather, Light, and Atmosphere"]
        else:
            # Plain indoor scenes should not inherit weak secondary labels.
            primary = "Other / Uncertain"
            top_score = evidence["Other / Uncertain"]
            review_flags.append("indoor_conservative_fallback")

    if raw_theme in {"travel snapshot of a place", "travel photograph showing a place", "travel showing place"}:
        if (
            evidence["Waterside and Harbour"] >= t["travel_waterside_override_min"]
            and evidence["Waterside and Harbour"] >= evidence["Place and Travel"] - 1
        ):
            primary = "Waterside and Harbour"
            top_score = evidence["Waterside and Harbour"]
        elif (
            waterside_hits < 2
            and evidence["Landscape"] >= t["travel_landscape_override_min"]
            and evidence["Landscape"] >= evidence["Place and Travel"] - 1
        ):
            primary = "Landscape"
            top_score = evidence["Landscape"]

    if (
        primary == "Place and Travel"
        and waterside_hits < 2
        and raw_theme not in {"village, town, or street", "old building or historic architecture"}
        and evidence["Landscape"] >= t["place_to_landscape_fallback_min"]
    ):
        primary = "Landscape"
        top_score = evidence["Landscape"]
    
    # ------------------------------------------------------------------------
    # POST-INDOOR SAFETY NET
    # ------------------------------------------------------------------------
    if raw_theme == "indoor":
        if primary == "People and Human Presence":
            if not (
                ("portrait of one person" in top_labels and people_hits >= 1 and evidence["People and Human Presence"] >= 6)
                or ("people or group" in top_labels and people_hits >= 2 and evidence["People and Human Presence"] >= 8)
            ):
                primary = "Other / Uncertain"
                top_score = evidence["Other / Uncertain"]
                review_flags.append("indoor_people_downgraded_weak_support")

        elif primary == "Waterside and Harbour":
            if not (
                "waterside or river" in top_labels
                and waterside_hits >= 2
                and evidence["Waterside and Harbour"] >= 8
            ):
                primary = "Other / Uncertain"
                top_score = evidence["Other / Uncertain"]
                review_flags.append("indoor_waterside_downgraded_weak_support")

        elif primary == "Weather, Light, and Atmosphere":
            if not (
                atmosphere_hits >= 3
                and evidence["Weather, Light, and Atmosphere"] >= 9
            ):
                primary = "Other / Uncertain"
                top_score = evidence["Other / Uncertain"]
                review_flags.append("indoor_weather_downgraded_weak_support")

        elif primary == "Place and Travel":
            if not (
                "old building or historic architecture" in top_labels
                and place_hits >= 2
                and evidence["Place and Travel"] >= 7
            ):
                primary = "Other / Uncertain"
                top_score = evidence["Other / Uncertain"]
                review_flags.append("indoor_place_downgraded_weak_support")

    # ------------------------------------------------------------------------
    # WATERSIDE PROTECTION (ANTI-WEATHER BLEED)
    # ------------------------------------------------------------------------
    if raw_theme == "waterside or river":
        if evidence["Waterside and Harbour"] >= evidence["Weather, Light, and Atmosphere"] - 2:
            primary = "Waterside and Harbour"
            top_score = evidence["Waterside and Harbour"]

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
    # FINAL INDOOR HARD SAFETY NET
    # ------------------------------------------------------------------------
    if raw_theme == "indoor" or str(row.get("display_theme_name", "")).lower().startswith("indoor"):
        has_portrait_label = "portrait of one person" in top_labels
        has_people_group_label = "people or group" in top_labels
        has_architecture_label = "old building or historic architecture" in top_labels
        has_waterside_label = "waterside or river" in top_labels
        has_weather_mood_label = (
            "photograph where light and weather create the mood" in top_labels
            or "light and weather create the mood" in full_text
        )

        indoor_people_ok = (
            (has_portrait_label and people_hits >= 1 and evidence["People and Human Presence"] >= 6)
            or (has_people_group_label and people_hits >= 2 and evidence["People and Human Presence"] >= 8)
        )

        indoor_place_ok = (
            has_architecture_label
            and place_hits >= 2
            and evidence["Place and Travel"] >= 7
        )

        indoor_waterside_ok = (
            has_waterside_label
            and waterside_hits >= 2
            and evidence["Waterside and Harbour"] >= 8
        )

        indoor_weather_ok = (
            has_weather_mood_label
            and atmosphere_hits >= 3
            and evidence["Weather, Light, and Atmosphere"] >= 9
        )

        if primary == "People and Human Presence" and not indoor_people_ok:
            primary = "Other / Uncertain"
            confidence = t["uncertain_confidence_cap"]
            review_flags.append("final_indoor_people_downgraded")

        elif primary == "Place and Travel" and not indoor_place_ok:
            primary = "Other / Uncertain"
            confidence = t["uncertain_confidence_cap"]
            review_flags.append("final_indoor_place_downgraded")

        elif primary == "Waterside and Harbour" and not indoor_waterside_ok:
            primary = "Other / Uncertain"
            confidence = t["uncertain_confidence_cap"]
            review_flags.append("final_indoor_waterside_downgraded")

        elif primary == "Weather, Light, and Atmosphere" and not indoor_weather_ok:
            primary = "Other / Uncertain"
            confidence = t["uncertain_confidence_cap"]
            review_flags.append("final_indoor_weather_downgraded")

    # ------------------------------------------------------------------------
    # INDOOR LOW-CONFIDENCE CLAMP
    # ------------------------------------------------------------------------
    if raw_theme == "indoor":
        if confidence < 0.7 and primary != "Other / Uncertain":
            primary = "Other / Uncertain"
            confidence = t["uncertain_confidence_cap"]
            review_flags.append("indoor_low_confidence_forced_uncertain")

    # ------------------------------------------------------------------------
    # FINAL REVIEW FLAGS
    # ------------------------------------------------------------------------
    if raw_theme in {"travel snapshot of a place", "travel photograph showing a place", "travel showing place"} and confidence < 0.7:
        review_flags.append("generic_travel_theme_low_confidence")

    if raw_theme == "indoor" and primary == "Other / Uncertain":
        review_flags.append("indoor_low_confidence_or_generic")

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

def build_metrics(df: pd.DataFrame) -> dict:
    category_counts = df["primary_master_category"].value_counts().to_dict()
    low_confidence_count = int((df["mapping_confidence"] < 0.7).sum())
    uncertain_count = int((df["primary_master_category"] == "Other / Uncertain").sum())
        
    return {
        "total_images": int(len(df)),
        "category_counts": category_counts,
        "avg_confidence": round(float(df["mapping_confidence"].mean()), 3),
        "low_confidence_count": low_confidence_count,
        "low_confidence_rate": round(low_confidence_count / max(len(df), 1), 3),
        "uncertain_count": uncertain_count,
        "uncertain_rate": round(uncertain_count / max(len(df), 1), 3),
    }

def build_audit_html(df: pd.DataFrame, output_path: Path, per_category: int = 50) -> None:
    blocks = []

    for category in MASTER_CATEGORIES:
        group = df[df["primary_master_category"] == category].copy()
        if group.empty:
            continue

        group = group.sort_values(
            ["mapping_confidence", "display_theme_name", "relative_path"],
            ascending=[True, True, True],
            na_position="last",
        ).head(per_category)

        cards = []
        for _, row in group.iterrows():
            year = html.escape(str(row.get("year", "")).strip())
            thumb = html.escape(str(row.get("thumb", "")).strip())
            thumb_src = f"../{year}/{thumb}" if year and thumb else ""

            file_name = html.escape(str(row.get("file", "")))
            rel_path = html.escape(str(row.get("archive_relative_path") or row.get("relative_path") or ""))
            theme = html.escape(str(row.get("display_theme_name", "")))
            confidence = html.escape(str(row.get("mapping_confidence", "")))
            flags = html.escape(str(row.get("review_flags", "")))

            img_html = (
                f'<img src="{thumb_src}" alt="{file_name}" loading="lazy">'
                if thumb_src
                else '<div class="no-thumb">No thumbnail</div>'
            )

            cards.append(f"""
            <div class="card">
                <div class="thumb">{img_html}</div>
                <div class="info">
                    <div class="file">{file_name}</div>
                    <div class="path">{rel_path}</div>
                    <div class="theme">{theme}</div>
                    <div class="meta">Confidence: {confidence}</div>
                    <div class="flags">{flags or "—"}</div>
                </div>
            </div>
            """)

        blocks.append(f"""
        <section class="category">
            <h2>{html.escape(category)} <span>{len(group)} sampled</span></h2>
            <div class="grid">
                {''.join(cards)}
            </div>
        </section>
        """)

    page = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Master Gallery Audit</title>
<style>
body {{ margin:0; background:#111; color:#eee; font-family:system-ui,-apple-system,sans-serif; }}
.wrap {{ max-width:1600px; margin:0 auto; padding:24px; }}
h1 {{ margin:0 0 20px; }}
h2 {{ margin:34px 0 12px; border-bottom:1px solid #444; padding-bottom:8px; }}
h2 span {{ color:#aaa; font-size:14px; font-weight:400; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:14px; }}
.card {{ background:#1f1f1f; border:1px solid #333; border-radius:10px; overflow:hidden; }}
.thumb {{ aspect-ratio:1; background:#050505; display:flex; align-items:center; justify-content:center; }}
.thumb img {{ width:100%; height:100%; object-fit:cover; display:block; }}
.no-thumb {{ color:#888; font-size:13px; }}
.info {{ padding:10px; font-size:12px; }}
.file {{ font-weight:700; margin-bottom:4px; }}
.path, .theme, .meta, .flags {{ color:#bbb; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; margin-top:4px; }}
.flags {{ color:#f0c36a; }}
</style>
</head>
<body>
<div class="wrap">
<h1>Master Gallery Audit</h1>
<p>Lowest-confidence samples first, up to {per_category} per category.</p>
{''.join(blocks)}
</div>
</body>
</html>
"""
    output_path.write_text(page, encoding="utf-8")

# ============================================================================
# MAPPING DIAGNOSTICS
# ============================================================================

def add_diagnostic_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["_raw_theme_norm"] = out.get("theme_name", "").apply(normalize_text)
    out["_display_theme_norm"] = out.get("display_theme_name", "").apply(normalize_text)
    out["_top_label_1_norm"] = out.get("theme_top_label_1", "").apply(normalize_text)
    out["_top_label_2_norm"] = out.get("theme_top_label_2", "").apply(normalize_text)
    out["_top_label_3_norm"] = out.get("theme_top_label_3", "").apply(normalize_text)

    return out


def build_mapping_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    diag = add_diagnostic_columns(df)

    group_cols = [
        "_raw_theme_norm",
        "_top_label_1_norm",
        "_top_label_2_norm",
        "_top_label_3_norm",
        "primary_master_category",
    ]

    rows = []
    for keys, group in diag.groupby(group_cols, dropna=False):
        raw_theme, top1, top2, top3, primary = keys
        examples = group["archive_relative_path"].fillna(group["path"]).head(8).tolist()
        rows.append({
            "raw_theme": raw_theme,
            "top_label_1": top1,
            "top_label_2": top2,
            "top_label_3": top3,
            "assigned_master_category": primary,
            "image_count": len(group),
            "avg_confidence": round(float(group["mapping_confidence"].mean()), 3),
            "min_confidence": round(float(group["mapping_confidence"].min()), 3),
            "max_confidence": round(float(group["mapping_confidence"].max()), 3),
            "review_flag_count": int(group["review_flags"].fillna("").astype(str).str.len().gt(0).sum()),
            "example_images": json.dumps(examples, ensure_ascii=False),
        })

    return pd.DataFrame(rows).sort_values(
        ["image_count", "raw_theme", "assigned_master_category"],
        ascending=[False, True, True],
    )


def build_category_theme_matrix(df: pd.DataFrame) -> pd.DataFrame:
    diag = add_diagnostic_columns(df)

    matrix = pd.pivot_table(
        diag,
        index="_raw_theme_norm",
        columns="primary_master_category",
        values="file",
        aggfunc="count",
        fill_value=0,
    ).reset_index()

    matrix = matrix.rename(columns={"_raw_theme_norm": "raw_theme"})

    ordered = ["raw_theme"] + [c for c in MASTER_CATEGORIES if c in matrix.columns]
    extras = [c for c in matrix.columns if c not in ordered]
    matrix = matrix[ordered + extras]

    matrix["total"] = matrix[[c for c in matrix.columns if c != "raw_theme"]].sum(axis=1)
    return matrix.sort_values("total", ascending=False)

def build_audit_sample(df: pd.DataFrame, per_category: int = 50) -> pd.DataFrame:
    parts = []
    for category in MASTER_CATEGORIES:
        group = df[df["primary_master_category"] == category].copy()
        if group.empty:
            continue
        group = group.sort_values(
            ["mapping_confidence", "display_theme_name", "relative_path"],
            ascending=[True, True, True],
            na_position="last",
        ).head(per_category)
        parts.append(group)

    if not parts:
        return pd.DataFrame()

    audit = pd.concat(parts, ignore_index=True)
    cols = [
        "primary_master_category",
        "year",
        "file",
        "path",
        "archive_relative_path",
        "display_theme_name",
        "mapping_confidence",
        "review_flags",
        "mapping_evidence",
    ]
    return audit[[c for c in cols if c in audit.columns]]


def build_coastal_landscape_candidates(df: pd.DataFrame) -> pd.DataFrame:
    # Find Landscape assignments that still contain coastal/waterside language.
    # This is diagnostic only. It helps tune Waterside -> Landscape failures.
    diag = add_diagnostic_columns(df)
    rows = []

    coastal_terms = {
        "coast", "coastal", "beach", "shore", "shoreline",
        "harbour", "harbor", "port", "pier", "quay", "jetty",
        "marina", "dock", "docks", "river", "waterside", "boat", "boats"
    }

    rural_terms = {
        "countryside", "woodland", "forest", "field", "fields",
        "farmland", "pasture", "hill", "hills", "valley", "moor"
    }

    for _, row in diag.iterrows():
        primary = row.get("primary_master_category", "")
        if primary != "Landscape":
            continue

        text = normalize_text(collect_text_fields(row))
        tokens = tokenize(text)

        if not (tokens & coastal_terms):
            continue

        rows.append({
            "year": row.get("year", ""),
            "file": row.get("file", ""),
            "path": row.get("path", ""),
            "archive_relative_path": row.get("archive_relative_path", ""),
            "theme_name": row.get("theme_name", ""),
            "display_theme_name": row.get("display_theme_name", ""),
            "top_label_1": row.get("theme_top_label_1", ""),
            "top_label_2": row.get("theme_top_label_2", ""),
            "top_label_3": row.get("theme_top_label_3", ""),
            "assigned_master_category": primary,
            "mapping_confidence": row.get("mapping_confidence", ""),
            "has_rural_terms": bool(tokens & rural_terms),
            "review_flags": row.get("review_flags", ""),
            "mapping_evidence": row.get("mapping_evidence", ""),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "year", "file", "path", "archive_relative_path",
            "theme_name", "display_theme_name",
            "top_label_1", "top_label_2", "top_label_3",
            "assigned_master_category", "mapping_confidence",
            "has_rural_terms", "review_flags", "mapping_evidence",
        ])

    return pd.DataFrame(rows).sort_values(
        ["has_rural_terms", "year", "theme_name", "file"],
        na_position="last",
    )

def build_suspicious_mappings(df: pd.DataFrame) -> pd.DataFrame:
    diag = add_diagnostic_columns(df)
    rows = []

    for _, row in diag.iterrows():
        raw_theme = row.get("_raw_theme_norm", "")
        display_theme = row.get("_display_theme_norm", "")
        top1 = row.get("_top_label_1_norm", "")
        top2 = row.get("_top_label_2_norm", "")
        top3 = row.get("_top_label_3_norm", "")
        top_labels = {top1, top2, top3}
        primary = row.get("primary_master_category", "")
        confidence = float(row.get("mapping_confidence", 0) or 0)

        reasons = []

        is_indoor = raw_theme == "indoor" or display_theme.startswith("indoor")
        is_generic_travel = raw_theme in {
            "travel snapshot of a place",
            "travel photograph showing a place",
            "travel showing place",
        }

        if is_indoor and primary == "People and Human Presence":
            if "portrait of one person" not in top_labels and "people or group" not in top_labels:
                reasons.append("indoor_assigned_people_without_people_or_portrait_label")

        if is_indoor and primary == "Waterside and Harbour":
            if "waterside or river" not in top_labels:
                reasons.append("indoor_assigned_waterside_without_waterside_label")

        if is_indoor and primary == "Weather, Light, and Atmosphere":
            weather_support_labels = {
                "photograph where light and weather create the mood",
                "light and weather create the mood",
                "moody atmospheric landscape",
                "stormy weather",
                "dramatic sky",
                "sunset or sunrise",
                "misty or foggy landscape",
            }
            if not (top_labels & weather_support_labels):
                reasons.append("indoor_assigned_weather_without_weather_mood_label")

        if is_indoor and primary == "Place and Travel":
            if "old building or historic architecture" not in top_labels:
                reasons.append("indoor_assigned_place_without_architecture_label")

        if raw_theme == "stormy weather" and primary == "Wildlife":
            reasons.append("stormy_weather_assigned_wildlife")

        if raw_theme == "waterside or river" and primary == "Weather, Light, and Atmosphere":
            reasons.append("waterside_assigned_weather")

        if is_generic_travel and confidence < 0.7:
            reasons.append("generic_travel_low_confidence")

        protected_by_rule = any(flag in str(row.get("review_flags", "")) for flag in {
            "waterside_protected_from_weather",
            "stormy_weather_protected_from_wildlife",
        })

        if confidence < 0.7 and primary != "Other / Uncertain" and not protected_by_rule:
            reasons.append("low_confidence_non_uncertain_assignment")

        if reasons:
            rows.append({
                "year": row.get("year", ""),
                "file": row.get("file", ""),
                "path": row.get("path", ""),
                "archive_relative_path": row.get("archive_relative_path", ""),
                "display_theme_name": row.get("display_theme_name", ""),
                "theme_name": row.get("theme_name", ""),
                "top_label_1": row.get("theme_top_label_1", ""),
                "top_label_2": row.get("theme_top_label_2", ""),
                "top_label_3": row.get("theme_top_label_3", ""),
                "assigned_master_category": primary,
                "mapping_confidence": confidence,
                "suspicion_reasons": ", ".join(reasons),
                "review_flags": row.get("review_flags", ""),
                "mapping_evidence": row.get("mapping_evidence", ""),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "year", "file", "path", "archive_relative_path", "display_theme_name",
            "theme_name", "top_label_1", "top_label_2", "top_label_3",
            "assigned_master_category", "mapping_confidence",
            "suspicion_reasons", "review_flags", "mapping_evidence"
        ])

    return pd.DataFrame(rows).sort_values(
        ["suspicion_reasons", "assigned_master_category", "year", "file"],
        na_position="last",
    )


def print_suspicious_summary(suspicious_df: pd.DataFrame) -> None:
    if suspicious_df.empty:
        print("Suspicious mappings:       none found")
        return

    reason_counts = Counter()
    for value in suspicious_df["suspicion_reasons"].fillna(""):
        for reason in [r.strip() for r in str(value).split(",") if r.strip()]:
            reason_counts[reason] += 1

    print("Suspicious mappings:")
    for reason, count in reason_counts.most_common(12):
        print(f"  - {reason}: {count}")

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
    # ------------------------------------------------------------------------
    # DIAGNOSTIC OUTPUTS FOR TUNING
    # ------------------------------------------------------------------------
    diagnostics_df = build_mapping_diagnostics(combined)
    diagnostics_csv = output_root / "master_gallery_mapping_diagnostics.csv"
    diagnostics_df.to_csv(diagnostics_csv, index=False)

    suspicious_df = build_suspicious_mappings(combined)
    suspicious_csv = output_root / "master_gallery_suspicious_mappings.csv"
    suspicious_df.to_csv(suspicious_csv, index=False)

    coastal_candidates_df = build_coastal_landscape_candidates(combined)
    coastal_candidates_csv = output_root / "master_gallery_coastal_landscape_candidates.csv"
    coastal_candidates_df.to_csv(coastal_candidates_csv, index=False)

    matrix_df = build_category_theme_matrix(combined)
    matrix_csv = output_root / "master_gallery_category_theme_matrix.csv"
    matrix_df.to_csv(matrix_csv, index=False)
    if args.include_html:
        html_path = output_root / "master_gallery.html"
        build_html_gallery(combined, html_path)
        print(f"HTML gallery:              {html_path}")

    audit_df = build_audit_sample(combined, per_category=50)
    audit_csv = output_root / "master_gallery_audit_sample.csv"
    audit_df.to_csv(audit_csv, index=False)

    metrics = build_metrics(combined)
    metrics_json = output_root / "master_gallery_metrics.json"
    metrics_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    audit_html = output_root / "master_gallery_audit.html"
    build_audit_html(combined, audit_html, per_category=50)
    
    print()
    print("Done.")
    print(f"Master images CSV:         {images_csv}")
    print(f"Master categories CSV:     {categories_csv}")
    print(f"Review flags CSV:          {flags_csv}")
    print(f"Mapping diagnostics CSV:   {diagnostics_csv}")
    print(f"Suspicious mappings CSV:   {suspicious_csv}")
    print(f"Coastal candidates CSV:    {coastal_candidates_csv}")
    print(f"Category/theme matrix CSV: {matrix_csv}")
    print_suspicious_summary(suspicious_df)
    print(f"Total images consolidated: {len(combined)}")
    print(f"Categories present:        {combined['primary_master_category'].nunique()}")
    print(f"Audit sample CSV:          {audit_csv}")
    print(f"Metrics JSON:              {metrics_json}")
    print(f"Audit HTML:                {audit_html}")
    

if __name__ == "__main__":
    main()
