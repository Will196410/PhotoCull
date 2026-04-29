#!/usr/bin/env python3
"""
archive_review_alpha_v2.py

Alpha review cockpit for a photo archive.

Reads, by default:
    theme_output/master_gallery/master_gallery_images.csv

Writes:
    theme_output/archive_review_alpha/
        index.html
        review_queue.csv
        review_alpha_metrics.json
        categories/*.html
        years/*.html
        best_review_candidates.html

Run from your project root:
    python archive_review_alpha_v2.py .

Quick test:
    python archive_review_alpha_v2.py . --limit 500

This is deliberately an alpha review tool, not a production classifier.
It creates multiple smaller review pages: dashboard, each bucket/category, each year,
and a best-review-candidates page.
"""

import argparse
import html
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd

DEFAULT_INPUT = "theme_output/master_gallery/master_gallery_images.csv"
DEFAULT_OUTPUT = "theme_output/archive_review_alpha"

STORAGE_KEY = "archive_review_alpha_decisions_v3"

BUCKET_ORDER = [
    "01 Best Review Candidates",
    "02 Waterside / Coastal Candidates",
    "03 Landscape Candidates",
    "04 Weather / Atmosphere Candidates",
    "05 People Candidates",
    "06 Place / Travel Candidates",
    "07 Wildlife Candidates",
    "08 Nature / Detail Candidates",
    "09 Rural / Farm Candidates",
    "10 Low Confidence / Needs Review",
    "11 Other / Uncertain",
]


def safe_str(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", safe_str(value).lower()).strip("_")
    return slug or "untitled"


def normalize_text(value) -> str:
    text = safe_str(value).strip().lower()
    text = text.replace("&", " and ")
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9\s,/-]", " ", text)
    text = re.sub(r"\b(a|an)\b", " ", text)
    for phrase in ["photograph", "photo", "scene", "image"]:
        text = text.replace(phrase, " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(value) -> Set[str]:
    return set(normalize_text(value).split())


def collect_text_fields(row: pd.Series) -> str:
    fields = [
        "primary_master_category",
        "secondary_master_categories",
        "theme_name",
        "display_theme_name",
        "theme_top_label_1",
        "theme_top_label_2",
        "theme_top_label_3",
        "review_flags",
        "archive_relative_path",
        "relative_path",
        "path",
        "folder",
        "dominant_folder",
    ]
    return " | ".join(safe_str(row.get(f, "")) for f in fields if safe_str(row.get(f, "")).strip())


def parse_evidence(value) -> Dict[str, float]:
    try:
        obj = json.loads(safe_str(value))
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


def evidence_score(row: pd.Series, category: str) -> float:
    try:
        return float(parse_evidence(row.get("mapping_evidence", "")).get(category, 0) or 0)
    except Exception:
        return 0.0


def has_any(tokens: Set[str], terms: Set[str]) -> bool:
    return bool(tokens & terms)


def classify_review_bucket(row: pd.Series) -> str:
    primary = safe_str(row.get("primary_master_category", ""))
    try:
        confidence = float(row.get("mapping_confidence", 0) or 0)
    except Exception:
        confidence = 0.0

    flags = safe_str(row.get("review_flags", "")).strip()
    tokens = tokenize(collect_text_fields(row))

    waterside_terms = {
        "harbour", "harbor", "port", "boat", "boats", "pier", "quay", "jetty", "marina",
        "river", "waterside", "shore", "shoreline", "coast", "coastal", "beach", "dock", "docks",
        "fishing", "sea", "seafront", "ocean"
    }
    weather_terms = {
        "weather", "mist", "misty", "fog", "foggy", "storm", "stormy", "sunset", "sunrise",
        "dramatic", "rain", "rainy", "frost", "cloud", "clouds", "sky", "moody", "gloom",
        "golden", "dusk", "dawn", "atmospheric", "light"
    }
    people_terms = {
        "people", "person", "portrait", "group", "crowd", "musician", "musicians",
        "performer", "performers", "judge", "judges", "police", "officer", "officers",
        "worker", "workers", "vendor", "vendors", "tourist", "tourists", "child",
        "children", "couple", "man", "woman", "women", "men", "family", "families"
    }
    place_terms = {
        "travel", "village", "town", "street", "architecture", "building", "historic",
        "city", "market", "place", "urban", "square", "plaza", "church", "cathedral"
    }
    wildlife_terms = {
        "wildlife", "bird", "birds", "bear", "bears", "deer", "fox", "foxes", "seal", "seals",
        "squirrel", "squirrels", "raptor", "raptors", "owl", "owls", "eagle", "eagles",
        "hawk", "hawks", "heron", "gull", "gulls", "otter", "otters"
    }
    nature_terms = {
        "flower", "flowers", "plant", "plants", "leaf", "leaves", "foliage", "macro",
        "texture", "detail", "bark", "fungi", "mushroom", "mushrooms", "garden"
    }
    rural_farm_terms = {
        "rural", "field", "fields", "farmland", "farm", "tractor", "barn", "barns",
        "hedgerow", "country", "countryside", "pasture", "sheep", "cow", "cows",
        "goat", "goats", "pig", "pigs", "livestock"
    }

    if confidence >= 0.78 and not flags and primary not in {"Other / Uncertain", ""}:
        return "01 Best Review Candidates"
    if primary == "Waterside and Harbour" or has_any(tokens, waterside_terms) or evidence_score(row, "Waterside and Harbour") >= 4:
        return "02 Waterside / Coastal Candidates"
    if primary == "Landscape":
        return "03 Landscape Candidates"
    if primary == "Weather, Light, and Atmosphere" or has_any(tokens, weather_terms) or evidence_score(row, "Weather, Light, and Atmosphere") >= 5:
        return "04 Weather / Atmosphere Candidates"
    if primary == "People and Human Presence" or has_any(tokens, people_terms) or evidence_score(row, "People and Human Presence") >= 5:
        return "05 People Candidates"
    if primary == "Place and Travel" or has_any(tokens, place_terms) or evidence_score(row, "Place and Travel") >= 5:
        return "06 Place / Travel Candidates"
    if primary == "Wildlife" or has_any(tokens, wildlife_terms) or evidence_score(row, "Wildlife") >= 5:
        return "07 Wildlife Candidates"
    if primary == "Nature Detail" or has_any(tokens, nature_terms) or evidence_score(row, "Nature Detail") >= 5:
        return "08 Nature / Detail Candidates"
    if primary in {"Farm Animals", "Rural Life and Working Country"} or has_any(tokens, rural_farm_terms):
        return "09 Rural / Farm Candidates"
    if confidence < 0.70 or flags:
        return "10 Low Confidence / Needs Review"
    return "11 Other / Uncertain"


def classify_review_priority(row: pd.Series) -> int:
    try:
        confidence = float(row.get("mapping_confidence", 0) or 0)
    except Exception:
        confidence = 0.0
    flags = safe_str(row.get("review_flags", "")).strip()
    primary = safe_str(row.get("primary_master_category", ""))
    score = 50
    if confidence >= 0.88:
        score += 20
    elif confidence >= 0.78:
        score += 12
    elif confidence < 0.70:
        score -= 8
    if flags:
        score -= 6
    if primary in {"Other / Uncertain", ""}:
        score -= 10
    if primary in {
        "Waterside and Harbour", "Landscape", "Weather, Light, and Atmosphere",
        "Wildlife", "People and Human Presence", "Place and Travel",
    }:
        score += 6
    return max(1, min(99, score))


def make_review_id(row: pd.Series, index: int) -> str:
    source = (
        safe_str(row.get("path", ""))
        or safe_str(row.get("archive_relative_path", ""))
        or safe_str(row.get("relative_path", ""))
        or str(index)
    )
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", source).strip("_")
    return cleaned[-180:] or f"row_{index}"


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "mapping_confidence" not in out.columns:
        out["mapping_confidence"] = 0.5
    if "primary_master_category" not in out.columns:
        out["primary_master_category"] = "Other / Uncertain"
    if "review_flags" not in out.columns:
        out["review_flags"] = ""
    if "archive_relative_path" not in out.columns:
        if "relative_path" in out.columns:
            out["archive_relative_path"] = out["relative_path"]
        elif "path" in out.columns:
            out["archive_relative_path"] = out["path"]
        else:
            out["archive_relative_path"] = ""
    if "display_theme_name" not in out.columns:
        out["display_theme_name"] = out["theme_name"] if "theme_name" in out.columns else ""
    if "year" not in out.columns:
        out["year"] = "Unknown"
    if "file" not in out.columns:
        out["file"] = out["archive_relative_path"].apply(lambda p: Path(safe_str(p)).name)

    out["mapping_confidence"] = pd.to_numeric(out["mapping_confidence"], errors="coerce").fillna(0.0)
    out["year"] = out["year"].fillna("Unknown").astype(str)
    out["review_bucket"] = out.apply(classify_review_bucket, axis=1)
    out["review_priority"] = out.apply(classify_review_priority, axis=1)
    out["review_id"] = [make_review_id(row, i) for i, row in out.iterrows()]
    return out


def build_metrics(df: pd.DataFrame) -> Dict:
    confidence = pd.to_numeric(df["mapping_confidence"], errors="coerce").fillna(0)
    return {
        "total_images": int(len(df)),
        "bucket_counts": {str(k): int(v) for k, v in df["review_bucket"].value_counts().to_dict().items()},
        "year_counts": {str(k): int(v) for k, v in df["year"].fillna("Unknown").value_counts().sort_index().to_dict().items()},
        "primary_category_counts": {str(k): int(v) for k, v in df["primary_master_category"].fillna("").value_counts().to_dict().items()},
        "low_confidence_count": int((confidence < 0.70).sum()),
        "review_flag_count": int(df["review_flags"].fillna("").astype(str).str.strip().ne("").sum()),
    }


def relative_link(from_dir: Path, to_file: Path) -> str:
    return html.escape(str(to_file.relative_to(from_dir.parent) if from_dir == from_dir.parent else Path("../") / to_file.name))


def make_thumb_src(row: pd.Series, page_depth: int) -> str:
    year = safe_str(row.get("year", "")).strip()
    thumb = safe_str(row.get("thumb", "")).strip()
    if not (year and thumb):
        return ""
    prefix = "../" * page_depth
    return f"{prefix}../{html.escape(year)}/{html.escape(thumb)}"


def csv_escape(value: str) -> str:
    s = safe_str(value).replace('"', '""')
    return f'"{s}"'


def write_review_queue(df: pd.DataFrame, output_path: Path) -> None:
    cols = [
        "review_id", "review_bucket", "review_priority", "primary_master_category",
        "mapping_confidence", "review_flags", "year", "file", "path",
        "archive_relative_path", "relative_path", "theme_name", "display_theme_name",
        "theme_top_label_1", "theme_top_label_2", "theme_top_label_3",
        "secondary_master_categories", "mapping_evidence",
    ]
    existing = [c for c in cols if c in df.columns]
    out = df[existing].copy()
    sort_cols = [c for c in ["review_bucket", "year", "review_priority", "mapping_confidence", "archive_relative_path"] if c in out.columns]
    ascending = [True, True, False, False, True][:len(sort_cols)]
    out = out.sort_values(sort_cols, ascending=ascending, na_position="last")
    out.to_csv(output_path, index=False)


def top_nav(page_depth: int) -> str:
    prefix = "../" * page_depth
    return f'''
    <nav class="top-nav">
        <a href="{prefix}index.html">Dashboard</a>
        <a href="{prefix}best_review_candidates.html">Best Review Candidates</a>
        <a href="{prefix}review_queue.csv">Review Queue CSV</a>
    </nav>
    '''


def render_cards(df: pd.DataFrame, page_depth: int, max_cards: int) -> Tuple[str, int, int]:
    cards = []
    total = len(df)
    shown_df = df.head(max_cards) if max_cards and max_cards > 0 else df
    for _, row in shown_df.iterrows():
        review_id_raw = safe_str(row.get("review_id", ""))
        review_id = html.escape(review_id_raw, quote=True)
        file_name = html.escape(safe_str(row.get("file", "")))
        rel_path = html.escape(safe_str(row.get("archive_relative_path") or row.get("relative_path") or ""))
        full_path = safe_str(row.get("path", "")) or safe_str(row.get("archive_relative_path", ""))
        full_path_attr = html.escape(full_path, quote=True)
        primary = html.escape(safe_str(row.get("primary_master_category", "")))
        secondary = html.escape(safe_str(row.get("secondary_master_categories", "")))
        theme = html.escape(safe_str(row.get("display_theme_name", "")))
        confidence = html.escape(f'{float(row.get("mapping_confidence", 0) or 0):.2f}')
        flags = html.escape(safe_str(row.get("review_flags", "")))
        year = html.escape(safe_str(row.get("year", "")))
        priority = html.escape(safe_str(row.get("review_priority", "")))
        top_labels = " · ".join(
            html.escape(safe_str(row.get(c, "")))
            for c in ["theme_top_label_1", "theme_top_label_2", "theme_top_label_3"]
            if safe_str(row.get(c, "")).strip()
        )
        thumb_src = make_thumb_src(row, page_depth=page_depth)
        if thumb_src:
            thumb_html = f'<img src="{thumb_src}" alt="{file_name}" loading="lazy">'
        else:
            thumb_html = f'<div class="thumb-fallback">{year or "No thumb"}</div>'
        cards.append(f'''
        <article class="card" data-review-id="{review_id}" data-path="{full_path_attr}">
            <button class="thumb-btn" data-copy-path="{full_path_attr}" title="Copy archive path">
                {thumb_html}
            </button>
            <div class="card-body">
                <div class="card-title">{file_name or "Untitled"}</div>
                <div class="path" title="{rel_path}">{rel_path}</div>
                <div class="chips">
                    <span class="chip primary">{primary or "No category"}</span>
                    <span class="chip">{year}</span>
                    <span class="chip">conf {confidence}</span>
                    <span class="chip">priority {priority}</span>
                </div>
                <div class="meta"><strong>Theme:</strong> {theme or "—"}</div>
                <div class="meta"><strong>Top labels:</strong> {top_labels or "—"}</div>
                <div class="meta"><strong>Secondary:</strong> {secondary or "—"}</div>
                <div class="flags">{flags or "—"}</div>
                <div class="decision-row">
                    <button data-decision-button="keep" data-review-id="{review_id}">Keep</button>
                    <button data-decision-button="maybe" data-review-id="{review_id}">Maybe</button>
                    <button data-decision-button="reject" data-review-id="{review_id}">Reject</button>
                    <button data-decision-button="fix" data-review-id="{review_id}">Fix cat.</button>
                    <button data-clear-button="1" data-review-id="{review_id}">Clear</button>
                </div>
                <div class="decision-status" data-status-for="{review_id}">Not reviewed</div>
            </div>
        </article>
        ''')
    return "\n".join(cards), total, len(shown_df)


def styles() -> str:
    return r'''
:root { --bg:#101214; --panel:#181b1f; --panel2:#20242a; --text:#f2f2f2; --muted:#b8bec8; --line:#3b414b; --accent:#8ee7ff; --warn:#ffd166; --good:#b7ffbf; }
* { box-sizing:border-box; }
html { scroll-behavior:smooth; }
body { margin:0; background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
.wrap { max-width:1800px; margin:0 auto; padding:20px; }
.top-panel { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:18px; margin-bottom:24px; }
h1 { margin:0 0 8px; font-size:30px; }
h2 { margin:26px 0 8px; }
.alpha { display:inline-block; margin-left:8px; padding:3px 8px; border-radius:999px; background:var(--warn); color:#111; font-size:13px; font-weight:800; vertical-align:middle; }
.top-nav { display:flex; flex-wrap:wrap; gap:10px; margin:0 0 14px; }
.top-nav a, .button-link { border:1px solid var(--line); background:#272c33; color:var(--text); border-radius:10px; padding:9px 11px; text-decoration:none; font-weight:700; display:inline-block; }
.top-nav a:hover, .button-link:hover { border-color:var(--accent); text-decoration:none; }
.summary-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:10px; margin:16px 0; }
.summary-card { background:var(--panel2); border:1px solid var(--line); border-radius:12px; padding:12px; }
.summary-card .num { font-size:24px; font-weight:800; }
.summary-card .label { color:var(--muted); font-size:13px; }
.toolbar { display:flex; flex-wrap:wrap; gap:10px; margin:14px 0; }
.toolbar button, .decision-row button { border:1px solid var(--line); background:#272c33; color:var(--text); border-radius:10px; padding:9px 11px; cursor:pointer; font-weight:700; }
.toolbar button:hover, .decision-row button:hover { border-color:var(--accent); }
.link-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:10px; margin:14px 0; }
.link-card { background:var(--panel2); border:1px solid var(--line); border-radius:12px; padding:12px; }
.link-card a { color:var(--accent); font-weight:800; text-decoration:none; }
.link-card a:hover { text-decoration:underline; }
.small { color:var(--muted); font-size:13px; margin-top:5px; }
.bucket { margin:30px 0 40px; border-top:1px solid var(--line); padding-top:18px; }
.bucket-head { display:flex; justify-content:space-between; align-items:baseline; gap:20px; }
.bucket h2 { margin:0; font-size:24px; }
.bucket p { color:var(--muted); margin:4px 0 12px; }
.more-note { padding:10px 12px; background:#2a2418; border:1px solid #6e571e; border-radius:12px; color:#ffe2a8; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(245px,1fr)); gap:14px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:14px; overflow:hidden; box-shadow:0 8px 18px rgba(0,0,0,0.2); }
.card[data-decision="keep"] { outline:3px solid var(--good); }
.card[data-decision="maybe"] { outline:3px solid var(--warn); }
.card[data-decision="reject"] { opacity:0.56; }
.card[data-decision="fix"] { outline:3px solid #ff9ad5; }
.thumb-btn { width:100%; aspect-ratio:1; display:block; border:0; padding:0; margin:0; background:#050607; cursor:pointer; }
.thumb-btn img { width:100%; height:100%; display:block; object-fit:cover; }
.thumb-fallback { width:100%; height:100%; display:flex; align-items:center; justify-content:center; color:var(--muted); font-size:22px; }
.card-body { padding:11px; font-size:12px; }
.card-title { font-weight:800; font-size:13px; margin-bottom:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.path { color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-bottom:7px; }
.chips { display:flex; flex-wrap:wrap; gap:5px; margin-bottom:8px; }
.chip { border:1px solid var(--line); background:#11161b; color:var(--muted); border-radius:999px; padding:3px 7px; font-size:11px; }
.chip.primary { color:#111; background:var(--accent); border-color:var(--accent); font-weight:800; }
.meta { color:var(--muted); margin-top:5px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.flags { color:var(--warn); min-height:16px; margin-top:6px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.decision-row { display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-top:10px; }
.decision-row button { font-size:12px; padding:7px 8px; }
.decision-status { margin-top:8px; color:var(--muted); font-weight:700; }
.toast { position:fixed; left:50%; bottom:22px; transform:translateX(-50%); background:var(--accent); color:#111; border-radius:999px; padding:11px 16px; font-weight:800; opacity:0; pointer-events:none; transition:opacity .18s ease; z-index:9999; }
.toast.show { opacity:1; }
textarea.export { width:100%; min-height:120px; background:#08090a; color:var(--text); border:1px solid var(--line); border-radius:12px; padding:12px; margin-top:12px; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12px; }
.year-section { margin-top:24px; padding-top:12px; border-top:1px solid var(--line); }
@media (max-width:900px) { .bucket-head { display:block; } }
'''


def script_js() -> str:
    return f'''
<script>
(function () {{
    "use strict";
    const STORAGE_KEY = "{STORAGE_KEY}";

    function loadDecisions() {{
        try {{ return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}"); }}
        catch (e) {{ return {{}}; }}
    }}
    function saveDecisions(data) {{
        try {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(data)); return true; }}
        catch (e) {{ showToast("Could not save in browser"); return false; }}
    }}
    function showToast(message) {{
        const toast = document.getElementById("toast");
        if (!toast) return;
        toast.textContent = message;
        toast.classList.add("show");
        clearTimeout(window.__archiveReviewToastTimer);
        window.__archiveReviewToastTimer = setTimeout(() => toast.classList.remove("show"), 1500);
    }}
    async function copyText(text) {{
        const value = String(text || "");
        try {{
            if (navigator.clipboard && window.isSecureContext) {{
                await navigator.clipboard.writeText(value);
                return true;
            }}
        }} catch (e) {{}}
        try {{
            const ta = document.createElement("textarea");
            ta.value = value;
            ta.setAttribute("readonly", "");
            ta.style.position = "fixed";
            ta.style.left = "-9999px";
            ta.style.top = "0";
            document.body.appendChild(ta);
            ta.focus();
            ta.select();
            ta.setSelectionRange(0, ta.value.length);
            const ok = document.execCommand("copy");
            document.body.removeChild(ta);
            return !!ok;
        }} catch (e) {{ return false; }}
    }}
    function findCard(id) {{
        const cards = document.querySelectorAll(".card");
        for (let i = 0; i < cards.length; i += 1) {{
            if (cards[i].getAttribute("data-review-id") === id) return cards[i];
        }}
        return null;
    }}
    function findStatus(id) {{
        const statuses = document.querySelectorAll(".decision-status");
        for (let i = 0; i < statuses.length; i += 1) {{
            if (statuses[i].getAttribute("data-status-for") === id) return statuses[i];
        }}
        return null;
    }}
    function applyDecisionToCard(id, decision) {{
        const card = findCard(id);
        const status = findStatus(id);
        if (!card) return;
        if (decision) {{
            card.setAttribute("data-decision", decision);
            if (status) status.textContent = "Reviewed: " + decision;
        }} else {{
            card.removeAttribute("data-decision");
            if (status) status.textContent = "Not reviewed";
        }}
    }}
    function setDecision(id, decision) {{
        const data = loadDecisions();
        data[id] = {{ decision: decision, reviewed_at: new Date().toISOString() }};
        saveDecisions(data);
        applyDecisionToCard(id, decision);
        showToast("Marked " + decision);
    }}
    function clearDecision(id) {{
        const data = loadDecisions();
        delete data[id];
        saveDecisions(data);
        applyDecisionToCard(id, "");
        showToast("Cleared");
    }}
    async function copyArchivePath(path) {{
        const ok = await copyText(path);
        const box = document.getElementById("exportBox");
        if (!ok && box) {{ box.value = path; box.focus(); box.select(); }}
        showToast(ok ? "Copied archive path" : "Path shown in export box");
    }}
    function exportDecisions() {{
        const data = loadDecisions();
        const rows = [["review_id","decision","reviewed_at"]];
        Object.keys(data).sort().forEach(id => rows.push([id, data[id].decision || "", data[id].reviewed_at || ""]));
        const csv = rows.map(row => row.map(cell => '"' + String(cell).replaceAll('"', '""') + '"').join(",")).join("\\n");
        const box = document.getElementById("exportBox");
        if (box) {{ box.value = csv; box.focus(); box.select(); }}
        copyText(csv).then(ok => showToast(ok ? "Copied review decisions" : "Export ready"));
    }}
    function copyAllKeptPaths() {{
        const data = loadDecisions();
        const keptIds = new Set(Object.keys(data).filter(id => data[id].decision === "keep"));
        const paths = [];
        document.querySelectorAll(".card").forEach(card => {{
            const id = card.getAttribute("data-review-id");
            if (keptIds.has(id)) {{
                const path = card.getAttribute("data-path") || "";
                if (path) paths.push(path);
            }}
        }});
        const text = paths.join("\\n");
        const box = document.getElementById("exportBox");
        if (box) box.value = text;
        copyText(text).then(ok => showToast(ok ? "Copied " + paths.length + " kept paths" : "Kept paths ready"));
    }}
    function clearAllDecisions() {{
        if (!confirm("Clear all review decisions stored in this browser?")) return;
        try {{ localStorage.removeItem(STORAGE_KEY); }} catch (e) {{}}
        document.querySelectorAll(".card").forEach(card => card.removeAttribute("data-decision"));
        document.querySelectorAll(".decision-status").forEach(el => el.textContent = "Not reviewed");
        showToast("All decisions cleared");
    }}
    function restoreDecisions() {{
        const data = loadDecisions();
        Object.keys(data).forEach(id => applyDecisionToCard(id, data[id].decision));
    }}
    function bindEvents() {{
        document.querySelectorAll("[data-copy-path]").forEach(btn => {{
            btn.addEventListener("click", function () {{ copyArchivePath(btn.getAttribute("data-copy-path") || ""); }});
        }});
        document.querySelectorAll("[data-decision-button]").forEach(btn => {{
            btn.addEventListener("click", function () {{
                setDecision(btn.getAttribute("data-review-id"), btn.getAttribute("data-decision-button"));
            }});
        }});
        document.querySelectorAll("[data-clear-button]").forEach(btn => {{
            btn.addEventListener("click", function () {{ clearDecision(btn.getAttribute("data-review-id")); }});
        }});
        const exportBtn = document.getElementById("exportDecisionsBtn");
        if (exportBtn) exportBtn.addEventListener("click", exportDecisions);
        const copyKeptBtn = document.getElementById("copyKeptBtn");
        if (copyKeptBtn) copyKeptBtn.addEventListener("click", copyAllKeptPaths);
        const clearAllBtn = document.getElementById("clearAllBtn");
        if (clearAllBtn) clearAllBtn.addEventListener("click", clearAllDecisions);
    }}
    bindEvents();
    restoreDecisions();
    showToast("Review page ready");
}}());
</script>
'''


def page_shell(title: str, body: str, page_depth: int) -> str:
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{styles()}</style>
</head>
<body>
<div class="wrap" id="top">
{top_nav(page_depth)}
{body}
</div>
<div id="toast" class="toast"></div>
{script_js()}
</body>
</html>
'''


def toolbar_html() -> str:
    return '''
    <div class="toolbar">
        <button id="exportDecisionsBtn">Export review decisions</button>
        <button id="copyKeptBtn">Copy kept paths on this page</button>
        <button id="clearAllBtn">Clear all decisions</button>
    </div>
    <textarea class="export" id="exportBox" placeholder="Exported review decisions or copied paths will appear here."></textarea>
    '''


def write_gallery_page(df: pd.DataFrame, output_path: Path, title: str, page_depth: int, max_cards: int, group_by_year: bool = True) -> None:
    df = df.sort_values(["year", "review_priority", "mapping_confidence", "archive_relative_path"], ascending=[True, False, False, True], na_position="last")
    intro = f'''
    <div class="top-panel">
        <h1>{html.escape(title)} <span class="alpha">ALPHA</span></h1>
        <p>{len(df)} images. Thumbnail click copies the archive path. Review decisions are stored in this browser until exported.</p>
        {toolbar_html()}
    </div>
    '''
    sections = []
    if group_by_year:
        for year in sorted(df["year"].fillna("Unknown").astype(str).unique()):
            sub = df[df["year"].astype(str) == year].copy()
            sub = sub.sort_values(["review_priority", "mapping_confidence", "archive_relative_path"], ascending=[False, False, True], na_position="last")
            cards, total, shown = render_cards(sub, page_depth, max_cards)
            note = f'<p class="more-note">Showing first {shown} of {total} for this year.</p>' if shown < total else ""
            sections.append(f'''
            <section class="year-section" id="year-{html.escape(slugify(year))}">
                <h2>{html.escape(year)} <span class="small">({total})</span></h2>
                {note}
                <div class="grid">{cards}</div>
            </section>
            ''')
    else:
        sub = df.sort_values(["review_priority", "mapping_confidence", "year", "archive_relative_path"], ascending=[False, False, True, True], na_position="last")
        cards, total, shown = render_cards(sub, page_depth, max_cards)
        note = f'<p class="more-note">Showing first {shown} of {total}.</p>' if shown < total else ""
        sections.append(f'{note}<div class="grid">{cards}</div>')
    output_path.write_text(page_shell(title, intro + "\n".join(sections), page_depth), encoding="utf-8")


def write_dashboard(df: pd.DataFrame, output_root: Path, metrics: Dict, max_cards: int) -> None:
    category_links = []
    categories_dir = output_root / "categories"
    years_dir = output_root / "years"
    for bucket in BUCKET_ORDER:
        count = int((df["review_bucket"] == bucket).sum())
        if count == 0:
            continue
        file = categories_dir / f"{slugify(bucket)}.html"
        category_links.append(f'''
        <div class="link-card">
            <a href="categories/{html.escape(file.name)}">{html.escape(bucket)}</a>
            <div class="small">{count} images, broken down by year</div>
        </div>
        ''')
    year_links = []
    for year, count in sorted(metrics["year_counts"].items()):
        file = years_dir / f"{slugify(year)}.html"
        year_links.append(f'''
        <div class="link-card">
            <a href="years/{html.escape(file.name)}">{html.escape(year)}</a>
            <div class="small">{count} images, grouped by review bucket</div>
        </div>
        ''')
    best_count = int((df["review_bucket"] == "01 Best Review Candidates").sum())
    body = f'''
    <div class="top-panel">
        <h1>Archive Review Alpha <span class="alpha">ALPHA</span></h1>
        <p>Dashboard for browsing the archive by practical review bucket and by year. The classifier does not need to be perfect; these are discovery lanes.</p>
        <div class="summary-grid">
            <div class="summary-card"><div class="num">{metrics["total_images"]}</div><div class="label">images in review queue</div></div>
            <div class="summary-card"><div class="num">{metrics["low_confidence_count"]}</div><div class="label">low confidence</div></div>
            <div class="summary-card"><div class="num">{metrics["review_flag_count"]}</div><div class="label">with review flags</div></div>
            <div class="summary-card"><div class="num">{best_count}</div><div class="label">best review candidates</div></div>
        </div>
        {toolbar_html()}
        <p><a class="button-link" href="best_review_candidates.html">Open Best Review Candidates</a></p>
    </div>
    <section>
        <h2>Review buckets</h2>
        <div class="link-grid">{''.join(category_links)}</div>
    </section>
    <section>
        <h2>Years</h2>
        <div class="link-grid">{''.join(year_links)}</div>
    </section>
    '''
    (output_root / "index.html").write_text(page_shell("Archive Review Alpha", body, page_depth=0), encoding="utf-8")


def write_year_page(df: pd.DataFrame, output_path: Path, year: str, page_depth: int, max_cards: int) -> None:
    parts = []
    ydf = df[df["year"].astype(str) == str(year)].copy()
    intro = f'''
    <div class="top-panel">
        <h1>{html.escape(str(year))} <span class="alpha">ALPHA</span></h1>
        <p>{len(ydf)} images, grouped by review bucket.</p>
        {toolbar_html()}
    </div>
    '''
    for bucket in BUCKET_ORDER:
        sub = ydf[ydf["review_bucket"] == bucket].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["review_priority", "mapping_confidence", "archive_relative_path"], ascending=[False, False, True], na_position="last")
        cards, total, shown = render_cards(sub, page_depth, max_cards)
        note = f'<p class="more-note">Showing first {shown} of {total}.</p>' if shown < total else ""
        parts.append(f'''
        <section class="bucket">
            <div class="bucket-head"><div><h2>{html.escape(bucket)}</h2><p>{total} images</p></div><a href="#top">Back to top</a></div>
            {note}
            <div class="grid">{cards}</div>
        </section>
        ''')
    output_path.write_text(page_shell(f"Archive Review Alpha — {year}", intro + "\n".join(parts), page_depth), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an alpha multi-page HTML review cockpit from master_gallery_images.csv")
    parser.add_argument("project_root", help="Project root containing theme_output/")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input master_gallery_images.csv path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output folder")
    parser.add_argument("--limit", type=int, default=0, help="Optional total image limit for quick testing")
    parser.add_argument("--max-per-section", type=int, default=500, help="Max cards shown per year section or bucket section")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    input_path = Path(args.input).expanduser()
    output_root = Path(args.output).expanduser()
    if not input_path.is_absolute():
        input_path = project_root / input_path
    if not output_root.is_absolute():
        output_root = project_root / output_root
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    categories_dir = output_root / "categories"
    years_dir = output_root / "years"
    categories_dir.mkdir(parents=True, exist_ok=True)
    years_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df = prepare_dataframe(df)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    metrics = build_metrics(df)
    write_review_queue(df, output_root / "review_queue.csv")
    (output_root / "review_alpha_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # Dashboard
    write_dashboard(df, output_root, metrics, args.max_per_section)

    # Best review candidates page
    best_df = df[df["review_bucket"] == "01 Best Review Candidates"].copy()
    write_gallery_page(best_df, output_root / "best_review_candidates.html", "Best Review Candidates", page_depth=0, max_cards=args.max_per_section, group_by_year=True)

    # Category/bucket pages, each broken down by year.
    for bucket in BUCKET_ORDER:
        sub = df[df["review_bucket"] == bucket].copy()
        if sub.empty:
            continue
        write_gallery_page(
            sub,
            categories_dir / f"{slugify(bucket)}.html",
            bucket,
            page_depth=1,
            max_cards=args.max_per_section,
            group_by_year=True,
        )

    # Year pages, each grouped by bucket.
    for year in sorted(df["year"].fillna("Unknown").astype(str).unique()):
        write_year_page(df, years_dir / f"{slugify(year)}.html", year, page_depth=1, max_cards=args.max_per_section)

    print()
    print("Archive Review Alpha v2 built.")
    print(f"Input CSV:       {input_path}")
    print(f"Output folder:   {output_root}")
    print(f"Dashboard:       {output_root / 'index.html'}")
    print(f"Best candidates: {output_root / 'best_review_candidates.html'}")
    print(f"Categories dir:  {categories_dir}")
    print(f"Years dir:       {years_dir}")
    print(f"Review queue:    {output_root / 'review_queue.csv'}")
    print()
    print("Open the dashboard in your browser. Thumbnail click copies the archive path.")


if __name__ == "__main__":
    main()
