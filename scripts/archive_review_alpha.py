#!/usr/bin/env python3
"""
archive_review_alpha.py — alpha photo archive review cockpit.

Run from your project root:
    python archive_review_alpha.py .

Quick test:
    python archive_review_alpha.py . --limit 500

Default input:
    theme_output/master_gallery/master_gallery_images.csv

Default output:
    theme_output/archive_review_alpha/index.html
    theme_output/archive_review_alpha/review_queue.csv
    theme_output/archive_review_alpha/review_alpha_metrics.json
"""

import argparse
import html
import json
import re
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = "theme_output/master_gallery/master_gallery_images.csv"
DEFAULT_OUTPUT = "theme_output/archive_review_alpha"

BUCKETS = [
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

TERMS = {
    "waterside": "harbour harbor port boat boats pier quay jetty marina river waterside shore shoreline coast coastal beach dock docks fishing sea seafront ocean".split(),
    "weather": "weather mist misty fog foggy storm stormy sunset sunrise dramatic rain rainy frost cloud clouds sky moody gloom golden dusk dawn atmospheric light".split(),
    "people": "people person portrait group crowd musician musicians performer performers judge judges police officer officers worker workers vendor vendors tourist tourists child children couple man woman women men family families".split(),
    "place": "travel village town street architecture building historic city market place urban square plaza church cathedral".split(),
    "wildlife": "wildlife bird birds bear bears deer fox foxes seal seals squirrel squirrels raptor raptors owl owls eagle eagles hawk hawks heron gull gulls otter otters".split(),
    "nature": "flower flowers plant plants leaf leaves foliage macro texture detail bark fungi mushroom mushrooms garden".split(),
    "rural": "rural field fields farmland farm tractor barn barns hedgerow country countryside pasture sheep cow cows goat goats pig pigs livestock".split(),
}
TERMS = {k: set(v) for k, v in TERMS.items()}


def s(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def norm(value):
    text = s(value).lower().replace("&", " and ").replace("_", " ")
    text = re.sub(r"[^a-z0-9\s,/-]", " ", text)
    text = re.sub(r"\b(a|an)\b", " ", text)
    for word in ["photograph", "photo", "scene", "image"]:
        text = text.replace(word, " ")
    return re.sub(r"\s+", " ", text).strip()


def tokens_for(row):
    fields = [
        "primary_master_category", "secondary_master_categories", "theme_name",
        "display_theme_name", "theme_top_label_1", "theme_top_label_2", "theme_top_label_3",
        "review_flags", "archive_relative_path", "relative_path", "path", "folder", "dominant_folder",
    ]
    text = " | ".join(s(row.get(f, "")) for f in fields)
    return set(norm(text).split())


def evidence(row, category):
    try:
        obj = json.loads(s(row.get("mapping_evidence", "")))
        if isinstance(obj, dict):
            return float(obj.get(category, 0) or 0)
    except Exception:
        pass
    return 0.0


def confidence(row):
    try:
        return float(row.get("mapping_confidence", 0) or 0)
    except Exception:
        return 0.0


def bucket_for(row):
    primary = s(row.get("primary_master_category", ""))
    conf = confidence(row)
    flags = s(row.get("review_flags", "")).strip()
    tok = tokens_for(row)

    if conf >= 0.78 and not flags and primary not in {"Other / Uncertain", ""}:
        return BUCKETS[0]
    if primary == "Waterside and Harbour" or tok & TERMS["waterside"] or evidence(row, "Waterside and Harbour") >= 4:
        return BUCKETS[1]
    if primary == "Landscape":
        return BUCKETS[2]
    if primary == "Weather, Light, and Atmosphere" or tok & TERMS["weather"] or evidence(row, "Weather, Light, and Atmosphere") >= 5:
        return BUCKETS[3]
    if primary == "People and Human Presence" or tok & TERMS["people"] or evidence(row, "People and Human Presence") >= 5:
        return BUCKETS[4]
    if primary == "Place and Travel" or tok & TERMS["place"] or evidence(row, "Place and Travel") >= 5:
        return BUCKETS[5]
    if primary == "Wildlife" or tok & TERMS["wildlife"] or evidence(row, "Wildlife") >= 5:
        return BUCKETS[6]
    if primary == "Nature Detail" or tok & TERMS["nature"] or evidence(row, "Nature Detail") >= 5:
        return BUCKETS[7]
    if primary in {"Farm Animals", "Rural Life and Working Country"} or tok & TERMS["rural"]:
        return BUCKETS[8]
    if conf < 0.70 or flags:
        return BUCKETS[9]
    return BUCKETS[10]


def priority_for(row):
    conf = confidence(row)
    primary = s(row.get("primary_master_category", ""))
    flags = s(row.get("review_flags", "")).strip()
    score = 50
    if conf >= 0.88:
        score += 20
    elif conf >= 0.78:
        score += 12
    elif conf < 0.70:
        score -= 8
    if flags:
        score -= 6
    if primary in {"Other / Uncertain", ""}:
        score -= 10
    if primary in {"Waterside and Harbour", "Landscape", "Weather, Light, and Atmosphere", "Wildlife", "People and Human Presence", "Place and Travel"}:
        score += 6
    return max(1, min(99, score))


def review_id(row, idx):
    source = s(row.get("path", "")) or s(row.get("archive_relative_path", "")) or s(row.get("relative_path", "")) or str(idx)
    rid = re.sub(r"[^a-zA-Z0-9_-]+", "_", source).strip("_")[-180:]
    return rid or "row_" + str(idx)


def prepare(df):
    out = df.copy()
    if "mapping_confidence" not in out.columns:
        out["mapping_confidence"] = 0.5
    if "primary_master_category" not in out.columns:
        out["primary_master_category"] = "Other / Uncertain"
    if "review_flags" not in out.columns:
        out["review_flags"] = ""
    if "archive_relative_path" not in out.columns:
        out["archive_relative_path"] = out["relative_path"] if "relative_path" in out.columns else out["path"] if "path" in out.columns else ""
    if "display_theme_name" not in out.columns:
        out["display_theme_name"] = out["theme_name"] if "theme_name" in out.columns else ""
    out["review_bucket"] = out.apply(bucket_for, axis=1)
    out["review_priority"] = out.apply(priority_for, axis=1)
    out["review_id"] = [review_id(row, i) for i, row in out.iterrows()]
    return out


def thumb_src(row):
    year = s(row.get("year", "")).strip()
    thumb = s(row.get("thumb", "")).strip()
    return "../" + html.escape(year) + "/" + html.escape(thumb) if year and thumb else ""


def write_queue(df, path):
    cols = [
        "review_bucket", "review_priority", "primary_master_category", "mapping_confidence",
        "review_flags", "year", "file", "path", "archive_relative_path", "relative_path",
        "theme_name", "display_theme_name", "theme_top_label_1", "theme_top_label_2",
        "theme_top_label_3", "secondary_master_categories", "mapping_evidence",
    ]
    cols = [c for c in cols if c in df.columns]
    df[cols].sort_values(["review_bucket", "review_priority"], ascending=[True, False]).to_csv(path, index=False)


def build_html(df, path, title, max_per_bucket):
    low_conf = int((pd.to_numeric(df["mapping_confidence"], errors="coerce").fillna(0) < 0.70).sum())
    flagged = int(df["review_flags"].fillna("").astype(str).str.strip().ne("").sum())
    toc = []
    sections = []

    for bucket in BUCKETS:
        group = df[df["review_bucket"] == bucket].copy()
        if group.empty:
            continue
        group = group.sort_values(["review_priority", "mapping_confidence"], ascending=[False, False]).head(max_per_bucket)
        total = int((df["review_bucket"] == bucket).sum())
        anchor = "bucket-" + re.sub(r"[^a-z0-9]+", "-", bucket.lower()).strip("-")
        toc.append(f'<li><a href="#{anchor}">{html.escape(bucket)}</a> <span>({total})</span></li>')
        cards = []
        for _, row in group.iterrows():
            rid = html.escape(s(row.get("review_id", "")))
            fname = html.escape(s(row.get("file", "")) or "Untitled")
            full_path = s(row.get("path", "")) or s(row.get("archive_relative_path", ""))
            rel = html.escape(s(row.get("archive_relative_path", "")) or s(row.get("relative_path", "")))
            primary = html.escape(s(row.get("primary_master_category", "")) or "No category")
            conf = html.escape(s(row.get("mapping_confidence", "")))
            pri = html.escape(s(row.get("review_priority", "")))
            theme = html.escape(s(row.get("display_theme_name", "")))
            flags = html.escape(s(row.get("review_flags", "")))
            labels = " · ".join(html.escape(s(row.get(c, ""))) for c in ["theme_top_label_1", "theme_top_label_2", "theme_top_label_3"] if s(row.get(c, "")).strip())
            img = f'<img src="{thumb_src(row)}" loading="lazy" alt="{fname}">' if thumb_src(row) else '<div class="no-thumb">No thumb</div>'
            cards.append(f'''
<article class="card" data-review-id="{rid}" data-path="{html.escape(full_path)}">
  <button class="thumb" onclick='copyPath({json.dumps(full_path)})'>{img}</button>
  <div class="body">
    <div class="title">{fname}</div>
    <div class="path">{rel}</div>
    <div class="chips"><span>{primary}</span><span>conf {conf}</span><span>priority {pri}</span></div>
    <div class="meta"><b>Theme:</b> {theme or "—"}</div>
    <div class="meta"><b>Labels:</b> {labels or "—"}</div>
    <div class="flags">{flags or "—"}</div>
    <div class="buttons"><button onclick="setDecision('{rid}','keep')">Keep</button><button onclick="setDecision('{rid}','maybe')">Maybe</button><button onclick="setDecision('{rid}','reject')">Reject</button><button onclick="setDecision('{rid}','fix')">Fix cat.</button><button onclick="clearDecision('{rid}')">Clear</button></div>
    <div class="status" id="status-{rid}">Not reviewed</div>
  </div>
</article>''')
        more = f'<p class="note">Showing {len(group)} of {total}. See review_queue.csv for the full bucket.</p>' if total > len(group) else ""
        sections.append(f'<section id="{anchor}"><h2>{html.escape(bucket)} <small>{total}</small></h2><a href="#top">Back to top</a>{more}<div class="grid">{"".join(cards)}</div></section>')

    page = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title><style>
body{margin:0;background:#101214;color:#f2f2f2;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}.wrap{max-width:1800px;margin:auto;padding:20px}.top{background:#181b1f;border:1px solid #3b414b;border-radius:16px;padding:18px}.alpha{background:#ffd166;color:#111;padding:3px 8px;border-radius:999px;font-size:13px}.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:14px 0}.stat{background:#20242a;border:1px solid #3b414b;border-radius:12px;padding:12px}.num{font-size:24px;font-weight:800}.label{color:#b8bec8}.toolbar button,.buttons button{border:1px solid #3b414b;background:#272c33;color:#f2f2f2;border-radius:9px;padding:8px;margin:3px;cursor:pointer;font-weight:700}a{color:#8ee7ff}.toc ul{columns:2}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(245px,1fr));gap:14px;margin-top:12px}.card{background:#181b1f;border:1px solid #3b414b;border-radius:14px;overflow:hidden}.card[data-decision=keep]{outline:3px solid #b7ffbf}.card[data-decision=maybe]{outline:3px solid #ffd166}.card[data-decision=reject]{opacity:.55}.card[data-decision=fix]{outline:3px solid #ff9ad5}.thumb{width:100%;aspect-ratio:1;border:0;padding:0;background:#050607;cursor:pointer}.thumb img{width:100%;height:100%;object-fit:cover}.no-thumb{height:100%;display:flex;align-items:center;justify-content:center;color:#b8bec8}.body{padding:11px;font-size:12px}.title{font-weight:800;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.path,.meta,.flags{color:#b8bec8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:5px}.flags{color:#ffd166}.chips{display:flex;gap:5px;flex-wrap:wrap;margin:8px 0}.chips span{border:1px solid #3b414b;border-radius:999px;padding:3px 7px}.status{color:#b8bec8;font-weight:700;margin-top:6px}textarea{width:100%;min-height:110px;margin-top:12px;background:#08090a;color:#f2f2f2;border:1px solid #3b414b;border-radius:12px;padding:12px}.toast{position:fixed;left:50%;bottom:22px;transform:translateX(-50%);background:#8ee7ff;color:#111;border-radius:999px;padding:11px 16px;font-weight:800;opacity:0}.toast.show{opacity:1}@media(max-width:900px){.toc ul{columns:1}}
</style></head><body><div class="wrap" id="top"><div class="top"><h1>__TITLE__ <span class="alpha">ALPHA</span></h1><p>Review cockpit, not a final classifier. Buckets are discovery lanes.</p><div class="stats"><div class="stat"><div class="num">__TOTAL__</div><div class="label">images</div></div><div class="stat"><div class="num">__LOW__</div><div class="label">low confidence</div></div><div class="stat"><div class="num">__FLAGS__</div><div class="label">with flags</div></div></div><div class="toolbar"><button onclick="exportDecisions()">Export review decisions</button><button onclick="copyAllKeptPaths()">Copy kept paths</button><button onclick="clearAllDecisions()">Clear all decisions</button></div><div class="toc"><b>Jump to bucket</b><ul>__TOC__</ul></div><textarea id="exportBox" placeholder="Exported review decisions will appear here."></textarea></div>__SECTIONS__</div><div id="toast" class="toast"></div><script>
const KEY='archive_review_alpha_decisions_v1';function load(){try{return JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){return {}}}function save(d){localStorage.setItem(KEY,JSON.stringify(d))}function toast(m){const t=document.getElementById('toast');t.textContent=m;t.classList.add('show');clearTimeout(window.tt);window.tt=setTimeout(()=>t.classList.remove('show'),1400)}async function copyText(x){try{if(navigator.clipboard&&window.isSecureContext){await navigator.clipboard.writeText(x);return true}}catch(e){}const a=document.createElement('textarea');a.value=x;document.body.appendChild(a);a.select();const ok=document.execCommand('copy');document.body.removeChild(a);return ok}function copyPath(p){copyText(p).then(ok=>toast(ok?'Copied path':'Could not copy'))}function apply(id,d){const c=document.querySelector('[data-review-id="'+CSS.escape(id)+'"]');const s=document.getElementById('status-'+id);if(!c)return;if(d){c.dataset.decision=d;if(s)s.textContent='Reviewed: '+d}else{delete c.dataset.decision;if(s)s.textContent='Not reviewed'}}function setDecision(id,d){const x=load();x[id]={decision:d,reviewed_at:new Date().toISOString()};save(x);apply(id,d);toast('Marked '+d)}function clearDecision(id){const x=load();delete x[id];save(x);apply(id,'');toast('Cleared')}function exportDecisions(){const x=load();const rows=[['review_id','decision','reviewed_at']];Object.keys(x).sort().forEach(id=>rows.push([id,x[id].decision||'',x[id].reviewed_at||'']));const csv=rows.map(r=>r.map(c=>'"'+String(c).replaceAll('"','""')+'"').join(',')).join('\n');document.getElementById('exportBox').value=csv;copyText(csv).then(ok=>toast(ok?'Copied decisions':'Export ready'))}function copyAllKeptPaths(){const x=load();const ids=new Set(Object.keys(x).filter(id=>x[id].decision==='keep'));const paths=[];document.querySelectorAll('.card').forEach(c=>{if(ids.has(c.dataset.reviewId)&&c.dataset.path)paths.push(c.dataset.path)});const text=paths.join('\n');document.getElementById('exportBox').value=text;copyText(text).then(ok=>toast(ok?'Copied '+paths.length+' kept paths':'Kept paths ready'))}function clearAllDecisions(){if(!confirm('Clear all decisions in this browser?'))return;localStorage.removeItem(KEY);document.querySelectorAll('.card').forEach(c=>delete c.dataset.decision);document.querySelectorAll('.status').forEach(s=>s.textContent='Not reviewed');toast('All decisions cleared')}Object.keys(load()).forEach(id=>apply(id,load()[id].decision));
</script></body></html>"""
    page = page.replace("__TITLE__", html.escape(title))
    page = page.replace("__TOTAL__", str(len(df)))
    page = page.replace("__LOW__", str(low_conf))
    page = page.replace("__FLAGS__", str(flagged))
    page = page.replace("__TOC__", "".join(toc))
    page = page.replace("__SECTIONS__", "".join(sections))
    path.write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build an alpha HTML review cockpit from master_gallery_images.csv")
    parser.add_argument("project_root")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--title", default="Archive Review Alpha")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-per-bucket", type=int, default=500)
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    input_path = Path(args.input).expanduser()
    output_root = Path(args.output).expanduser()
    if not input_path.is_absolute():
        input_path = root / input_path
    if not output_root.is_absolute():
        output_root = root / output_root
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    df = prepare(pd.read_csv(input_path))
    if args.limit > 0:
        df = df.head(args.limit).copy()

    write_queue(df, output_root / "review_queue.csv")
    build_html(df, output_root / "index.html", args.title, args.max_per_bucket)
    metrics = {
        "total_images": int(len(df)),
        "bucket_counts": {str(k): int(v) for k, v in df["review_bucket"].value_counts().to_dict().items()},
        "low_confidence_count": int((pd.to_numeric(df["mapping_confidence"], errors="coerce").fillna(0) < 0.70).sum()),
        "review_flag_count": int(df["review_flags"].fillna("").astype(str).str.strip().ne("").sum()),
    }
    (output_root / "review_alpha_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nArchive Review Alpha built.")
    print(f"Input CSV:       {input_path}")
    print(f"Output folder:   {output_root}")
    print(f"HTML review:     {output_root / 'index.html'}")
    print(f"Review queue:    {output_root / 'review_queue.csv'}")
    print(f"Metrics JSON:    {output_root / 'review_alpha_metrics.json'}")
    print("\nOpen index.html in your browser and start reviewing.")


if __name__ == "__main__":
    main()
