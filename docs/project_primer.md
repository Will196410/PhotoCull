# Master Gallery Consolidation Script Spec

## Purpose

This document defines the next major build step for the photo-culling project: a script that reads annual theme outputs and produces a consolidated cross-year master gallery view.

This is the bridge between:

* annual diary discovery,
* category consolidation,
* a reusable master gallery.

The annual theme script already discovers useful clusters within a year.
What is missing is the layer that combines those results across years.

---

## Goal

Build a script that:

1. reads annual outputs from `theme_output/`,
2. loads `*_themes.csv` and `*_images.csv` for each year,
3. maps rough annual discovery labels into approved master categories,
4. supports one primary category and optional secondary categories,
5. flags likely misclassifications,
6. produces consolidated CSV outputs,
7. optionally builds a reviewable HTML gallery grouped by master category.

---

## Canonical Inputs

### Project root

The project should assume a configurable project root.

Example on the current system:

```text
/Volumes/All Photos/
```

### Consolidation input root

```text
<PROJECT_ROOT>/theme_output/
```

Example structure:

```text
theme_output/
├── 2001/
│   ├── 2001_gallery.html
│   ├── 2001_images.csv
│   ├── 2001_themes.csv
│   └── thumbs/
├── 2008/
│   ├── 2008_gallery.html
│   ├── 2008_images.csv
│   ├── 2008_themes.csv
│   └── thumbs/
└── ...
```

---

## Files the script should use

### Required

* `*_images.csv`
* `*_themes.csv`

### Ignore by default

* `*_gallery.html`
* `thumbs/`
* `*_failed.txt`

These may still be useful later for review, but they are not required as primary machine inputs.

---

## Core Inputs from Annual Files

### From `*_images.csv`

Useful fields likely include:

* `cluster_id`
* `theme_name`
* `display_theme_name`
* `file`
* `path`
* `relative_path`
* `archive_relative_path`
* `folder`
* `thumb`
* `dominant_folder`
* `theme_top_label_1`
* `theme_top_label_2`
* `theme_top_label_3`

### From `*_themes.csv`

Useful fields likely include:

* `cluster_id`
* `theme_name`
* `display_theme_name`
* `image_count`
* `top_label_1`
* `top_label_2`
* `top_label_3`
* `dominant_folder`
* `representative_1`
* `representative_2`
* `representative_3`

---

## Approved Master Categories

The consolidation layer should map annual discovery labels into this current master set:

* Landscape
* Waterside and Harbour
* Nature Detail
* Wildlife
* Farm Animals
* People and Human Presence
* Place and Travel
* Rural Life and Working Country
* Weather, Light, and Atmosphere

This list may evolve, but the script should start with it.

---

## Required Classification Rules

### 1. One primary category per image

Each image gets one primary master category.

This is the category that best reflects the main viewing experience.

### 2. Optional secondary categories

Each image may also receive zero, one, or more secondary categories.

These help discovery and gallery reuse.

### 3. People-first rule

If the human subject is the main reason the image matters, use:

* primary = `People and Human Presence`

Possible secondary categories may still include:

* Place and Travel
* Landscape
* Waterside and Harbour
* Rural Life and Working Country
* Weather, Light, and Atmosphere

### 4. Setting-first rule

If the location or environment matters more than the people, use the environmental category as primary.

### 5. Wildlife is not farm animals

The script must keep these separate.

Examples:

* bear → Wildlife
* deer → Wildlife
* sheep → Farm Animals
* cows → Farm Animals

If uncertain, broad but correct is better than narrow and wrong.

### 6. Travel-snapshot safety rule

If an image records a place but does not strongly belong elsewhere, use:

* primary = `Place and Travel`

### 7. Atmosphere can be primary or secondary

Weather, mist, dramatic light, and seasonal mood may be primary when they are the main point of the image, otherwise secondary.

---

## Mapping Logic

The script should use a mapping layer based on combinations of:

* `theme_name`
* `theme_top_label_1`
* `theme_top_label_2`
* `theme_top_label_3`
* keyword matches in `display_theme_name`
* optional keyword matches in `relative_path` or `folder`

### Suggested order of operations

1. load annual image rows,
2. normalise strings,
3. apply explicit mapping rules,
4. assign primary master category,
5. assign secondary categories,
6. flag uncertain rows,
7. write outputs.

---

## Mapping Strategy

### Tier 1: explicit exact mappings

Map known annual labels directly.

Examples:

* `bird or wildlife` → Wildlife
* `farm animal` → Farm Animals
* `coastal landscape` → Landscape
* `harbour or port with boats` → Waterside and Harbour
* `flower or plant close-up` → Nature Detail
* `travel snapshot of a place` → Place and Travel
* `people or group` → People and Human Presence
* `portrait of one person` → People and Human Presence

### Tier 2: keyword-based corrections

Use simple rules to catch obvious adjustments.

Examples:

* if a label contains `harbour`, `port`, `boat`, `pier`, or `quay`, promote Waterside and Harbour
* if a label contains `bear`, `deer`, `fox`, `seal`, or `wildlife`, promote Wildlife
* if a label contains `sheep`, `cow`, `goat`, `pig`, or `farm`, promote Farm Animals
* if a label contains `musician`, `judge`, `police`, `people`, `portrait`, `crowd`, or `group`, promote People and Human Presence

### Tier 3: fallback mapping

If no better rule applies, map from top labels or fallback annual theme name.

### Tier 4: unresolved bucket

If the script still cannot classify confidently, set:

* primary category = `Place and Travel` or `Uncertain`
* add a flag for manual review

A review flag is better than silent bad classification.

---

## Misclassification Flags

The script should produce review flags for likely problems.

Examples:

* `possible_wildlife_farm_conflict`
* `possible_people_place_conflict`
* `possible_atmosphere_primary_conflict`
* `low_mapping_confidence`

This matters because the script should help reduce errors, not hide them.

---

## Proposed Outputs

### 1. Consolidated image-level CSV

Suggested file:

```text
master_gallery_images.csv
```

Suggested fields:

* `year`
* `cluster_id`
* `file`
* `path`
* `archive_relative_path`
* `relative_path`
* `theme_name`
* `display_theme_name`
* `theme_top_label_1`
* `theme_top_label_2`
* `theme_top_label_3`
* `primary_master_category`
* `secondary_master_categories`
* `mapping_confidence`
* `review_flags`
* `thumb`

### 2. Consolidated category summary CSV

Suggested file:

```text
master_gallery_categories.csv
```

Suggested fields:

* `master_category`
* `image_count`
* `years_present`
* `top_source_themes`
* `example_images`

### 3. Review flags CSV

Suggested file:

```text
master_gallery_review_flags.csv
```

Suggested fields:

* `year`
* `file`
* `path`
* `display_theme_name`
* `primary_master_category`
* `review_flags`
* `notes`

### 4. Optional HTML gallery

Suggested file:

```text
master_gallery.html
```

Grouped by master category, then optionally by year or source theme.

---

## Output Location

Recommended default output root:

```text
<PROJECT_ROOT>/theme_output/master_gallery/
```

Example:

```text
/Volumes/All Photos/theme_output/master_gallery/
```

This keeps consolidation outputs close to the annual theme outputs.

---

## Script Interface

Suggested command pattern:

```bash
python consolidate_themes.py "/Volumes/All Photos"
```

Suggested arguments:

`project_root`
: root containing `Photos/`, `dedupe_output/`, and `theme_output/`

`--theme-output-root`
: optional override for annual theme folders

`--output-root`
: optional override for consolidated outputs

`--years`
: optional comma-separated list or repeated argument to restrict processing

`--include-html`
: whether to build the consolidated HTML gallery

`--strict`
: optionally fail on missing annual files instead of warning and skipping

---

## Missing-Data Behaviour

The script should handle partial year coverage gracefully.

If a year folder is missing one of the needed CSV files:

* warn clearly,
* skip the year by default,
* continue processing other valid years.

Do not crash unless running in strict mode.

---

## Manual Override Support

This will likely become important.

A future-friendly design would allow:

* a simple CSV mapping override file,
* manual category corrections,
* custom keyword rules,
* manual image exclusions.

Even if version 1 does not implement all of this, the script should not make it awkward to add later.

---

## First-Version Priorities

For the first usable version, focus on:

1. load all annual outputs,
2. combine image rows,
3. map to master categories,
4. support primary and secondary categories,
5. flag doubtful classifications,
6. write consolidated CSVs.

The HTML gallery can come next if needed.

---

## What Success Looks Like

The first version succeeds if it lets you answer questions like:

* How many images across all years map to each master category?
* Which annual labels are feeding each master category?
* Which years are strongest for Wildlife, Harbour, or People?
* Which images need review because the mapping looks doubtful?
* Which categories are too thin or too messy and need refinement?

That is enough to move from annual browsing toward a real master gallery.

---

## Likely Future Enhancements

After the first version works, useful upgrades would include:

* stronger keyword and synonym handling,
* image ranking within master categories,
* best-of-category promotion,
* HTML review UI for manual correction,
* duplicate awareness using `dedupe_output`,
* explicit support for subcategories like musicians, police, judges, city, beach, countryside.

---

## Immediate Next Action

Implement version 1 of `consolidate_themes.py` using the folder layout and outputs already in place.
