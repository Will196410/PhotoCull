# Photo Culling

A Python-based workflow for exploring large photo archives year by year, discovering recurring visual themes, and preparing a curated master gallery from annual outputs.

This project is designed around two stages:

1. **Annual discovery**: scan one year of photos, cluster visually similar images, and assign rough descriptive themes.
2. **Master-gallery consolidation**: combine annual outputs into a smaller, more stable set of categories suitable for long-term galleries, website use, and possible print selection.

The annual stage helps reveal what is actually in the archive.
The consolidation stage turns that into a cleaner gallery structure.

---

## What this project does

The current workflow:

* scans a single year of a photo archive,
* ignores unsupported and obviously irrelevant files,
* generates embeddings for images using CLIP,
* clusters similar images,
* assigns rough discovery labels using a fixed list of prompts,
* creates CSV outputs for themes and images,
* builds an HTML gallery for review,
* writes thumbnails for quick browsing.

This is a discovery-first system. The labels are useful for surfacing material, but they are not yet a polished final taxonomy.

---

## Why this exists

Browsing large archives manually is slow and inconsistent. The annual theme workflow makes it easier to answer questions like:

* What kinds of images do I actually have from this year?
* Which subjects recur across multiple years?
* Which images are worth surfacing for a website gallery?
* Which categories are strong enough to keep long-term?
* Which files are strong candidates for print or public display?

The aim is not just technical sorting. It is practical curation.

---

## Project layout

The scripts can live anywhere. The project works best when the photo archive and generated outputs live under one common project root.

Example layout:

```text
<PROJECT_ROOT>/
├── Photos/
├── dedupe_output/
└── theme_output/
    ├── 2001/
    ├── 2002/
    ├── 2008/
    └── ...
```

Example project root on my system:

```text
/Volumes/All Photos/
```

That is only an example. Other users can choose any project root they like.

### Folder roles

* `Photos/` — raw photo archive, typically organised by year
* `dedupe_output/` — outputs from deduplication or related filtering steps
* `theme_output/` — annual theme-discovery outputs grouped by year

---

## Annual output structure

Each processed year gets its own folder inside `theme_output/`.

Example:

```text
theme_output/2008/
├── 2008_gallery.html
├── 2008_images.csv
├── 2008_themes.csv
├── 2008_failed.txt
└── thumbs/
```

### Output files

`*_images.csv`
: image-level output. Includes file path, relative path, theme label, cluster id, thumbnail path, and related metadata.

`*_themes.csv`
: cluster-level output. Summarises each discovered theme, cluster size, top prompt matches, dominant folder, and representative images.

`*_gallery.html`
: human-review output. Lets you browse the year visually by cluster/theme and copy full file paths from thumbnails.

`*_failed.txt`
: diagnostic output listing files that could not be loaded.

`thumbs/`
: thumbnails used by the HTML gallery.

---

## Installation

This project uses Python and a small set of image, machine-learning, and data-processing libraries.

### Basic setup

Create and activate a virtual environment, then install the pinned dependencies from `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Main dependencies

The project currently relies on packages including:

* `pandas`
* `rawpy`
* `torch`
* `Pillow`
* `scikit-learn`
* `tqdm`
* `transformers`

### Notes on Torch

`torch` can be the package most likely to vary by platform.

Examples:

* Apple Silicon Macs may use MPS acceleration,
* NVIDIA systems may use CUDA,
* some users may need to follow platform-specific PyTorch installation guidance if the pinned install does not work cleanly in their environment.

The scripts will try to choose the best available device automatically.

### Model download

The annual theme script uses the CLIP model:

* `openai/clip-vit-base-patch32`

This will be downloaded automatically by Hugging Face the first time the script runs, if it is not already present locally.

## Running the annual theme script

Example command:

```bash
python photo_themes.py "/Volumes/All Photos/Photos" --year 2008
```

This scans the `2008` folder under the supplied archive root and writes outputs to the configured theme-output root.

### Common arguments

`root`
: archive root. Example: `/Volumes/All Photos/Photos`

`--year`
: required. The year folder to process.

`--batch-size`
: number of images per inference batch.

`--distance-threshold`
: controls cluster tightness. Lower values create tighter clusters; higher values create broader ones.

`--min-cluster-size`
: clusters smaller than this are merged into a miscellaneous bucket.

`--max-images`
: optional test cap.

`--exclude-file`
: optional text file of relative paths to skip.

`--output-root`
: root folder for annual theme outputs. This should usually point at `<PROJECT_ROOT>/theme_output`.

Example with explicit output root:

```bash
python photo_themes.py "/Volumes/All Photos/Photos" --year 2008 --output-root "/Volumes/All Photos/theme_output"
```

---

## Workflow philosophy

This project works best when discovery and curation are kept separate.

### Stage 1: discovery

Use flexible, descriptive labels to understand what is present in a given year.

Examples:

* coastal landscape photograph
* harbour or port scene with boats
* travel snapshot of a place
* bird or wildlife photograph

These do not need to be elegant. They need to help surface good photographs quickly.

### Stage 2: consolidation

Map annual labels into a smaller, more durable set of gallery categories.

Current working master categories:

* Landscape
* Waterside and Harbour
* Nature Detail
* Wildlife
* Farm Animals
* People and Human Presence
* Place and Travel
* Rural Life and Working Country
* Weather, Light, and Atmosphere

This second layer is essential. If annual discovery labels are used directly as permanent gallery categories, the result becomes fragmented and inconsistent.

---

## Category rules

### Primary and secondary categories

A single image may belong to more than one category.

Recommended approach:

* assign one **primary** category for the main viewing experience,
* allow one or more **secondary** categories for discovery and reuse.

Example:

* a musician on a harbour wall

  * primary: `People and Human Presence`
  * secondary: `Waterside and Harbour`

### People and place

Images with people should not automatically disappear into generic travel or street-scene buckets.

Use `People and Human Presence` when the human role, activity, or expression is the real subject.

Examples:

* musicians
* police officers
* judges
* beachgoers
* people in the city
* people in the countryside

### Wildlife versus farm animals

This distinction matters.

* wild animals belong in `Wildlife`
* domesticated or agricultural animals belong in `Farm Animals`

A bear should never be classified as a farm animal.

When uncertain, broader but correct is better than narrow and wrong.

---

## Current limitations

This project is useful, but it is not finished.

Known limitations include:

* discovery labels can still be messy or overlapping,
* some categories are too broad or too prompt-dependent,
* wildlife and farm-animal classification may still need manual correction,
* people/place images often require judgment about what the image is really about,
* annual outputs are not yet the same thing as a final master gallery,
* the cross-year consolidation layer is still under active design.

These are not minor details. They directly affect gallery quality.

---

## What counts as success

A successful output is not just a set of clusters. It should help identify:

* visually strong images,
* recurring subjects across years,
* categories worth keeping in a master gallery,
* candidates for website display,
* candidates for prints or stock-like reuse,
* weak or over-fragmented categories that should be merged.

This is ultimately a curation tool, not just an archive-analysis toy.

---

## Roadmap

Planned next steps:

1. **Master-gallery consolidation script**

   * read all annual outputs under `theme_output/`
   * map raw annual labels to approved master categories
   * support primary and secondary categories
   * flag likely misclassifications

2. **Theme-mapping layer**

   * formalise translation from annual discovery labels to master categories
   * improve consistency across years

3. **Better category handling for people**

   * distinguish role-based and setting-based human images
   * keep people-first images from being buried in place categories

4. **Improve animal classification**

   * reduce false labels such as wild animals being classified as farm animals

5. **Consolidated gallery output**

   * build one cross-year report or gallery from the annual results

6. **Ranking / promotion logic**

   * surface strongest candidate images within a theme instead of just all cluster members

---

## Suggested repository structure

A clean repo layout would look something like this:

```text
photo-culling/
├── README.md
├── scripts/
│   ├── photo_themes.py
│   └── ...
└── docs/
    ├── project_primer.md
    └── theme_mapping.md
```

Suggested roles:

* `README.md` — public-facing overview and usage guide
* `docs/project_primer.md` — fuller working brief and rationale
* `docs/theme_mapping.md` — master-category rules and mapping logic

---

## Further documentation

- [Project Primer](docs/project_primer.md)
- [Theme Mapping](docs/theme_mapping.md)
- [Consolidation Script Spec](docs/consolidation_script_spec.md)

---

## Notes for contributors

The current system has been shaped by practical use on a large personal archive. That means some defaults reflect one real-world setup, especially around paths and folder layout. Those defaults should remain configurable.

Design rule:

* provide sensible defaults,
* avoid assumptions that cannot be overridden.

---

## Status

The annual theme-discovery stage is working.
The project brief and category rules are established.
The next major milestone is the consolidation layer that turns annual outputs into a true master gallery.
