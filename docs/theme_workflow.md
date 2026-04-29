# 2026-04-27 PhotoCull Workflow  
#photography  #AI  #python   
  
  
# Locations  
  
Archive:  
/Volumes/All Photos/Photos  
= image source only  
  
Local repo/scripts:  
theme_output/  
theme_output/master_gallery/  
= generated analysis outputs  
  
# Creating a base-line  
  
```
cd ~/photo-culling-tools/scripts

python3 photo_dedupe.py "/Volumes/All Photos/Photos" \
  --outdir theme_output/dedupe_output \
  --verbose

python3 photo_themes.py "/Volumes/All Photos/Photos" \
  --year 2023 \
  --distance-threshold 0.26 \

```
```
  --exclude-file theme_output/dedupe_output/gallery_excludes.txt \

```
```
  --outdir theme_output/2023

```
```

python3 consolidate_themes_dev.py "/Volumes/All Photos" \
  --theme-output-root theme_output \
  --output-root theme_output/master_gallery

python3 evaluate_gold_labels.py \
  gold_labels_v1.csv \
  theme_output/master_gallery/master_gallery_images.csv

```
  
  
# Archive  
```
/Volumes/All Photos/Photos

```
  
# Local Repo  
/scripts  
/doc  
/theme_output  
/theme_output/master_gallery  
  
  
# Scripts  
  
## photo_dedupe.py  
```
usage: photo_dedupe.py [-h] [--outdir OUTDIR] [--seconds-window SECONDS_WINDOW] [--phash-threshold PHASH_THRESHOLD] [--verbose] root

```
 python photo_dedupe.py "/Volumes/All Photos/Photos" --outdir "/Volumes/All Photos/dedupe_output" --verbose    
  
```
print(f"Photos scanned:        {len(photos)}")
    print(f"Exact duplicate groups:{len(exact_groups)}")
    print(f"Near duplicate groups: {len(near_groups)}")
    print(f"Excluded for gallery:  {len(excluded)}")
    print(f"Reports written to:    {outdir}")

```
  
April 27, 2026: output is in root of All Ph**otos. "/Volumes/All Photos/dedupe_output" **  
  
python3 photo_dedupe.py "/Volumes/All Photos/Photos" \  
  --outdir theme_output/dedupe_output \  
  --verbose  
  
```
Reports written to:    /Users/williammurphy/photo-culling-tools/scripts/theme_output/dedupe_output

```
  
  
## python photo_themes.py  
```
python3 photo_themes.py "/Volumes/All Photos/Photos" \

```
```
  --year 2023 \
  --distance-threshold 0.26 \
  --exclude-file theme_output/dedupe_output/gallery_excludes.txt \
  --output-root theme_output

```
  
Produced:  
Images CSV:  /Users/williammurphy/photo-culling-tools/scripts/theme_output/2023/2023_images.csv  
Themes CSV:  /Users/williammurphy/photo-culling-tools/scripts/theme_output/2023/2023_themes.csv  
Gallery:     /Users/williammurphy/photo-culling-tools/scripts/theme_output/2023/2023_gallery.html  
  
  
  
## consolidate_themes.py and consolidate_themes_dev.py  
```
usage: consolidate_themes_dev.py [-h] [--theme-output-root THEME_OUTPUT_ROOT] [--output-root OUTPUT_ROOT] [--years YEARS] [--prompts-file PROMPTS_FILE] [--rules-file RULES_FILE] [--include-html] [--strict]

```
  
```
python3 consolidate_themes_dev.py "/Volumes/All Photos" \
  --theme-output-root theme_output \
  --output-root theme_output/master_gallery \
  --years 2023 \
  --include-html

```
  
Produced:  
Master images CSV:         /Users/williammurphy/photo-culling-tools/scripts/theme_output/master_gallery/master_gallery_images.csv  
Master categories CSV:     /Users/williammurphy/photo-culling-tools/scripts/theme_output/master_gallery/master_gallery_categories.csv  
Review flags CSV:          /Users/williammurphy/photo-culling-tools/scripts/theme_output/master_gallery/master_gallery_review_flags.csv  
Mapping diagnostics CSV:   /Users/williammurphy/photo-culling-tools/scripts/theme_output/master_gallery/master_gallery_mapping_diagnostics.csv  
Suspicious mappings CSV:   /Users/williammurphy/photo-culling-tools/scripts/theme_output/master_gallery/master_gallery_suspicious_mappings.csv  
Category/theme matrix CSV: /Users/williammurphy/photo-culling-tools/scripts/theme_output/master_gallery/master_gallery_category_theme_matrix.csv  
Suspicious mappings:       none found  
Total images consolidated: 2177  
Categories present:        8  
Audit sample CSV:          /Users/williammurphy/photo-culling-tools/scripts/theme_output/master_gallery/master_gallery_audit_sample.csv  
Metrics JSON:              /Users/williammurphy/photo-culling-tools/scripts/theme_output/master_gallery/master_gallery_metrics.json  
Audit HTML:                /Users/williammurphy/photo-culling-tools/scripts/theme_output/master_gallery/master_gallery_audit.html  
(.venv) williammurphy@Odin scripts %   
  
```
python3 consolidate_themes_dev.py "/Volumes/All Photos" \
  --theme-output-root theme_output \
  --output-root theme_output/master_gallery \
  --years 2012,2013,2014,2018,2019,2024

```
  
```
python3 consolidate_themes_dev.py "/Volumes/All Photos" \
  --theme-output-root theme_output \
  --output-root theme_output/master_gallery

```
  
## evaluate_gold_labels.py  
usage: evaluate_gold_labels.py [-h] [--output OUTPUT] [--confusion-output CONFUSION_OUTPUT] gold_labels_csv master_gallery_images_csv  
  
* python3 evaluate_gold_labels.py \  
*   gold_labels_v1.csv \  
*   theme_output/master_gallery/master_gallery_images.csv  
  
Produces:  
Gold label evaluation  
---------------------  
Gold rows:        200  
Matched rows:     4  
Unmatched rows:   196  
Correct matched:  4  
Accuracy:         1.000  
  
Warning: unmatched gold rows exist. Check path differences.  
  
Largest error patterns:  
  None  
  
Comparison CSV:   /Users/williammurphy/photo-culling-tools/scripts/gold_label_evaluation.csv  
Confusion matrix: /Users/williammurphy/photo-culling-tools/scripts/gold_label_confusion_matrix.csv  
  
  
  
## python archive_review_alpha.py . --limit 1000    
  
Produces:  
Archive Review Alpha built.  
```
Input CSV:       /Users/williammurphy/photo-culling-tools/scripts/theme_output/master_gallery/master_gallery_images.csv
Output folder:   /Users/williammurphy/photo-culling-tools/scripts/theme_output/archive_review_alpha
HTML review:     /Users/williammurphy/photo-culling-tools/scripts/theme_output/archive_review_alpha/index.html
Review queue:    /Users/williammurphy/photo-culling-tools/scripts/theme_output/archive_review_alpha/review_queue.csv
Metrics JSON:    /Users/williammurphy/photo-culling-tools/scripts/theme_output/archive_review_alpha/review_alpha_metrics.json

Open index.html in your browser and start reviewing.

```
  
  
  
  
