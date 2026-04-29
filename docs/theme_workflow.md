# PhotoCull Workflow    
# Locations  
Archive of photos eg:  
/Volumes/All Photos/Photos  
= image source only  
  
Local repo/scripts:  
theme_output/  
theme_output/master_gallery/  
= generated analysis outputs  
    
# Workflow for creating theme website  
  
## 1. photo_dedupe.py  
Reduce the number of duplicate photos.
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

python3 photo_dedupe.py "/Volumes/All Photos/Photos" \  
  --outdir theme_output/dedupe_output \  
  --verbose  
  
```
Reports written to:    /scripts/theme_output/dedupe_output

```
  
  
## 2. python photo_themes.py  
Search out the themes.
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
Images CSV:  scripts/theme_output/2023/2023_images.csv  
Themes CSV:  scripts/theme_output/2023/2023_themes.csv  
Gallery:     scripts/theme_output/2023/2023_gallery.html  
  
  
  
## 3. consolidate_themes.py and consolidate_themes_dev.py  
Consolidate the themes.
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
Master images CSV:          scripts/theme_output/master_gallery/master_gallery_images.csv  
Master categories CSV:      scripts/theme_output/master_gallery/master_gallery_categories.csv  
Review flags CSV:           scripts/theme_output/master_gallery/master_gallery_review_flags.csv  
Mapping diagnostics CSV:    scripts/theme_output/master_gallery/master_gallery_mapping_diagnostics.csv  
Suspicious mappings CSV:    scripts/theme_output/master_gallery/master_gallery_suspicious_mappings.csv  
Category/theme matrix CSV:  scripts/theme_output/master_gallery/master_gallery_category_theme_matrix.csv  
Suspicious mappings:       none found  
Total images consolidated: 2177  
Categories present:        8  
Audit sample CSV:           scripts/theme_output/master_gallery/master_gallery_audit_sample.csv  
Metrics JSON:               scripts/theme_output/master_gallery/master_gallery_metrics.json  
Audit HTML:                 scripts/theme_output/master_gallery/master_gallery_audit.html  
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
  
## 4. evaluate_gold_labels.py  
Evaluate the themes to enable creation of the views. Gold refers to labels used for tuning the categorisation engine. Work in progress. 
```
usage: evaluate_gold_labels.py [-h] [--output OUTPUT] [--confusion-output CONFUSION_OUTPUT] gold_labels_csv master_gallery_images_csv  
```
```bash  
python3 evaluate_gold_labels.py \  
gold_labels_v1.csv \  
theme_output/master_gallery/master_gallery_images.csv  
```

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
  
Comparison CSV:    scripts/gold_label_evaluation.csv  
Confusion matrix:  scripts/gold_label_confusion_matrix.csv  
  
  
  
## 5. python archive_review_alpha.py . --limit 1000    
Creates the views. Alpha refers to the state of the categorisation engine. I spent days tuning it with ChatGPT. Perfection is the enemy of the good enough. The contents of the category pages aren't perfect, but still useful. 
Produces:  
Archive Review Alpha built.  
```
Input CSV:        scripts/theme_output/master_gallery/master_gallery_images.csv
Output folder:    scripts/theme_output/archive_review_alpha
HTML review:      scripts/theme_output/archive_review_alpha/index.html
Review queue:     scripts/theme_output/archive_review_alpha/review_queue.csv
Metrics JSON:     scripts/theme_output/archive_review_alpha/review_alpha_metrics.json

Open index.html in your browser and start reviewing.

```
  
  
  
  
