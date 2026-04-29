# Object Search Workflow
I want to look for particular objects in my photo archive. It depends on [photo theme](https://github.com/Will196410/PhotoCull/blob/main/docs/theme_workflow.md) workflow to know which photos to search.
```
theme_output/master_gallery/master_gallery_images.csv
```
## The Script
```
python archive_object_search_alpha.py . --queries "cars,bicycles,men,women,tents,night,winter,summer"
```
Quick test:
```
python archive_object_search_alpha.py . \
  --queries "cars,bicycles,tents,night,winter" \
  --limit 500 \
  --top-k 50
```
Separate out the top 50 after searching 500 images.
## Viewing the results
It produces: 
```
theme_output/archive_object_search_alpha/
  index.html
  search_results.csv
  object_search_metrics.json
  cache/
    image_embeddings.npy
    image_manifest.csv
  queries/
    cars.html
    bicycles.html
    men.html
    women.html
    tents.html
    night.html
    winter.html
    summer.html
```
_Open the index.html file in theme_output/archive_object_search_alpha_
```
open theme_output/archive_object_search_alpha/index.html
```
On macOS, you can also run a web server from the script directory.
```
cd theme_output/archive_object_search_alpha
python -m http.server 8000
```
Open a browser. From the terminal:
```
open http://localhost:8000
```


