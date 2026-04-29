# Object Search Workflow
I want to look for particular objects in my photo archive. It depends on [photo theme](https://github.com/Will196410/PhotoCull/blob/main/docs/theme_workflow.md) workflow to know which photos to search.
```
theme_output/master_gallery/master_gallery_images.csv
```
Here's the script.
```
python archive_object_search_alpha.py . --queries "cars,bicycles,men,women,tents,night,winter,summer"
```
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

