# PhotoCull
Python scripts for culling photos on an SSD

## Prerequisites

Before running the scripts, ensure you have **Python 3.8+** installed. If you are using an Apple Silicon Mac (M1/M2/M3), these scripts are optimized to use the **MPS (Metal Performance Shaders)** acceleration.

## Installation

You can install all the required dependencies using `pip`. It is recommended to use a virtual environment.

```bash
# Clone the repository
git clone https://github.com/Will196410/PhotoCull.git
cd PhotoCull

# (Optional) Create and activate a virtual environment
python -m venv venv # creates the virtual environment, which is a folder. Do this only once.
source venv/bin/activate  # On Windows use: venv\Scripts\activate. Do this after each reboot.

# Install dependencies
pip install torch torchvision torchaudio
pip install transformers pillow rawpy pandas tqdm
```

Install libraries with command:
`pip install -r requirements.txt`

## Library Overview

The scripts rely on the following powerful libraries:

| Library | Purpose |
| :--- | :--- |
| **PyTorch** (`torch`) | The deep learning framework used to run the CLIP model. |
| **Transformers** | Provides the pre-trained **CLIP** model (by OpenAI) for image and text understanding. |
| **RawPy** | A wrapper for `libraw` that allows the script to process professional RAW files (.ARW, .CR2, .DNG, etc.). |
| **Pillow** (`PIL`) | The standard Python imaging library for opening and resizing photos. |
| **Pandas** | Used to organize the scoring data and export results to CSV. |
| **Tqdm** | Provides the visual progress bar in the terminal during the scan. |

## Usage

To run the AI judge for fine art landscape selection:

```bash
python cullerV2.py "/Volumes/All Photos/Photos/2007" --mode landscape   
```

This creates a file called landscape_candidates.csv.

There is a dedicated culler for stock photogrpahy:
```bash
python stock_culler.py "/Volumes/Photos" --batch-size 16 --top 30 --min-score 0.40
```
Both scripts offer this parameter:
--batch-size is how many images the script scores at once in a single model pass.
So:
* --batch-size 1 = one image at a time
* --batch-size 16 = sixteen images at a time
* --batch-size 32 = thirty-two at a time

Why it matters:
* larger batch usually runs faster
* but uses more memory
* if it is too large, you can get slowdowns or memory errors

To create a gallery: 
```bash
python3 gallery_pro.py landscape_candidates.csv --threshold 10 --limit 500
```
This copies the top 500 matches to a thumbnail directory. Threshold determines how alike images have to be excluded. 1 means exact match. 10 is aggressive. It will exclude similar images. 

The script, gallery.py, is a helper for this script. 


## Identify Themes

Libraries Required:
```bash
pip install torch transformers pillow rawpy pandas tqdm scikit-learn
```

Simple version:
```bash
python photo_themes.py "/Volumes/All Photos/Photos" --year 2008
```

## Theme with deduplicated Galleries.

**photo_dedupe.py**
Installatin:
Inside the virtual environment:
```bash
python -m pip install pillow imagehash
```
To run:
```bash
python photo_dedupe.py "/Volumes/All Photos/Photos" --outdir "/Volumes/All Photos/dedupe_output" --verbose
```
Produces:
/Volumes/All Photos/dedupe_output/dedupe_exact.csv
/Volumes/All Photos/dedupe_output/dedupe_groups.csv
/Volumes/All Photos/dedupe_output/gallery_excludes.txt
/Volumes/All Photos/dedupe_output/dedupe_summary.json

How to use it with your gallery builder

The cleanest way is:

1. run photo_dedupe.py
2. load gallery_excludes.txt in your gallery builder
3. skip any file whose relative path appears there


