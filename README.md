# PhotoCull
Python scripts for culling photos on an SSD

## Prerequisites

Before running the scripts, ensure you have **Python 3.8+** installed. If you are using an Apple Silicon Mac (M1/M2/M3), these scripts are optimized to use the **MPS (Metal Performance Shaders)** acceleration.

## Installation

You can install all the required dependencies using `pip`. It is recommended to use a virtual environment.

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio
pip install transformers pillow rawpy pandas tqdm
```

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
python culler.py /path/to/your/photos --mode landscape
```

---

### Pro-Tip: The `requirements.txt` file
To make your repo even more professional, I recommend creating a file named `requirements.txt` in your root folder with this content:

```text
torch
torchvision
torchaudio
transformers
rawpy
pillow
pandas
tqdm
```

Then, you can update your README installation instructions to just one simple command:
`pip install -r requirements.txt`

How many photos are you planning to run through this at once? If it's thousands, we might want to look at adding a "batching" tweak to the code to prevent memory spikes!
