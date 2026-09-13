"""
download_data.py
----------------
Downloads the Zomato Restaurants dataset from Kaggle using kagglehub
and saves the raw CSV to data/raw/.

Dataset: Zomato Restaurants Dataset
Source : https://www.kaggle.com/datasets/shrutimehta/zomato-restaurants-data
Columns include: Restaurant ID, Restaurant Name, City, Address, Locality,
                 Cuisines, Average Cost for two, Currency, Has Table booking,
                 Has Online delivery, Is delivering now, Price range,
                 Aggregate rating, Rating color, Rating text, Votes
"""

import os
import shutil
import glob

# ── Try kagglehub first (already in requirements.txt) ─────────────────────────
def download_via_kagglehub():
    import kagglehub  # type: ignore
    print("Downloading Zomato dataset via kagglehub …")
    path = kagglehub.dataset_download("shrutimehta/zomato-restaurants-data")
    print(f"  Files downloaded to: {path}")
    return path

# ── Fallback: opendatasets ────────────────────────────────────────────────────
def download_via_opendatasets():
    import opendatasets as od  # type: ignore
    url = "https://www.kaggle.com/datasets/shrutimehta/zomato-restaurants-data"
    print("Downloading Zomato dataset via opendatasets …")
    od.download(url, data_dir="data/raw")
    return "data/raw/zomato-restaurants-data"

# ── Copy CSV files into data/raw/ ─────────────────────────────────────────────
def copy_csvs(src_dir: str, dest_dir: str = "data/raw") -> list:
    os.makedirs(dest_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(src_dir, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {src_dir}")
    copied = []
    for f in csv_files:
        dest = os.path.join(dest_dir, os.path.basename(f))
        shutil.copy2(f, dest)
        print(f"  Copied {os.path.basename(f)} → {dest}")
        copied.append(dest)
    return copied


if __name__ == "__main__":
    # ── Step 1: download ──────────────────────────────────────────────────────
    try:
        src = download_via_kagglehub()
    except Exception as e:
        print(f"kagglehub failed ({e}), trying opendatasets …")
        try:
            src = download_via_opendatasets()
        except Exception as e2:
            print(
                f"\nBoth download methods failed.\n"
                f"  kagglehub error   : {e}\n"
                f"  opendatasets error: {e2}\n\n"
                "Manual download instructions:\n"
                "  1. Visit https://www.kaggle.com/datasets/shrutimehta/zomato-restaurants-data\n"
                "  2. Click 'Download' and extract the ZIP.\n"
                "  3. Place all CSV files inside  data/raw/\n"
            )
            raise SystemExit(1)

    # ── Step 2: copy CSVs ─────────────────────────────────────────────────────
    copied = copy_csvs(src)
    print(f"\nDataset ready.  {len(copied)} CSV file(s) in data/raw/")
    for f in copied:
        print(f"  {f}")
