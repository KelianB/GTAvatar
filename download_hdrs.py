"""
polyhaven.com provides a convenient API for downloading free HDRI maps.
Please do not abuse this and only download once.  
"""

import argparse
from pathlib import Path
import os
import time
import requests

from utils.tqdm import tqdm

BASE_URL = "https://api.polyhaven.com"

def fetch_asset_list():
    """Fetch the list of assets from the API."""
    response = requests.get(f"{BASE_URL}/assets?type=hdris")
    response.raise_for_status()
    return response.json()

def fetch_asset_info(asset_name: str):
    """Fetch file info for a specific asset."""
    response = requests.get(f"{BASE_URL}/files/{asset_name}")
    response.raise_for_status()
    return response.json()

def download_file(url: str, output_path: str, size: int, text=""):
    """Download a file with a progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(output_path, "wb") as file, tqdm(total=size, unit="B", unit_scale=True, desc=text) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
            pbar.update(len(chunk))

def main():
    parser = argparse.ArgumentParser(description="Fetch and download HDRI maps from polyhaven.com")
    parser.add_argument("--output_dir", default="downloads", help="Directory to save downloaded files.")
    parser.add_argument("--delay", type=float, default=0.25, help="Delay between API requests in seconds.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching assets...")
    assets = fetch_asset_list()
    print(f"Found {len(assets)} assets")
    assets_name = list(assets.keys())
    max_name_length = max([len(s) for s in assets_name])

    for i, asset_name in enumerate(assets_name):
        time.sleep(args.delay) # Delay requests
        file_info = fetch_asset_info(asset_name)
        text = f"{asset_name:>{max_name_length}} ({i+1}/{len(assets)})"

        # Extract URL and size for the HDR file
        try:
            hdri_info = file_info["hdri"]["1k"]["hdr"]
            file_url = hdri_info["url"]
            file_size = hdri_info["size"]
        except KeyError:
            print(f"Warning: HDR file info not found for asset {asset_name}.")
            continue

        # Define the output file path
        output_path: Path = output_dir / os.path.basename(file_url)

        # Check if the file has already been downloaded        
        if output_path.exists() and output_path.stat().st_size == file_size:
            print(f"{text}: skipping (already downloaded)")
            continue

        time.sleep(args.delay) # Delay requests
        download_file(file_url, output_path, file_size, text=text)

    print("All downloads complete.")

if __name__ == "__main__":
    main()