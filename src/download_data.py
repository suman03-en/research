import os
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Download PlantDoc dataset from Kaggle")
    parser.add_argument("--dataset", type=str, default="nirmalsankalana/plantdoc-dataset", help="Kaggle dataset identifier")
    parser.add_argument("--dest", type=str, default="data/", help="Destination folder for the dataset")
    args = parser.parse_args()

    print(f"Downloading dataset '{args.dataset}' to '{args.dest}'...")
    
    from kaggle.api.kaggle_api_extended import KaggleApi
    
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    os.makedirs(args.dest, exist_ok=True)
    
    # Download dataset (without unzipping to avoid MemoryError on large files)
    import requests
    try:
        api.dataset_download_files(args.dataset, path=args.dest, unzip=False)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            print(f"\nERROR: 403 Forbidden.")
            print(f"You must accept the rules/terms for this dataset on the Kaggle website before downloading via API.")
            print(f"Please visit: https://www.kaggle.com/datasets/{args.dataset}")
            print(f"Log in with your account, click 'Download' to accept the terms, then cancel the browser download and retry this script.")
            exit(1)
        else:
            raise e
            
    # Extract the zip file manually to prevent high memory usage
    import zipfile
    import glob
    print("Extracting dataset...")
    zip_files = glob.glob(os.path.join(args.dest, "*.zip"))
    if zip_files:
        dataset_zip = zip_files[0]
        with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
            zip_ref.extractall(args.dest)
        print(f"Extracted to {args.dest}")
        # Optionally, remove the zip file to save space
        os.remove(dataset_zip)
    else:
        print("Warning: No zip file found to extract.")
    
    print("Download and extraction complete!")

if __name__ == "__main__":
    main()
