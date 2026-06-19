import shutil
import subprocess
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_DIR = BASE_DIR / "dataset"
KAGGLE_DIR = DATASET_DIR / "kaggle"

KAGGLE_DATASET = "jutrera/stanford-car-dataset-by-classes-folder"


def download_dataset():
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading Stanford Cars dataset from Kaggle...")

    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            KAGGLE_DATASET,
            "-p",
            str(KAGGLE_DIR),
        ],
        check=True,
    )


def unzip_dataset():
    zip_files = list(KAGGLE_DIR.glob("*.zip"))

    if not zip_files:
        raise FileNotFoundError("Zip file not found after download")

    zip_path = zip_files[0]

    print(f"Unzipping: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATASET_DIR)

    print("Dataset extracted.")


def main():
    download_dataset()
    unzip_dataset()


if __name__ == "__main__":
    main()
