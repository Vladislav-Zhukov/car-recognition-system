import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

SOURCE_DIR = BASE_DIR / "dataset" / "car_data"
TARGET_DIR = BASE_DIR / "dataset" / "car_data_subset"

CLASSES = [
    "Audi S4 Sedan 2012",
    "Audi TT RS Coupe 2012",
    "BMW M3 Coupe 2012",
    "BMW X5 SUV 2007",
    "Chevrolet Corvette Convertible 2012",
    "Dodge Challenger SRT8 2011",
    "Ford Mustang Convertible 2007",
    "Mercedes-Benz C-Class Sedan 2012",
    "Porsche Panamera Sedan 2012",
    "Tesla Model S Sedan 2012",
]


def copy_split(split: str):
    source_split = SOURCE_DIR / split
    target_split = TARGET_DIR / split

    target_split.mkdir(parents=True, exist_ok=True)

    for class_name in CLASSES:
        src = source_split / class_name
        dst = target_split / class_name

        if not src.exists():
            print(f"Not found: {src}")
            continue

        if dst.exists():
            shutil.rmtree(dst)

        shutil.copytree(src, dst)
        print(f"Copied {split}: {class_name}")


def main():
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)

    copy_split("train")
    copy_split("test")

    print("\nSubset created:")
    print(TARGET_DIR)


if __name__ == "__main__":
    main()
