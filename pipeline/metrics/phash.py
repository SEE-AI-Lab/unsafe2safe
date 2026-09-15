import argparse
import csv
import os
from multiprocessing import Pool, cpu_count

import imagehash
from PIL import Image
from tqdm import tqdm


def compute_phash(image_path):
    """Compute a perceptual hash for one image."""
    try:
        with Image.open(image_path) as image:
            return str(imagehash.phash(image))
    except Exception as error:
        print(f"Error processing {image_path}: {error}")
        return None


def compute_phash_worker(args):
    image_path, root_path = args
    hash_value = compute_phash(image_path)
    if hash_value is None:
        return None
    return os.path.relpath(image_path, root_path), hash_value


def hash_folder_images(folder_path, output_csv="phash_results.csv", num_workers=None):
    """Hash all images below a folder and save relative paths to a CSV file."""
    if num_workers is None:
        num_workers = cpu_count()

    image_files = []
    for root, _, filenames in os.walk(folder_path):
        image_files.extend((os.path.join(root, filename), folder_path) for filename in filenames)

    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(compute_phash_worker, image_files), total=len(image_files), desc="Hashing images"))

    with open(output_csv, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "phash"])
        writer.writerows(result for result in results if result is not None)

    print(f"Saved hashes to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute perceptual hashes for images in a folder.")
    parser.add_argument("folder", help="Path to a folder containing images")
    parser.add_argument("--output", default="phash_results.csv", help="Output CSV path")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    args = parser.parse_args()
    hash_folder_images(args.folder, args.output, num_workers=args.workers)
