"""
References: https://github.com/microsoft/FaceSynthetics
"""

import argparse
import os.path
import shutil
from concurrent.futures import ThreadPoolExecutor


def copy_file(source: str, destination: str):
    assert os.path.exists(source), f"File not found: {source}"
    shutil.copyfile(source, destination)


def main(args):
    data_root = args.data_root
    destination_folder = args.destination_folder
    file_ids_txt = args.file_ids

    # Read file names
    with open(file_ids_txt, mode="r") as f:
        lines = f.readlines()

    file_ids = []
    for file_id in lines:
        fid = file_id.replace("\n", "")
        file_ids.append(fid)

    # Copy files
    landmarks_folder = os.path.join(destination_folder, "landmarks")
    if not os.path.exists(landmarks_folder):
        os.makedirs(landmarks_folder)

    executor = ThreadPoolExecutor(max_workers=6)

    for file_id in file_ids:
        source_path = os.path.join(data_root, f"{file_id}_ldmks.txt")
        destination_path = source_path.replace(data_root, landmarks_folder)
        print(source_path, destination_path)
        executor.submit(copy_file, source_path, destination_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",
                        type=str,
                        help="Path to root data folder.")
    parser.add_argument("--destination_folder",
                        type=str,
                        help="Path to output folder.")
    parser.add_argument("--file_ids",
                        type=str,
                        help="Path to .txt file containing filenames")
    args = parser.parse_args()
    main(args)