from PIL import Image
import os
from tps.tps_io import TPSFile, TPSImage, TPSPoints
import argparse
import multiprocessing
from pathlib import Path
from functools import partial
from random import shuffle


def create_new_splits(root_path, split_sizes):
    src_tps_file = TPSFile.read_file(root_path / f"train_landmark_configuration.TPS")
    split_sizes = sorted(split_sizes, reverse=True)
    tps_data = src_tps_file.images
    shuffle(tps_data)
    per_split_lists = {}
    for split_size in split_sizes:
        tps_data = tps_data[:split_size]
        per_split_lists[split_size] = [tps_example.image for tps_example in tps_data]
        dest_tps_file = TPSFile(images=tps_data)
        dest_tps_file.write_to_file(root_path / f"train_{split_size}_landmark_configuration.TPS")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resize dataset')
    parser.add_argument('root_src', type=str, help='source dataset root directory')
    parser.add_argument('-split_sizes', type=int, nargs='+', help='sizes of splits', default=[100, 250, 500, 750])
    args = parser.parse_args()
    create_new_splits(Path(args.root_src), args.split_sizes)
