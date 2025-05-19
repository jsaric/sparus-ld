from PIL import Image
import os
from tps.tps_io import TPSFile, TPSImage, TPSPoints
import argparse
import multiprocessing
from pathlib import Path
from functools import partial


def rescale_image_and_update_tps(tps_example, src_images_path, dest_images_path, larger_side_size):
    print("Processing image: ", tps_example.image)
    image = Image.open(src_images_path / tps_example.image)
    width, height = image.size
    scale = larger_side_size / max(width, height)
    image_resized = image.resize((int(width * scale), int(height * scale)))
    image_resized.save(dest_images_path / tps_example.image)

    landmarks_resized = TPSPoints(points=(tps_example.landmarks.points * scale).astype(int))
    if tps_example.scale is not None:
        scale_resized = tps_example.scale / scale
    else:
        scale_resized = None
        print("Warning: scale is None for image: ", tps_example.image)

    tps_example_resized = TPSImage(
        image=tps_example.image,
        landmarks=landmarks_resized,
        scale=scale_resized,
        id_number=tps_example.id_number,
        comment=tps_example.comment,
        curves=tps_example.curves
    )
    return tps_example_resized


def resize_dataset(root_src, root_dest, larger_side_size):
    for split in ["train", "val", "test"]:
        print("Processing split: ", split)
        src_root_path = Path(root_src)
        src_images_path = src_root_path / split
        dest_root_path = Path(root_dest)
        dest_images_path = dest_root_path / split
        os.makedirs(dest_images_path, exist_ok=True)

        src_tps_file = TPSFile.read_file(src_root_path / f"{split}_landmark_configuration.TPS")
        with multiprocessing.Pool(8) as pool:
            dest_tps_data = pool.map(
                partial(
                    rescale_image_and_update_tps,
                    src_images_path=src_images_path,
                    dest_images_path=dest_images_path,
                    larger_side_size=larger_side_size
                ),
                src_tps_file.images
            )
        print("Writing TPS file", dest_root_path / f"{split}_landmark_configuration.TPS")
        dest_tps_data = [tps_example for tps_example in dest_tps_data]
        dest_tps_file = TPSFile(images=dest_tps_data)
        dest_tps_file.write_to_file(dest_root_path / f"{split}_landmark_configuration.TPS")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resize dataset')
    parser.add_argument('root_src', type=str, help='source dataset root directory')
    parser.add_argument('root_dest', type=str, help='destination dataset root directory')
    parser.add_argument('larger_side_size', type=int, help='larger side size')
    args = parser.parse_args()
    resize_dataset(args.root_src, args.root_dest, args.larger_side_size)
