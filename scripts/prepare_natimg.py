"""Prepare the natural-image test images from the original dataset: the images listed in data/lists/<dataset>_test.txt,
processed as the data loader does (256x256) and written as PNGs to data/<dataset>/, where configs/natimg/<dataset>.yaml
points.

    python scripts/prepare_natimg.py imagenet /path/to/ILSVRC2012/train        # train/<class>/<file>.JPEG
    python scripts/prepare_natimg.py ffhq /path/to/ffhq-dataset/images1024x1024  # images1024x1024/<thousands>/<file>.png

Run from the repository root. Each image is compared with the one used for the paper (MD5 of its pixels).
"""
import argparse
import hashlib
import os
import sys

import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
from utils.datasets_natimg import center_crop_arr

IMAGE_SIZE = 256


def read_list(dataset):
    """(file name, MD5 of the pixels used for the paper) of every test image."""
    with open(os.path.join('data', 'lists', f'{dataset}_test.txt')) as f:
        return [line.split() for line in f if line.strip() and not line.startswith('#')]


def locate(dataset, source, names):
    """Path of every file in the original dataset's layout, in source itself, or anywhere below source."""
    paths = {}
    for name in names:
        if dataset == 'imagenet':
            folder = name.split('_')[0]  # class
        else:
            folder = f'{int(os.path.splitext(name)[0]) // 1000 * 1000:05d}'
        for path in (os.path.join(source, folder, name), os.path.join(source, name)):
            if os.path.isfile(path):
                paths[name] = path
                break
    missing = set(names) - set(paths)
    if missing:  # another layout: search the whole folder
        for root, _, files in os.walk(source):
            paths.update({f: os.path.join(root, f) for f in missing.intersection(files)})
    missing = sorted(set(names) - set(paths))
    if missing:
        sys.exit(f'{len(missing)} of the {len(names)} images are not in {source}, e.g. {missing[0]}')
    return paths


def process(dataset, path):
    """The 256x256 RGB pixels the data loader (utils/datasets_natimg.py) makes of an original image."""
    image = Image.open(path).convert('RGB')
    if dataset == 'imagenet':  # ImageNetSubsetDataset
        return center_crop_arr(image, IMAGE_SIZE)
    resize = transforms.Compose([  # get_ffhq_dataset_from_folder
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(IMAGE_SIZE)])
    return np.asarray(resize(image))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', choices=['imagenet', 'ffhq'])
    parser.add_argument('source', help='folder of the original dataset (see above)')
    args = parser.parse_args()

    if not os.path.exists('main_natimg.py'):
        sys.exit('Run from the repository root.')

    entries = read_list(args.dataset)
    paths = locate(args.dataset, args.source, [name for name, _ in entries])
    out = os.path.join('data', args.dataset)
    os.makedirs(out, exist_ok=True)
    same = 0
    for name, md5 in entries:
        pixels = np.ascontiguousarray(process(args.dataset, paths[name]))
        same += hashlib.md5(pixels.tobytes()).hexdigest() == md5
        Image.fromarray(pixels).save(os.path.join(out, os.path.splitext(name)[0] + '.png'))
    print(f'{out}: {len(entries)} images, {same} identical to the ones used for the paper')
    if same < len(entries):
        print('The others differ slightly: the installed Pillow decodes or resizes them differently.')


if __name__ == '__main__':
    main()
