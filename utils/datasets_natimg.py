"""ImageNet and FFHQ datasets for the natural-image experiments."""
import os
from torch.utils.data import Dataset
from torchvision import transforms
from functools import partial
from PIL import Image
import numpy as np
import glob
from torch.utils.data import Subset


def center_crop_arr(pil_image, image_size=256):
    # Imported from openai/guided-diffusion
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

class ImageNetSubsetDataset(Dataset):
    def __init__(self, folder_path, image_size=256, transform_type='diffusion'):
        # self.image_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpeg")))
        # self.image_paths = sorted(glob.glob(os.path.join(folder_path, "*.JPEG")))
        self.image_paths = sorted(
            glob.glob(os.path.join(folder_path, "*.JPEG")) +
            glob.glob(os.path.join(folder_path, "*.jpeg")) +
            glob.glob(os.path.join(folder_path, "*.png"))  # scripts/prepare_natimg.py: already cropped to 256x256
        )
        # print(self.image_paths)
        self.image_size = image_size

        if transform_type == 'diffusion':
            self.transform = transforms.Compose([partial(center_crop_arr, image_size=image_size), transforms.ToTensor()])

        else:
            raise ValueError(f"Transform {transform_type} not implemented.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = Image.open(image_path).convert("RGB")
        img = self.transform(img)
        return img #, 165, {"class_id": wnid, "name": class_name, "index": idx} # 165 is a placeholder, it should correspond to the class index

class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        root: path to the folder containing images (all are considered part of the same split)
        transform: torchvision transform or callable to apply to each image
        """
        self.root = root
        # print(os.listdir(root))
        # print(len(os.listdir(root)))
        self.files = [f for f in os.listdir(root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.files.sort()  # so that index ordering is deterministic
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = os.path.join(self.root, self.files[index])
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        target = 0  # dummy, to keep same interface as LMDBDataset
        return img #, target, {'index': index}


def get_ffhq_dataset_from_folder(root, transform='default', image_size=256, subset=-1, **kwargs):
    # if transform == 'default':
    #     transform = transforms.ToTensor()
    if transform == 'default':
        transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    dset = ImageFolderDataset(root, transform=transform)
    print(len(dset))

    if isinstance(subset, int) and subset > 0:
        if subset > len(dset):
            print(
                f"Requested subset={subset}, but dataset only has {len(dset)} samples. "
                f"Clamping to {len(dset)}."
            )
        n = min(subset, len(dset))
        dset = Subset(dset, list(range(n)))
    elif isinstance(subset, list):
        # filter out invalid indices
        filtered = [i for i in subset if i < len(dset)]
        if len(filtered) < len(subset):
            print(
                f"Some indices in subset were out of range (dataset size={len(dset)}). "
                f"Using {len(filtered)} valid indices."
            )
        dset = Subset(dset, filtered)
    return dset

