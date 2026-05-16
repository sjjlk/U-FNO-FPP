import h5py
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F


class CropDataset(Dataset):
    def __init__(self, data_path, target_size):
        self.h5_path = data_path
        self.target_size = target_size
        self.h5_file = None

        with h5py.File(self.h5_path, 'r') as f:
            self.dataset_len = len(f['img'])
            if not (len(f['img']) == len(f['gt'])):
                raise ValueError("The length of inputs and labels in the HDF5 file must be consistent.")

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        img = self.h5_file['img'][idx]
        gt = self.h5_file['gt'][idx]

        img_tensor = torch.from_numpy(img).float().clone()
        gt_tensor = torch.from_numpy(gt).float().clone()

        _, h, w = img_tensor.shape
        th, tw = self.target_size

        pad_h = max(0, th - h)
        pad_w = max(0, tw - w)

        if pad_h > 0 or pad_w > 0:
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            padding = (pad_left, pad_right, pad_top, pad_bottom)

            img_tensor = F.pad(img_tensor, padding, value=0)
            gt_tensor = F.pad(gt_tensor, padding, value=0)

        i, j, h, w = T.RandomCrop.get_params(
            img_tensor,
            output_size=self.target_size
        )

        img_tensor = F.crop(img_tensor, i, j, h, w)
        gt_tensor = F.crop(gt_tensor, i, j, h, w)

        return img_tensor, gt_tensor
