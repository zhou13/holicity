import cv2
import numpy as np
from torch.utils.data import Dataset


class ShapeNetDataset(Dataset):
    def __init__(self, rootdir, split):
        self.rootdir = rootdir
        self.split = split

        filelist = np.genfromtxt(f"{rootdir}/filelist.txt", dtype=str)
        filter_ = np.genfromtxt(f"{rootdir}/{split}-middlesplit.txt", dtype=str)
        length = len(filter_[0])
        filter_ = set(filter_)

        self.filelist = [f"{rootdir}/{f}" for f in filelist if f[:length] in filter_]
        self.size = len(self.filelist)
        print(f"n{split}:", self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        prefix = self.filelist[idx]
        image = cv2.imread(f"{prefix}_imag.jpg", -1).astype(np.float32) / 255.0
        plane_mask = cv2.imread(f"{prefix}_plan.jpng", -1)
        with np.load(f"{prefix}_plan.npz") as N:
            plane_normal = N["ws"]

        with np.load(f"{prefix}_dpth.npz") as N:
            depth = N["depth"]
        with np.load(f"{prefix}_nrml.npz") as N:
            normal = N["normal"]

        return {
            "image": image,
            "plane_mask": plane_mask,
            "plane_normal": plane_normal,
            "depth": depth,
            "normal": normal,
        }
