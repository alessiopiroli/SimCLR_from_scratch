import os
import pickle

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from simclr.utils.misc import IDX_TO_CLASS


class CIFRAR10Dataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.split = split
        self.base_path = self.cfg.DATA.root_dir

        if "cifar-10-batchs-py" not in self.base_path and os.path.exists(
            os.path.join(self.base_path, "cifar-10-batches-py")
        ):
            self.data_dir = os.path.join(self.base_path, "cifar-10-batches-py")
        else:
            self.data_dir = self.base_path

        self.data = []
        self.labels = []

        file_names = [f"data_batch_{i}" for i in range(1, 6)] if self.split == "train" else ["test_batch"]

        for file_name in file_names:
            file_path = os.path.join(self.data_dir, file_name)

            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                self.labels.extend(entry["labels"] if "labels" in entry else entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))

        self.original_transform = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )

        self.resize_transform = transforms.Compose(
            [
                transforms.Resize((self.cfg.DATA.img_size, self.cfg.DATA.img_size)),
                transforms.ToTensor()
            ]
        )

        self.simclr_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.cfg.DATA.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                transforms.GaussianBlur(kernel_size=23),
                transforms.ToTensor()
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]
        class_name = IDX_TO_CLASS[label]
        img = Image.fromarray(img)

        original = self.original_transform(img)
        resized = self.resize_transform(img)
        aug1 = self.simclr_transform(img)
        aug2 = self.simclr_transform(img)

        return original, resized, aug1, aug2, label, class_name