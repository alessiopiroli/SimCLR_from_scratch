from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image


class MNISTDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.split = split
        self.base_path = self.cfg.DATA.root_dir
        self.img_size = self.cfg.DATA.img_size
        self.img_resize = self.cfg.DATA.img_resize
        self.blur_ker = int(self.cfg.DATA.blur_ker)

        self.dataset = datasets.MNIST(
            root=self.base_path,
            train=("train" == self.split),
            download=True
        )

        self.original_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        self.simclr_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomResizedCrop(self.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        img = img.convert("RGB")

        class_name = str(label)

        original = self.original_transform(img)

        aug1 = self.simclr_transform(img)
        aug2 = self.simclr_transform(img)

        return original, aug1, aug2, label, class_name