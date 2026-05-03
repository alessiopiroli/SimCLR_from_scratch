import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from simclr.dataset.cifrar10_dataset import CIFRAR10Dataset
from simclr.model.simclr_model import SimCLR
from simclr.utils.misc import get_device, setup_logging


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger, self.writer, self.experiment_dir = setup_logging(self.cfg)
        self.device = get_device(self.logger)
        self.n_epochs = self.cfg.TRAINING.n_epochs
        self.batch_size = int(self.cfg.TRAINING.batch_size)
        self.lr = float(self.cfg.TRAINING.lr)
        self.build_loaders()
        self.build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def build_loaders(self):
        self.train_dataset = CIFRAR10Dataset(self.cfg, split="train")
        self.val_dataset = CIFRAR10Dataset(self.cfg, split="test")
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.logger.info("Built loaders")

    def build_model(self):
        self.model = SimCLR()
        self.model = self.model.to(self.device)
        self.logger.info("Built model")

    def train(self):
        for epoch in range(self.n_epochs):
            train_loss = self.train_one_epoch(epoch)


    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Training epoch {epoch+1}")

        for orig_img, train_img, label, class_name in tqdm(self.train_loader, desc=f"Train epoch {epoch+1}"):
            self.optimizer.zero_grad()