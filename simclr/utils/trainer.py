import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from simclr.dataset.cifrar10_dataset import CIFRAR10Dataset
from simclr.model.simclr_model import SimCLR
from simclr.loss.simclr_loss import SimCLRLoss
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
        self.loss_fn = SimCLRLoss(self.cfg)
        self.training_steps = 1
        self.val_steps = 1

    def build_loaders(self):
        self.train_dataset = CIFRAR10Dataset(self.cfg, split="train")
        self.val_dataset = CIFRAR10Dataset(self.cfg, split="test")
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.logger.info("Built loaders")

    def build_model(self):
        self.model = SimCLR(self.cfg)
        self.model = self.model.to(self.device)
        self.logger.info("Built model")

    def train(self):
        for epoch in range(self.n_epochs):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate_one_epoch(epoch)

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Training epoch {epoch+1}")

        for _, _, aug1, aug2, _, _ in tqdm(self.train_loader, desc=f"Train epoch {epoch+1}"):
            self.optimizer.zero_grad()
            aug1, aug2 = aug1.to(self.device), aug2.to(self.device)
            pred_aug1, pred_aug2 = self.model(aug1), self.model(aug2)
            loss = self.loss_fn(pred_aug1, pred_aug2)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            self.training_steps += 1
            self.writer.add_scalar("training_loss", loss, self.training_steps)

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate_one_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0
        pbar = tqdm(self.val_loader, desc=f"Validating epoch {epoch+1}")

        with torch.no_grad():
            for _, _, aug1, aug2, _, _ in tqdm(self.val_loader, desc=f"Train epoch {epoch+1}"):
                aug1, aug2 = aug1.to(self.device), aug2.to(self.device)
                pred_aug1, pred_aug2 = self.model(aug1), self.model(aug2)
                loss = self.loss_fn(pred_aug1, pred_aug2)
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                self.val_steps += 1
                self.writer.add_scalar("validation_loss", loss, self.val_steps)

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss