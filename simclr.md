## Tree for 
```
├── simclr/
│   ├── test/
│   │   ├── test_model.py
│   │   └── test_loss.py
│   ├── loss/
│   │   └── simclr_loss.py
│   ├── config/
│   │   └── simclr_config.yml
│   ├── dataset/
│   │   └── cifrar10_dataset.py
│   ├── utils/
│   │   ├── misc.py
│   │   └── trainer.py
│   └── model/
│       └── simclr_model.py
├── artifacts/
├── requirements.txt
├── .pre-commit-config.yaml
├── pyproject.toml
├── .gitignore
├── .github/
│   └── workflows/
│       └── python-package.yml
└── train.py
```

## File: requirements.txt
```
torch
pytest
matplotlib
numpy
PyYAML
easydict
tensorboard
torchvision
```
## File: .pre-commit-config.yaml
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.0.1
    hooks:
      - id: check-json
      - id: end-of-file-fixer
        types: [file, python]
      - id: trailing-whitespace
        types: [file, python]
      - id: mixed-line-ending
      - id: check-added-large-files
        args: [--maxkb=4096]
      - id: check-case-conflict
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        name: isort (python)
  - repo: https://github.com/hadialqattan/pycln
    rev: v2.5.0
    hooks:
      - id: pycln
        args: [--config=pyproject.toml]
```
## File: pyproject.toml
```
[tool.black]
--line-length = 120

[tool.isort]
profile = "black"
include_trailing_comma = true
line_length = 120
multi_line_output = 3

[tool.flake8]
max-line-length = 120
ignore = "E203,E501,W503,W504"

[tool.pycln]
all = true
```
## File: .gitignore
```
# --- project related stuff ---
data/
artifacts/logs
artifacts/media
*tmp.png

# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
#   For a library or package, you might want to ignore these files since the code is
#   intended to run in multiple environments; otherwise, check them in:
# .python-version

# pipenv
#   According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
#   However, in case of collaboration, if having platform-specific dependencies or dependencies
#   having no cross-platform support, pipenv may install dependencies that don't work, or not
#   install all needed dependencies.
#Pipfile.lock

# poetry
#   Similar to Pipfile.lock, it is generally recommended to include poetry.lock in version control.
#   This is especially recommended for binary packages to ensure reproducibility, and is more
#   commonly ignored for libraries.
#   https://python-poetry.org/docs/basic-usage/#commit-your-poetrylock-file-to-version-control
#poetry.lock

# pdm
#   Similar to Pipfile.lock, it is generally recommended to include pdm.lock in version control.
#pdm.lock
#   pdm stores project-wide configurations in .pdm.toml, but it is recommended to not include it
#   in version control.
#   https://pdm.fming.dev/#use-with-ide
.pdm.toml

# PEP 582; used by e.g. github.com/David-OConnor/pyflow and github.com/pdm-project/pdm
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
#  JetBrains specific template is maintained in a separate JetBrains.gitignore that can
#  be found at https://github.com/github/gitignore/blob/main/Global/JetBrains.gitignore
#  and can be added to the global gitignore or merged into this file.  For a more nuclear
#  option (not recommended) you can uncomment the following to ignore the entire idea folder.
#.idea/

.vscode
.DS_Store
simclr/__init__.py
simclr/config/__init__.py
simclr/dataset/__init__.py
simclr/loss/__init__.py
simclr/model/__init__.py
simclr/test/__init__.py
simclr/utils/__init__.py
```
## File: train.py
```python
import argparse

from simclr.utils.misc import load_config
from simclr.utils.trainer import Trainer

def main(args):
    cfg = load_config(args.cfg)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str, default="simclr/config/simclr_config.yml", help="cfg path")
    args = parser.parse_args()
    main(args)
```
## File: .github/workflows/python-package.yml
```yaml
name: Run Python Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:

    - uses: actions/checkout@v4

    - name: Set up Python 3.11
      uses: actions/setup-python@v5
      with:
        python-version: "3.11"

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
          
    - name: Run tests
      run: |
        python -m pytest simclr/test/
```
## File: simclr/test/test_model.py
```python
import pytest
import torch

from simclr.model.simclr_model import SimCLR
from simclr.utils.misc import load_config


@pytest.fixture
def cfg():
    return load_config("simclr/config/simclr_config.yml")


@pytest.mark.parametrize("bs, n_ch, h, w", [(1, 3, 224, 224)])
def test_model(cfg, bs, n_ch, h, w):
    x = torch.randn([bs, n_ch, h, w])
    model = SimCLR(cfg)
    out = model(x)

    assert out.shape == (bs, cfg.MODEL.HEAD.dim2)
```
## File: simclr/loss/simclr_loss.py
```python
import torch
import torch.nn as nn


class SimCLRLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.temperature = self.cfg.LOSS.temperature

    def forward(self, pred_aug1, pred_aug2):
        assert pred_aug1.shape == pred_aug2.shape
        B = pred_aug1.shape[0]

        pred_aug1 = nn.functional.normalize(pred_aug1, p=2, dim=1)
        pred_aug2 = nn.functional.normalize(pred_aug2, p=2, dim=1)

        preds = torch.cat([pred_aug1, pred_aug2], dim=0)
        sim_mat = (preds @ preds.T) / self.temperature

        mask = torch.eye(2 * B, device=sim_mat.device, dtype=torch.bool)
        sim_mat = sim_mat.masked_fill(mask, float("-inf"))
        
        pos_idx = torch.arange(2 * B, device=sim_mat.device)
        pos_idx = (pos_idx + B) % (2 * B)
        positives = sim_mat[torch.arange(2 * B), pos_idx]
        
        denom = torch.logsumexp(sim_mat, dim=1)
        loss = (-positives + denom).mean()

        return loss
```
## File: simclr/config/simclr_config.yml
```yaml
DATA:
  root_dir: "data/cifar-10-batches-py"
  img_size: 32

MODEL:
  HEAD:
    dim1: 512
    dim2: 128

TRAINING:
  batch_size: 1024
  n_epochs: 100
  lr: 1e-3

LOSS:
  temperature: 0.5

LOGGING:
  logging_dir: "artifacts/logs"
  media_dir: "media"
```
## File: simclr/dataset/cifrar10_dataset.py
```python
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

        self.original_transform = transforms.Compose([transforms.ToTensor()])

        self.resize_transform = transforms.Compose(
            [transforms.Resize((self.cfg.DATA.img_size, self.cfg.DATA.img_size)), transforms.ToTensor()]
        )

        self.simclr_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.cfg.DATA.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=3),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]
        class_name = IDX_TO_CLASS[label]
        img = Image.fromarray(img)

        aug1 = self.simclr_transform(img)
        aug2 = self.simclr_transform(img)

        return aug1, aug2, label, class_name
```
## File: simclr/utils/misc.py
```python
import logging
import os
import time

import torch
import yaml
from easydict import EasyDict as edict
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return edict(config)


def setup_logging(config):
    base_dir = config.LOGGING.logging_dir
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_dir = os.path.join(base_dir, f"exp_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    log_file = os.path.join(experiment_dir, "log.log")
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode="a")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    tb_dir = os.path.join(experiment_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    logger.info(f"Experiment logs saved to: {experiment_dir}")

    return logger, writer, experiment_dir


def get_device(logger):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


def vis_image(img: torch.Tensor, squeeze=False):
    if squeeze:
        img = img.squeeze(0)

    img = transforms.ToPILImage()(img)
    img.show()


def log_epoch(logger, writer, epoch, n_epochs, train_loss, val_loss, experiment_dir, model):
    logger.info(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    writer.add_scalars("per_epoch_loss", {"train": train_loss, "validation": val_loss}, epoch + 1)

    progress_dir = os.path.join(experiment_dir, "progress")
    os.makedirs(progress_dir, exist_ok=True)

    save_path = os.path.join(experiment_dir, f"model_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), save_path)


IDX_TO_CLASS = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}
```
## File: simclr/utils/trainer.py
```python
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from simclr.dataset.cifrar10_dataset import CIFRAR10Dataset
from simclr.loss.simclr_loss import SimCLRLoss
from simclr.model.simclr_model import SimCLR
from simclr.utils.misc import get_device, log_epoch, setup_logging


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger, self.writer, self.experiment_dir = setup_logging(self.cfg)
        self.device = get_device(self.logger)
        self.n_epochs = self.cfg.TRAINING.n_epochs
        self.batch_size = int(self.cfg.TRAINING.batch_size)
        self.lr = 1e-3
        self.logger.info(f"Learning rate: {self.lr}")
        self.build_loaders()
        self.build_model()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-6)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.n_epochs, eta_min=0)
        self.loss_fn = SimCLRLoss(self.cfg)
        self.training_steps = 0
        self.val_steps = 0

    def build_loaders(self):
        self.train_dataset = CIFRAR10Dataset(self.cfg, split="train")
        self.val_dataset = CIFRAR10Dataset(self.cfg, split="test")
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=True)
        self.logger.info("Built loaders")

    def build_model(self):
        self.model = SimCLR(self.cfg)
        self.model = self.model.to(self.device)
        self.logger.info("Built model")

    def train(self):
        for epoch in range(self.n_epochs):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate_one_epoch(epoch)

            log_epoch(
                self.logger, self.writer, epoch, self.n_epochs, train_loss, val_loss, self.experiment_dir, self.model
            )

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Training epoch {epoch+1}")

        for aug1, aug2, _, _ in pbar:
            self.optimizer.zero_grad()
            aug1, aug2 = aug1.to(self.device), aug2.to(self.device)
            pred_aug1, pred_aug2 = self.model(aug1), self.model(aug2)

            train_step_loss = self.loss_fn(pred_aug1, pred_aug2)
            train_step_loss.backward()
            self.optimizer.step()

            total_loss += train_step_loss.item()
            pbar.set_postfix(loss=f"{train_step_loss.item():.4f}")
            self.training_steps += 1
            self.writer.add_scalar("training_loss", train_step_loss, self.training_steps)

        self.scheduler.step()

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate_one_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0
        pbar = tqdm(self.val_loader, desc=f"Validating epoch {epoch+1}")

        with torch.no_grad():
            for aug1, aug2, _, _ in pbar:
                aug1, aug2 = aug1.to(self.device), aug2.to(self.device)
                pred_aug1, pred_aug2 = self.model(aug1), self.model(aug2)
                val_step_loss = self.loss_fn(pred_aug1, pred_aug2)
                total_loss += val_step_loss.item()
                pbar.set_postfix(loss=f"{val_step_loss.item():.4f}")
                self.val_steps += 1
                self.writer.add_scalar("validation_loss", val_step_loss, self.val_steps)

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
```
## File: simclr/model/simclr_model.py
```python
import torch.nn as nn
import torchvision.models as models


class ResNet18BackBone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)

        return x


class ProjHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dim1 = int(self.cfg.MODEL.HEAD.dim1)
        self.dim2 = int(self.cfg.MODEL.HEAD.dim2)
        self.model = nn.Sequential(nn.Linear(self.dim1, self.dim1), nn.ReLU(), nn.Linear(self.dim1, self.dim2))

    def forward(self, x):
        x = self.model(x)

        return x


class SimCLR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.backbone = ResNet18BackBone()
        self.head = ProjHead(self.cfg)

    def forward(self, x):
        backbone_out = self.backbone(x)
        head_out = self.head(backbone_out)

        return head_out
```
