import logging, os, time, yaml, torch
from easydict import EasyDict as edict
from torch.utils.tensorboard import SummaryWriter

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