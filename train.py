import argparse

from simclr.utils.misc import load_config
from simclr.utils.trainer import Trainer

######################################
import debugpy
debugpy.listen(('localhost', 6001))
print('Waiting for debugger attach...')
debugpy.wait_for_client()
######################################

def main(args):
    cfg = load_config(args.cfg)
    trainer = Trainer(cfg)
    # trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str, default="simclr/config/simclr_config.yml", help="cfg path")
    args = parser.parse_args()
    main(args)