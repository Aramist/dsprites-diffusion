import argparse
from pathlib import Path

from .trainer import Trainer

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path)
ap.add_argument("--save-path", type=Path, default=Path("model_weights"))
args = ap.parse_args()

trainer = Trainer(dset_path=args.data, save_path=args.save_path)

trainer.train()
