import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helper_functions import start_tensorboard
import torch
import pytorch_lightning as pl
from pathlib import Path
import tensorboard as _tb  # ensure package present
from torch.utils.tensorboard import SummaryWriter
from configs.default_config import Config
from datetime import datetime
from utils.helper_functions import make_trainer_config, make_tb_writer
import json

from models.zonkey import PlZonkey
from data.wiki_chars import create_dataloader


def run_training(args):
    time_now = datetime.now()

    # Initialize TensorBoard writer
    tb_writer = make_tb_writer(time_now)

    # Model init or resume from checkpoint if provided
    ckpt_path = None
    if args.resume:
        ckpt_path = Path(args.resume).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        model = PlZonkey.load_from_checkpoint(str(ckpt_path), writer=tb_writer)
    else:
        model = PlZonkey(writer=tb_writer)

    # Data & Trainer
    dataloader = create_dataloader(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS)
    trainer = pl.Trainer(**make_trainer_config(time_now))
    
    # Pass ckpt_path to trainer.fit() to restore trainer state (global_step, epoch, etc.)
    if ckpt_path:
        trainer.fit(model, dataloader, ckpt_path=str(ckpt_path))
    else:
        trainer.fit(model, dataloader)

    # Close the writer to flush remaining events
    if model.tb_writer is not None:
        model.tb_writer.close()
