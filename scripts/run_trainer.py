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
    resume_trainer_state = False
    if args.resume:
        ckpt_path = Path(args.resume).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        optimizer_states = checkpoint.get("optimizer_states", [])
        checkpoint_had_muon = any(
            group.get("use_muon", False)
            for state in optimizer_states
            for group in state.get("param_groups", [])
        )
        optimizer_compatible = (
            checkpoint.get("optimizer_layout_version", 1) == 2
            and checkpoint_had_muon == Config.USE_MUON
        )
        weights_only = getattr(args, 'load_weights_only', False) or not optimizer_compatible
        if weights_only:
            model = PlZonkey(writer=tb_writer)
            state_dict = checkpoint.get("state_dict", checkpoint)
            incompatible = model.load_state_dict(state_dict, strict=False)
            if checkpoint.get("ema_state") is not None and model.use_ema:
                model._ema = [value.clone() for value in checkpoint["ema_state"]]
            reason = "requested" if getattr(args, 'load_weights_only', False) else "optimizer layout changed"
            print(f"Loaded checkpoint weights with strict=False ({reason}); optimizer starts fresh")
            if incompatible.missing_keys:
                print(f"Missing keys (new params kept random): {len(incompatible.missing_keys)}")
            if incompatible.unexpected_keys:
                print(f"Unexpected keys (ignored): {len(incompatible.unexpected_keys)}")
            ckpt_path = None
        else:
            model = PlZonkey.load_from_checkpoint(str(ckpt_path), writer=tb_writer)
            resume_trainer_state = True
        del checkpoint
    else:
        model = PlZonkey(writer=tb_writer)

    # Optional: run memory calibration before training
    if getattr(args, 'calibrate', False):
        fits = model.calibrate_memory()
        if not fits:
            print("Aborting: worst-case batch does not fit in GPU memory.")
            return

    # Data & Trainer
    dataloader = create_dataloader(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS)
    trainer = pl.Trainer(**make_trainer_config(time_now))
    
    # Pass ckpt_path to trainer.fit() to restore trainer state (global_step, epoch, etc.)
    if ckpt_path and resume_trainer_state and Config.USE_OPTIMIZER_CHECKPOINT:
        trainer.fit(model, dataloader, ckpt_path=str(ckpt_path))
    else:
        trainer.fit(model, dataloader)

    # Close the writer to flush remaining events
    if model.tb_writer is not None:
        model.tb_writer.close()
