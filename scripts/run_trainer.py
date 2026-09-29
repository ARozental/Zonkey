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
        state_dict = checkpoint.get("state_dict", checkpoint)
        model = PlZonkey(writer=tb_writer)
        # A full (optimizer-state) resume needs the exact same parameters; a checkpoint from
        # an older architecture falls back to a weights-only resume instead of crashing.
        model_keys = set(model.state_dict().keys())
        own_state = model.state_dict()
        same_params = (set(state_dict.keys()) == model_keys
                       and all(tuple(state_dict[k].shape) == tuple(own_state[k].shape) for k in model_keys))
        weights_only = getattr(args, 'load_weights_only', False) or not optimizer_compatible or not same_params
        if weights_only:
            # Only tensors whose name AND shape match are loaded; everything else keeps its init.
            loadable = {k: v for k, v in state_dict.items() if k in own_state and tuple(v.shape) == tuple(own_state[k].shape)}
            skipped = sorted(k for k in state_dict if k in own_state and k not in loadable)
            incompatible = model.load_state_dict(loadable, strict=False)
            if checkpoint.get("ema_state") is not None and model.use_ema:
                model.load_ema_state(checkpoint["ema_state"])
            # Lightning restarts global_step at 0 on a weights-only resume: continue the LR
            # schedule (and TensorBoard steps) from where the checkpoint stopped.
            if not int(getattr(Config, "LR_SCHEDULE_OFFSET", 0) or 0):
                model._schedule_offset = int(checkpoint.get("schedule_step", checkpoint.get("global_step", 0)) or 0)
            if getattr(args, 'load_weights_only', False):
                reason = "requested"
            elif not optimizer_compatible:
                reason = "optimizer layout changed"
            else:
                reason = "parameters changed"
            print(f"Loaded checkpoint weights with strict=False ({reason}); optimizer starts fresh; "
                  f"LR schedule continues at step {model._schedule_offset}")
            missing = [k for k in incompatible.missing_keys if k not in skipped]
            if missing:
                print(f"Missing keys (new params kept random): {len(missing)}")
            if skipped:
                print(f"Shape-changed keys (kept random): {len(skipped)}")
            if incompatible.unexpected_keys:
                print(f"Unexpected keys (ignored): {len(incompatible.unexpected_keys)}")
            ckpt_path = None
        else:
            del model
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
