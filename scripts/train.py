import os
import sys
import argparse
import json
from pathlib import Path
import torch
torch.set_float32_matmul_precision("high")

# Add the project root to the Python path FIRST
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# Import Config early, before any model imports
from configs.default_config import Config


def apply_param_overrides(param_file: str | None):
    """Override Config class attributes from a JSON file if provided."""
    if not param_file:
        return
    path = Path(param_file).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Param file not found: {path}")
    with open(path, "r") as f:
        params = json.load(f)

    # Collect configurable attributes from Config (public, uppercase preferred)
    config_attrs = {name for name in dir(Config) if name.isupper()}

    # Apply overrides where keys match Config attributes (case-insensitive support)
    for key, value in params.items():
        exact_key = key
        upper_key = key.upper()
        target_key = None
        if exact_key in config_attrs:
            target_key = exact_key
        elif upper_key in config_attrs:
            target_key = upper_key

        if target_key is not None:
            setattr(Config, target_key, value)
            print(f"Overrode Config.{target_key} = {value}")
        else:
            print(f"Warning: key '{key}' not found in Config; skipping override")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train diffhuge LLM")
    parser.add_argument('--param_file', type=str, default=None,
                        help='Parameter file in param_versions directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint file')
    parser.add_argument('--load_weights_only', action='store_true',
                        help='Load model weights with strict=False (keeps new params random)')
    parser.add_argument('--calibrate', action='store_true',
                        help='Run memory calibration (worst-case batch) before training')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Apply parameter overrides BEFORE importing run_trainer (which imports models)
    apply_param_overrides(args.param_file)
    print(args)
    
    # Import run_trainer AFTER config overrides are applied
    from run_trainer import run_training
    
    # Start training
    print("Starting training...")
    run_training(args)