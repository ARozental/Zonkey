"""Turn a training checkpoint into an init checkpoint whose weights are its EMA weights.

The EMA is stored as a list in the order of the trainable parameters of the model that was trained,
so the names come from rebuilding that model (on the meta device) from the config it was trained with.
The result has no optimizer state: start from it with a weights-only resume, e.g.
    python scripts/ema_to_weights.py checkpoints/.../epoch=1-step=2090000.ckpt init_2090k_ema.ckpt
    python scripts/train.py --param_file configs/<new config>.json --resume init_2090k_ema.ckpt --load_weights_only
Any architecture change is fine: tensors are matched by name and shape, the rest keeps its init.
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--trained_with", default="configs/single_gpu_config.json",
                    help="config json the checkpoint was trained with (defaults + this file)")
    ap.add_argument("--keep_schedule", action="store_true", help="continue the old LR schedule step")
    args = ap.parse_args()
    from configs.default_config import Config
    for k, v in json.load(open(args.trained_with)).items():
        if k.upper() in dir(Config):
            setattr(Config, k.upper(), v)
    import torch
    from models.zonkey import Zonkey
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ema = ck.get("ema_state")
    if ema is None:
        raise SystemExit("checkpoint has no ema_state")
    with torch.device("meta"):
        ref = Zonkey()
    names = [n for n, p in ref.named_parameters() if p.requires_grad]
    if len(names) != len(ema) or any(tuple(p.shape) != tuple(e.shape) for (n, p), e in
                                     zip([(n, p) for n, p in ref.named_parameters() if p.requires_grad], ema)):
        raise SystemExit(f"EMA does not match the model built from {args.trained_with} "
                         f"({len(ema)} EMA tensors, {len(names)} parameters)")
    sd = dict(ck["state_dict"])
    for n, e in zip(names, ema):
        sd["model." + n] = e.to(sd["model." + n].dtype)
    # A new run on these weights starts a fresh LR schedule (warmup + cosine) unless --keep_schedule.
    step = ck.get("schedule_step", ck.get("global_step", 0)) if args.keep_schedule else 0
    out = {"state_dict": sd, "ema_state": None, "schedule_step": step, "global_step": 0,
           "optimizer_layout_version": 2}
    torch.save(out, args.out)
    print(f"wrote {args.out}: {len(names)} parameters replaced by their EMA")


if __name__ == "__main__":
    main()
