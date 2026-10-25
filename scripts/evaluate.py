"""Evaluate a training checkpoint with its EMA weights: generation word rate of every level
(100 and 400 sampler steps) and held-out reconstruction (utils/evaluation.py has the definitions).
    python scripts/evaluate.py checkpoints/<run>/<ckpt> --param_file configs/<its config>.json
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--param_file", required=True, help="config json the checkpoint was trained with")
    ap.add_argument("--n", type=int, default=128, help="samples per level and step count")
    ap.add_argument("--no_ema", action="store_true", help="evaluate the online weights")
    ap.add_argument("--out", default=None, help="also write the results to this json file")
    args = ap.parse_args()
    from configs.default_config import Config
    for k, v in json.load(open(args.param_file)).items():
        if k.upper() in dir(Config) and k.upper() not in ("DEVICE", "TB_WRITER"):
            setattr(Config, k.upper(), v)
    Config.DEVICE = "cuda"
    import torch
    torch.set_float32_matmul_precision("high")
    from models.zonkey import PlZonkey
    from utils.evaluation import evaluate
    pl = PlZonkey(writer=None).cuda()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = pl.load_state_dict(ck["state_dict"], strict=False)
    if missing or unexpected:
        print(f"warning: {len(missing)} missing / {len(unexpected)} unexpected tensors "
              f"(is {args.param_file} the config this checkpoint was trained with?)")
    if args.no_ema or ck.get("ema_state") is None:
        pl._ema = None      # the weights themselves (an ema_to_weights.py checkpoint holds the EMA as weights)
    elif not pl.load_ema_state(ck["ema_state"]):
        raise SystemExit("the checkpoint's EMA does not match the model built from " + args.param_file)
    del ck
    res = evaluate(pl, n=args.n)
    print(json.dumps(res, indent=1))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
