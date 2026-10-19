"""How many dimensions do a trained model's codes need? Inference only, EMA weights: project
held-out codes onto the top-k principal directions of a large code sample (k << code dim),
renormalize, and decode with the unchanged decoders. The decoders never saw projected codes, so
what survives is a lower bound on what k dimensions can carry.
L0: codes (CV x d) -> char agreement / exact word decodes vs the unprojected decode.
L1: codes -> decoded by L1 then L0 at t=0 -> exact text of the window vs the unprojected decode.
    python scripts/capacity_test.py checkpoints/20261018-074944/epoch=1-step=2090000.ckpt
"""
import argparse
import difflib
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--trained_with", default="configs/single_gpu_config.json",
                    help="config json the checkpoint was trained with (defaults + this file)")
    ap.add_argument("--batches", type=int, default=120, help="training batches whose codes are sampled")
    args = ap.parse_args()
    from configs.default_config import Config
    for k, v in json.load(open(args.trained_with)).items():
        if k.upper() in dir(Config) and k.upper() not in ("DEVICE", "TB_WRITER"):
            setattr(Config, k.upper(), v)
    Config.DEVICE = "cuda"
    Config.COMPILE_TRANSFORMER_STACKS = False
    import torch
    import torch.nn.functional as F
    from models.zonkey import PlZonkey
    from data.wiki_chars import create_dataloader
    from utils.evaluation import decode_down
    pl = PlZonkey(writer=None).cuda()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    pl.load_state_dict(ck["state_dict"], strict=False)
    if ck.get("ema_state") is not None:
        pl.load_ema_state(ck["ema_state"])
    else:
        pl._ema = None
    del ck
    m = pl.model
    L0 = m.layers[0]
    dl = create_dataloader(batch_size=3, num_workers=0)
    codes0, codes1 = [], []
    with torch.no_grad(), pl._ema_swapped():
        m.eval()
        torch.manual_seed(0)
        for i, b in enumerate(dl):
            b = {k: v.cuda() for k, v in b.items()}
            leveled, _ = m.forward(b)
            for layer in m.layers:
                layer._pending_clean_for_queue = None
            c0 = leveled[0].reshape(-1, leveled[0].shape[-1])
            codes0.append(c0[c0.abs().sum(-1) > 0])
            c1 = leveled[1].reshape(-1, leveled[1].shape[-1])
            codes1.append(c1[c1.abs().sum(-1) > 0])
            if i == args.batches:
                break
        C0, C1 = torch.cat(codes0), torch.cat(codes1)
        print(f"codes: L0 {tuple(C0.shape)}, L1 {tuple(C1.shape)}", flush=True)
        fit0, test0 = C0[:-512], C0[-512:]
        fit1, test1 = C1[:-96], C1[-96:]
        table = F.normalize(m.token_embedding_layer.weight, dim=-1)

        def pca_project(fit, x, k):
            mu = fit.mean(0, keepdim=True)
            u, s, vh = torch.linalg.svd(fit - mu, full_matrices=False)
            basis = vh[:k]
            y = (x - mu) @ basis.t() @ basis + mu
            return F.normalize(y, dim=-1) * x.norm(dim=-1, keepdim=True)

        def decode0(codes):
            n = codes.shape[0]
            seq, em, isr = L0.generate(fixed_compressed_vectors=codes, noise_level=torch.zeros(n, device="cuda"),
                                       treat_as_noisy=True, num_diffusion_steps=0)
            ids = (F.normalize(seq, dim=-1) @ table.t()).argmax(-1)
            return ids, em

        ref_ids, ref_em = decode0(test0)
        print(f"L0 ({C0.shape[-1]} dims): char agreement with the unprojected decode / exact word decodes", flush=True)
        for k in (512, 256, 128, 64, 32):
            k = min(k, fit0.shape[0] - 1)   # n centered codes span at most n - 1 directions
            ids, em = decode0(pca_project(fit0, test0, k))
            keep = ref_em.bool()
            char = float(((ids == ref_ids) & keep).sum() / keep.sum())
            word = float((((ids == ref_ids) | ~keep).all(-1) & (em.bool() == keep).all(-1)).float().mean())
            print(f"   k = {k:4d}: chars {char:.4f}  words {word:.4f}", flush=True)

        refs = [decode_down(m, 1, c.unsqueeze(0)) for c in test1]
        print(f"L1 ({C1.shape[-1]} dims): exact window text vs the unprojected decode / mean char similarity", flush=True)
        for k in (2048, 1024, 512, 256, 128):
            k = min(k, fit1.shape[0] - 1)
            proj = pca_project(fit1, test1, k)
            outs = [decode_down(m, 1, c.unsqueeze(0)) for c in proj]
            exact = sum(o == r for o, r in zip(outs, refs)) / len(refs)
            sim = sum(difflib.SequenceMatcher(None, o, r).ratio() for o, r in zip(outs, refs)) / len(refs)
            print(f"   k = {k:4d}: exact {exact:.3f}  similarity {sim:.3f}   e.g. {outs[0][:60]!r} vs {refs[0][:60]!r}", flush=True)


if __name__ == "__main__":
    main()
