"""Autoregressive baseline: a char-level decoder-only transformer (252M parameters by default) on
exactly Zonkey's data (the first 1024 chars of each training article via data/wiki_chars.py, END=1
after the text, PAD=0) with Zonkey's batch (3 documents) and optimizer (Muon 0.005 + aux Adam 2e-4,
momentum 0.95, warmup 2000, grad clip 1.0). At every eval point: the word rate of 64 unconditional
samples (from the start-of-document token, per temperature) and held-out bits per char, against
characters seen.
    python scripts/ar_baseline.py --steps 20000 --out ar_baseline.json
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--muon_lr", type=float, default=0.005)
    ap.add_argument("--adam_lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--evals", default="250,500,1000,1500,2000,3000,4000,5000,7500,10000,15000,20000,30000,40000")
    ap.add_argument("--temps", default="1.0,0.8")
    ap.add_argument("--out", default="ar_baseline.json", help="results json; the weights go to <out>.pt")
    args = ap.parse_args()
    temps = [float(t) for t in args.temps.split(",")]
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from configs.default_config import Config
    Config.MAX_DOC_LENGTHS = [1024, 128, 20]
    from data.wiki_chars import create_dataloader, text_to_ids
    from muon import SingleDeviceMuonWithAuxAdam
    from utils.evaluation import load_vocab, word_stats
    torch.manual_seed(0)
    dev = "cuda"
    T = 1024

    def rope_tables(hd):
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=dev).float() / hd))
        f = torch.outer(torch.arange(T, device=dev).float(), inv)
        return f.cos()[None, None], f.sin()[None, None]

    def rope(x, cos, sin):
        x0, x1 = x[..., 0::2], x[..., 1::2]
        L = x.shape[2]
        c, s = cos[:, :, :L], sin[:, :, :L]
        return torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], -1).flatten(-2)

    class Block(nn.Module):
        def __init__(s, d, h):
            super().__init__()
            s.h = h
            s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            s.qkv, s.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
            s.ff1, s.ff2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

        def forward(s, x, cos, sin):
            B, L, D = x.shape
            q, k, v = s.qkv(s.ln1(x)).view(B, L, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(rope(q, cos, sin), rope(k, cos, sin), v, is_causal=True)
            x = x + s.o(a.transpose(1, 2).reshape(B, L, D))
            return x + s.ff2(F.gelu(s.ff1(s.ln2(x))))

    class AR(nn.Module):
        def __init__(s, d, n, h):
            super().__init__()
            s.emb = nn.Embedding(256, d)
            s.blocks = nn.ModuleList([Block(d, h) for _ in range(n)])
            s.lnf = nn.LayerNorm(d)
            s.head = nn.Linear(d, 256, bias=False)
            s.cos, s.sin = rope_tables(d // h)

        def forward(s, idx):
            x = s.emb(idx)
            for b in s.blocks:
                x = b(x, s.cos, s.sin)
            return s.head(s.lnf(x))

    model = AR(args.d, args.layers, args.heads).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"AR params {n_params / 1e6:.1f}M", flush=True)
    muon_p = [p for n, p in model.named_parameters() if p.ndim == 2 and "emb" not in n and "head" not in n]
    other = [p for n, p in model.named_parameters() if not (p.ndim == 2 and "emb" not in n and "head" not in n)]
    opt = SingleDeviceMuonWithAuxAdam([
        dict(params=muon_p, lr=args.muon_lr, momentum=0.95, weight_decay=0.01, use_muon=True),
        dict(params=other, lr=args.adam_lr, eps=1e-7, use_muon=False)])
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    fwd = torch.compile(model)

    vocab = load_vocab()
    from datasets import load_dataset
    held = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[95%:]")
    hb = []
    for i in range(0, 96 * 50, 50):
        t = text_to_ids(held[i]["text"])[:1023]
        hb.append([1] + t + [1] + [0] * (1023 - len(t)))
    held_ids = torch.tensor(hb, device=dev)[:, :T + 1]
    real_starts = [held[i]["text"][:200] for i in range(0, 64 * 37, 37)]
    real_all, real_long, _ = word_stats(real_starts, vocab)
    print(f"real doc starts: word rate {real_all:.3f} / >=4 letters {real_long:.3f}", flush=True)

    @torch.no_grad()
    def evaluate():
        model.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            nll, cnt = 0.0, 0
            for j in range(0, held_ids.shape[0], 16):
                x = held_ids[j:j + 16]
                logits = fwd(x[:, :-1]).float()
                tgt = x[:, 1:]
                m = tgt > 0
                nll += F.cross_entropy(logits[m], tgt[m], reduction="sum").item()
                cnt += int(m.sum())
            per_temp = {}
            for temp in temps:
                g = torch.Generator(device=dev).manual_seed(7)
                seq = torch.ones(64, 1, dtype=torch.long, device=dev)
                done = torch.zeros(64, dtype=torch.bool, device=dev)
                for _ in range(200):
                    p = torch.softmax(model(seq)[:, -1].float() / temp, -1)
                    p[:, 0] = 0
                    nxt = torch.multinomial(p, 1, generator=g)
                    nxt[done] = 0
                    done |= nxt[:, 0] == 1
                    seq = torch.cat([seq, nxt], 1)
                texts = ["".join(chr(c) for c in row[1:].tolist() if c > 2) for row in seq]
                per_temp[temp] = (word_stats(texts, vocab), texts[:3])
        model.train()
        (a, l, n), ex = per_temp[temps[0]]
        return nll / cnt / math.log(2), a, l, n, ex, {t: v[0][:2] for t, v in per_temp.items()}

    evals = set(int(x) for x in args.evals.split(","))
    dl = create_dataloader(batch_size=args.batch, num_workers=4)
    curve = []
    step, chars = 0, 0
    t0 = time.time()
    while step < args.steps:
        for batch in dl:
            ids = batch["full_texts"].to(dev, non_blocking=True)
            x = torch.cat([torch.ones_like(ids[:, :1]), ids[:, :-1]], 1)
            chars += int((ids > 1).sum())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = fwd(x)
            loss = F.cross_entropy(logits.float().reshape(-1, 256), ids.reshape(-1), ignore_index=0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            warm = min(1.0, (step + 1) / args.warmup)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * warm
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step in evals or step == args.steps:
                bpc, a, l, n, ex, by_temp = evaluate()
                curve.append(dict(step=step, chars=chars, bpc=bpc, word_rate=a, word_rate_long=l, words=n,
                                  by_temp={str(t): v for t, v in by_temp.items()},
                                  train_loss=float(loss), sec=time.time() - t0))
                torch.save(model.state_dict(), args.out + ".pt")
                print(json.dumps(curve[-1]), flush=True)
                for e in ex:
                    print("   ", repr(e[:120]), flush=True)
                json.dump(dict(args=vars(args), params=n_params, real=[real_all, real_long], curve=curve),
                          open(args.out, "w"), indent=1)
            if step >= args.steps:
                break


if __name__ == "__main__":
    main()
