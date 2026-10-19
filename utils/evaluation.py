"""The metrics behind the overhaul's results (scripts/evaluate.py, ar_baseline.py, capacity_test.py).

Word rate: the share of generated words (letter runs of >= 2 letters, lowercased) that are real
words, i.e. letter runs seen at least twice in the first 1024 chars of 20k training documents; also
for words of >= 4 letters. Real Wikipedia text scores ~0.93 / 0.90.
"""
import difflib
import json
import os
import random
import re
from collections import Counter

WORD = re.compile(r"[^\W\d_]+")
VOCAB_CACHE = os.path.expanduser("~/.cache/zonkey/eval_vocab.json")


def load_vocab(path=VOCAB_CACHE):
    if os.path.exists(path):
        return set(json.load(open(path)))
    from datasets import load_dataset
    train = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:95%]")
    cnt = Counter()
    for i in random.Random(0).sample(range(len(train)), 20000):
        cnt.update(w.lower() for w in WORD.findall(train[i]["text"][:1024]))
    vocab = sorted(w for w, c in cnt.items() if c >= 2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(vocab, open(path, "w"))
    return set(vocab)


def word_stats(texts, vocab):
    """(share of words >= 2 letters in the vocab, same for words >= 4 letters, n words)."""
    words = [w.lower() for t in texts for w in WORD.findall(t) if len(w) >= 2]
    lw = [w for w in words if len(w) >= 4]
    return (sum(w in vocab for w in words) / max(1, len(words)),
            sum(w in vocab for w in lw) / max(1, len(lw)), len(words))


def codes_to_text(model, vectors):
    """Level-0 output vectors [n, d] -> text (nearest token embedding)."""
    import torch.nn.functional as F
    from models.zonkey import Zonkey
    table = F.normalize(model.token_embedding_layer.weight, dim=-1)
    return Zonkey.token_ids_to_text((F.normalize(vectors, dim=-1) @ table.t()).argmax(-1))


def decode_down(model, level, codes):
    """Codes [n, CV * d] of `level` (one document) -> text, every level decoding at t=0."""
    import torch
    doc = codes
    for lv in range(level, -1, -1):
        if doc.shape[0] == 0:
            return ""
        seq, emask, isr = model.layers[lv].generate(fixed_compressed_vectors=doc,
                                                    noise_level=torch.zeros(doc.shape[0], device=doc.device),
                                                    treat_as_noisy=True, num_diffusion_steps=0)
        doc = model.assemble(lv, seq, emask, isr).squeeze(0)
    return codes_to_text(model, doc)


def evaluate(pl, n=128, seed=1234, held_out_docs=48):
    """With the EMA weights (if loaded):
    gen_L{l}[_s400]: n unconditional samples of every level >= 1 with 100 (and 400) sampler steps
        at that level, decoded down to text: word rates and mean length in chars;
    gen_L0: 256 single-word samples: word rates;
    recon_L{l}: the first level-l code of each of `held_out_docs` held-out documents (the last 5%
        of the split, which training never sees) decoded down at t=0, against the text: char
        similarity and exact-match rate."""
    import torch
    from configs.default_config import Config
    from data.wiki_chars import text_to_ids
    from utils.content_id import prefix_hash
    vocab = load_vocab()
    m = pl.model
    res = {}
    dev = "cuda"
    with torch.no_grad(), pl._ema_swapped():
        m.eval()
        top = len(m.layers) - 1
        saved_steps = list(getattr(Config, "SAMPLE_STEPS", [30] * len(m.layers)))
        for lv in range(1, top + 1):
            for steps in (100, 400):
                sched = list(saved_steps) + [100] * (len(m.layers) - len(saved_steps))
                sched[lv] = steps
                Config.SAMPLE_STEPS = sched
                torch.manual_seed(seed)
                seq, em, isr = m.layers[lv].generate(batch_size=n, noise_level=1.0)
                texts = []
                for i in range(n):
                    kids = m.assemble(lv, seq[i:i + 1], em[i:i + 1], isr[i:i + 1]).squeeze(0)
                    texts.append(decode_down(m, lv - 1, kids))
                a, l, _ = word_stats(texts, vocab)
                tag = f"gen_L{lv}" + ("" if steps == 100 else f"_s{steps}")
                res[f"{tag}_word_rate"], res[f"{tag}_word_rate_long"] = a, l
                res[f"{tag}_chars"] = sum(len(t) for t in texts) / n
                res[f"{tag}_examples"] = texts[:8]
        Config.SAMPLE_STEPS = saved_steps
        torch.manual_seed(seed + 1)
        s0, e0, _ = m.layers[0].generate(batch_size=256, noise_level=1.0)
        w0 = [codes_to_text(m, s0[i, :int(e0[i].sum())]) for i in range(256)]
        res["gen_L0_word_rate"], res["gen_L0_word_rate_long"], _ = word_stats(w0, vocab)
        res["gen_L0_examples"] = w0[:12]

        from datasets import load_dataset
        held = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[95%:]")
        idx = random.Random(1).sample(range(len(held)), held_out_docs)
        L = Config.MAX_DOC_LENGTHS[0]
        sims = {lv: [] for lv in range(1, top + 1)}
        exact = {lv: [] for lv in range(1, top + 1)}
        examples = []
        for b in range(0, held_out_docs, Config.BATCH_SIZE):
            chunk, origs = [], []
            for i in idx[b:b + Config.BATCH_SIZE]:
                t = text_to_ids(held[i]["text"])[:L]
                origs.append("".join(chr(c) for c in t if c > 2))
                chunk.append(t + [1] + [0] * (L - 1 - len(t)) if len(t) < L else t)
            toks = torch.tensor(chunk, device=dev)
            ph = torch.tensor([prefix_hash(c) for c in chunk], device=dev)
            torch.manual_seed(b)
            leveled, _ = m.forward({"full_texts": toks, "prefix_hash": ph})
            for layer in m.layers:
                layer._pending_clean_for_queue = None
                layer._pending_ids_for_queue = None
            for d in range(len(chunk)):
                for lv in range(1, top + 1):
                    codes = leveled[lv][d]
                    if not bool(codes[0].abs().sum() > 0):
                        continue
                    o = decode_down(m, lv, codes[0:1])
                    g = origs[d]
                    sims[lv].append(difflib.SequenceMatcher(None, o, g[:max(1, len(o))]).ratio())
                    exact[lv].append(float(o.strip() == g[:len(o)].strip() and len(o.strip()) > 0))
                    if lv == top and len(examples) < 4:
                        examples.append([g[:len(o) + 10], o])
        for lv in range(1, top + 1):
            res[f"recon_L{lv}_similarity"] = sum(sims[lv]) / max(1, len(sims[lv]))
            res[f"recon_L{lv}_exact"] = sum(exact[lv]) / max(1, len(exact[lv]))
        res["recon_examples"] = examples
        m.train()
    return res
