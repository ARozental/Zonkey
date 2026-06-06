"""Standalone smoke test for the flow-matching (Path A) changes.

Runs 4 tiny training steps (forward + backward + optimizer step) on synthetic
batches, on CPU, with the tiny_local_debug_config. No HF download required.
Mirrors the loss aggregation in PlZonkey.training_step.
"""
import os, sys, json, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Stub training-infra modules we don't need (Zonkey nn.Module doesn't use them).
pl = types.ModuleType("pytorch_lightning")
pl.LightningModule = type("LightningModule", (object,), {})
cb = types.ModuleType("pytorch_lightning.callbacks")
cb.ModelCheckpoint = type("ModelCheckpoint", (object,), {})
pl.callbacks = cb
sys.modules["pytorch_lightning"] = pl
sys.modules["pytorch_lightning.callbacks"] = cb
try:
    import torch.utils.tensorboard  # noqa
except Exception:
    tb = types.ModuleType("torch.utils.tensorboard")
    tb.SummaryWriter = type("SummaryWriter", (object,), {})
    sys.modules["torch.utils.tensorboard"] = tb

from configs.default_config import Config

# Apply tiny config overrides BEFORE importing models.
with open("configs/tiny_local_debug_config.json") as f:
    params = json.load(f)
for k, v in params.items():
    if hasattr(Config, k.upper()):
        setattr(Config, k.upper(), v)
# Keep the smoke test light & fast on CPU.
Config.BATCH_SIZE = 2
Config.MAX_DOC_LENGTHS = [128, 64, 32]
Config.DEVICE = "cpu"
Config.USE_GRADIENT_CHECKPOINTING = bool(int(os.environ.get("GC", "0")))

import torch
torch.manual_seed(0)
from models.zonkey import Zonkey

device = "cpu"
model = Zonkey().to(device)
model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)

def make_batch():
    B, L = Config.BATCH_SIZE, Config.MAX_DOC_LENGTHS[0]
    toks = torch.zeros(B, L, dtype=torch.long)
    for b in range(B):
        n = int(torch.randint(40, L, (1,)).item())  # variable real length, rest padding(0)
        toks[b, :n] = torch.randint(1, Config.TOKENIZER_VOCAB_SIZE_CHARS, (n,))
    return {"full_texts": toks.to(device)}

print("=== FM Path A smoke test: 4 steps, tiny config, CPU ===")
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable params: {n_params:,}")

for step in range(4):
    batch = make_batch()
    leveled_compressed, leveled_losses = model.forward(batch)

    total_loss = 0.0
    per_level = []
    for l in range(len(leveled_losses)):
        vals = []
        for name, value in leveled_losses[l].items():
            if isinstance(value, torch.Tensor):
                vals.append(value if value.ndim == 0 else value.mean())
        lvl = torch.stack(vals).mean()
        per_level.append(lvl.item())
        total_loss = total_loss + lvl

    opt.zero_grad()
    total_loss.backward()

    # gradient sanity
    gnorm = 0.0
    n_grad = 0
    for p in model.parameters():
        if p.grad is not None:
            gnorm += float(p.grad.detach().pow(2).sum())
            n_grad += 1
    gnorm = gnorm ** 0.5
    opt.step()

    finite = torch.isfinite(total_loss).item()
    print(f"step {step}: total_loss={total_loss.item():.4f} per_level={['%.3f'%x for x in per_level]} "
          f"grad_norm={gnorm:.3f} params_with_grad={n_grad} finite={finite}")
    assert finite, "non-finite loss!"

# Exercise the new ODE sampler / generation paths used in the eval block.
print("\n=== sampler smoke (no_grad) ===")
with torch.no_grad():
    lc0 = leveled_compressed[0]  # (num_docs, max_sent, feat)
    feat = lc0[0][0:1]
    den, emask, isr = model.layers[0].generate(num_diffusion_steps=0, fixed_compressed_vectors=feat,
                                               noise_level=torch.tensor([0.0]))
    print(f"  decode@t=0  -> denoised {tuple(den.shape)}, exist_sum={emask.sum().item():.1f}")
    den2, _, _ = model.layers[0].generate(num_diffusion_steps=8, noise_level=1.0)
    print(f"  ODE 8-step from noise -> denoised {tuple(den2.shape)}")
    model.generate_sequence_from_level_N(1, fixed_compressed_vectors=leveled_compressed[1][0][0:1])
    print("  generate_sequence_from_level_N(1) OK")

    # Mirror the exact eval-block primitives in zonkey.py:245-250.
    lvl = 1
    nl = Config.NOISE_LAST_STEP_SIZE[lvl]
    noise_level = torch.full((leveled_compressed[lvl].shape[0],), nl, dtype=leveled_compressed[lvl].dtype)
    noised = model.layers[lvl].add_noise(leveled_compressed[lvl], noise_level)
    print(f"  add_noise on 3-D leveled tensor {tuple(noised.shape)} OK")
    model.generate_sequence_from_level_N(lvl, fixed_compressed_vectors=noised[0][0:1], noise_level=nl)
    den_rand, _, _ = model.layers[lvl].generate(num_diffusion_steps=20, noise_level=1.0)
    print(f"  20-step random ODE @ level 1 -> {tuple(den_rand.shape)} OK")

print("\nALL GOOD")
