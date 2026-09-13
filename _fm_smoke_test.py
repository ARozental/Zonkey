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
pio = types.ModuleType("pytorch_lightning.plugins.io")
pio.TorchCheckpointIO = type("TorchCheckpointIO", (object,), {})
ppl = types.ModuleType("pytorch_lightning.plugins")
ppl.io = pio
pl.callbacks = cb
pl.plugins = ppl
sys.modules["pytorch_lightning"] = pl
sys.modules["pytorch_lightning.callbacks"] = cb
sys.modules["pytorch_lightning.plugins"] = ppl
sys.modules["pytorch_lightning.plugins.io"] = pio
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
from models.zonkey import PlZonkey, Zonkey, aggregate_leveled_losses
from models.zonkey_layer import ZonkeyLayer

for d in (8, 64):
    x = torch.nn.functional.normalize(torch.randn(16, d), dim=-1)
    y = torch.nn.functional.normalize(torch.randn(16, d), dim=-1)
    recovered = ZonkeyLayer._sphere_exp_map(x, ZonkeyLayer._sphere_log_map(x, y))
    assert torch.allclose(recovered, y, atol=2e-5), f"sphere log/exp mismatch at d={d}"

device = "cpu"
model = Zonkey().to(device)
model.train()

# The denoiser must receive the exact compressed prompt, not decoder-rewritten
# prompt states.
captured_denoiser_input = {}
def capture_denoiser_input(_module, args):
    captured_denoiser_input["x"] = args[0].detach().clone()

hook = model.layers[0].denoiser.register_forward_pre_hook(capture_denoiser_input)
probe = torch.nn.functional.normalize(
    torch.randn(2, Config.COMPRESSION_VECTORS[0], Config.D_MODEL[0]).reshape(2, -1),
    dim=-1,
).reshape(2, Config.COMPRESSION_VECTORS[0], Config.D_MODEL[0])
probe = probe * model.layers[0].upwards_norm
with torch.no_grad():
    model.layers[0].compressed_to_denoised(probe, torch.zeros(2))
hook.remove()
assert torch.allclose(
    captured_denoiser_input["x"][:, 2:2 + Config.COMPRESSION_VECTORS[0]], probe
)

# Muon receives only 2-D hidden matrices; Conv1d and prediction heads stay AdamW.
old_use_muon = Config.USE_MUON
Config.USE_MUON = True
configured = PlZonkey.configure_optimizers(types.SimpleNamespace(model=model))["optimizer"]
muon_ids = {
    id(p)
    for group in configured.param_groups if group.get("use_muon")
    for p in group["params"]
}
names = dict(model.named_parameters())
assert all(p.ndim == 2 and min(p.shape) > 1 for p in model.parameters() if id(p) in muon_ids)
assert id(names["layers.0.local_feature_extractor.conv.weight"]) not in muon_ids
assert id(names["layers.0.bos_layer.weight"]) not in muon_ids
Config.USE_MUON = old_use_muon
del configured

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

    total_loss, per_level_t = aggregate_leveled_losses(leveled_losses)
    per_level = [t.item() for t in per_level_t]

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
    if step == 0:
        keys0 = set(leveled_losses[0].keys())
        keys1 = set(leveled_losses[1].keys())
        for k in ("fm_flow_velocity_loss", "average_bos_loss", "avg_bos_prob",
                  "metric_clean_existence_mae"):
            assert k in keys0, f"missing {k} at L0, have {sorted(keys0)}"
            assert k in keys1, f"missing {k} at L1, have {sorted(keys1)}"
        for k in ("interface_sequence_loss", "interface_existence_loss",
                  "metric_clean_positive_cosine"):
            assert k in keys1, f"missing {k} at L1, have {sorted(keys1)}"
        assert "metric_clean_token_accuracy" in keys0
        print("  flow/interface/clean metrics present")
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
    model.generate_sequence_from_level_N(1, fixed_compressed_vectors=leveled_compressed[1][0][0:1],
                                         noise_level=0.0, lower_t=0.0)
    print("  generate CLEAN from level 1 OK")

    # Mirror the exact eval-block primitives in zonkey.py:245-250.
    lvl = 1
    nl = Config.NOISE_LAST_STEP_SIZE[lvl]
    model.generate_sequence_from_level_N(
        lvl, fixed_compressed_vectors=leveled_compressed[lvl][0][0:1],
        noise_level=nl, lower_t=0.0)
    print("  single-noise diagnostic OK")
    den_rand, _, _ = model.layers[lvl].generate(num_diffusion_steps=20, noise_level=1.0)
    print(f"  20-step random ODE @ level 1 -> {tuple(den_rand.shape)} OK")

print("\nALL GOOD")
