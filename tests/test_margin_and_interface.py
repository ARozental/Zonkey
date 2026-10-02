"""Pure-tensor checks for the on-manifold margin and the masked queue use (CPU, no dataset).

Run: python tests/test_margin_and_interface.py   (or with pytest)
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default_config import Config

Config.DEVICE = "cpu"

import torch
import torch.nn.functional as F

from models.zonkey_layer import ZonkeyLayer


def _fake_layer(n_queue, D, filled):
    queue = torch.zeros(n_queue, D)
    queue_ids = torch.full((n_queue,), -1, dtype=torch.long)
    queue[:filled] = F.normalize(torch.randn(filled, D), dim=-1)
    queue_ids[:filled] = torch.arange(1000, 1000 + filled)
    return types.SimpleNamespace(_drifting_queue=queue, _drifting_queue_ids=queue_ids,
                                 upwards_norm=3.0, _slerp=ZonkeyLayer._slerp)


def test_neighbor_margin_stays_in_own_cell():
    torch.manual_seed(0)
    Config.CLEAN_NEIGHBOR_FRACTION = 1.0
    Config.CLEAN_NEIGHBOR_MAX_STEP = 0.4
    n, cv, d = 64, 4, 8
    D = cv * d
    layer = _fake_layer(200, D, filled=150)
    codes = (F.normalize(torch.randn(n, D), dim=-1) * 3.0).view(n, cv, d).requires_grad_(True)
    ids = torch.arange(n)
    ids[1] = ids[0]                                   # window 1 owns the same text as window 0
    t_aug = torch.full((n,), 0.01)
    t_aug[:4] = 0.0                                   # exact samples are never moved
    noisy = codes.detach() * 1.0
    moved = ZonkeyLayer._neighbor_margin(layer, codes, noisy, t_aug, ids)
    assert torch.equal(moved[:4], noisy[:4]), "exact samples must stay exact"
    m = F.normalize(moved.detach().reshape(n, -1), dim=-1)
    own = F.normalize(codes.detach().reshape(n, -1), dim=-1)
    pool = torch.cat([own, layer._drifting_queue[:150]])
    pool_ids = torch.cat([ids, layer._drifting_queue_ids[:150]])
    for i in range(4, n):
        cos_own = float(m[i] @ own[i])
        others = (pool_ids != ids[i])
        assert float((pool[others] @ m[i]).max()) < cos_own, f"sample {i} left its own cell"
    assert not torch.allclose(moved[4:], noisy[4:]), "noised samples should have been moved"
    moved.sum().backward()
    assert codes.grad is not None and codes.grad[4:].abs().sum() > 0
    print("margin: moved codes stay closest to their own code; gradient reaches the own code")


def test_neighbor_margin_off_keeps_noise():
    Config.CLEAN_NEIGHBOR_FRACTION = 0.0
    layer = _fake_layer(10, 8, filled=5)
    codes = torch.randn(6, 2, 4)
    noisy = torch.randn(6, 2, 4)
    out = ZonkeyLayer._neighbor_margin(layer, codes, noisy, torch.full((6,), 0.01), torch.arange(6))
    assert torch.equal(out, noisy)
    Config.CLEAN_NEIGHBOR_FRACTION = 0.5
    print("margin: fraction 0 leaves the isotropic noise untouched")


def test_mined_negatives_skip_empty_queue_slots():
    D = 16
    Config.MINED_NEGATIVES_K = 4
    target = F.normalize(torch.randn(D), dim=0)
    real = F.normalize(torch.randn(6, D), dim=-1)
    empty = torch.zeros(10, D)                         # unfilled queue slots (zero vectors)
    bank = torch.cat([real, empty])
    bank_ids = torch.cat([torch.arange(50, 56), torch.full((10,), -1)])
    denoised = target.view(1, 1, D)
    extra = {"bank": bank, "bank_ids": bank_ids, "target_ids": torch.tensor([[7]])}
    sims = ZonkeyLayer._extra_negative_sims(types.SimpleNamespace(), denoised, torch.ones(1, 1), extra)
    expected = (real @ target).topk(4).values
    assert torch.allclose(sims.view(-1).sort(descending=True).values, expected, atol=1e-5)
    print("mined negatives never use empty queue slots")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL TESTS PASSED")
