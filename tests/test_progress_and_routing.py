"""Pure-tensor checks for the extra-negative gradient routing and loss/progress (CPU, no dataset).

Run: python tests/test_progress_and_routing.py   (or with pytest)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default_config import Config

Config.DEVICE = "cpu"

import torch

from losses.reconstruction import calculate_reconstruction_loss, calculate_token_loss
from models.zonkey import aggregate_leveled_losses, aggregate_progress


def _recon(extra, seed=1):
    torch.manual_seed(0)
    B, L, D = 6, 9, 32
    den = torch.randn(B, L, D, dtype=torch.float64, requires_grad=True)
    tgt = torch.randn(B, L, D, dtype=torch.float64, requires_grad=True)
    real = (torch.rand(B, L) > 0.2).double()
    share = real * torch.rand(B, L, dtype=torch.float64)
    fakes = torch.randn(4, D, dtype=torch.float64)
    ext = None
    if extra:
        # the layer's extras are cosines of the prediction to detached codes
        bank = torch.nn.functional.normalize(torch.randn(5, D, dtype=torch.float64), dim=-1)
        ext = torch.nn.functional.normalize(den, dim=-1) @ bank.t()
    out = {}
    torch.manual_seed(seed)
    loss, _ = calculate_reconstruction_loss(den, real, tgt, share, num_negatives=7, fake_negatives=fakes,
                                            extra_neg_sim=ext, metric_out=out)
    loss.backward()
    return loss.detach(), den.grad, tgt.grad, out["value"]


def test_extra_negatives_only_sharpen_the_prediction():
    l_plain, gd_plain, gt_plain, m_plain = _recon(extra=False)
    l_x, gd_x, gt_x, m_x = _recon(extra=True)
    assert torch.allclose(gt_x, gt_plain, atol=1e-12), "child codes must get the plain in-batch gradient"
    assert not torch.allclose(gd_x, gd_plain), "the extras must still shape the prediction"
    assert float(l_x) > float(l_plain), "the logged value is the CE with the extras"
    assert torch.allclose(m_x, l_plain) and torch.allclose(m_plain, l_plain), "progress = plain CE"
    print("extra negatives: targets get the plain CE gradient, the prediction the full one; progress = plain CE")


def test_progress_ignores_the_gate():
    torch.manual_seed(0)
    B, L, D, V = 5, 7, 16, 11
    table = torch.nn.Embedding(V, D)
    tokens = torch.randint(0, V, (B, L))
    enc = torch.randn(B, L, D)
    share = torch.rand(B, L)
    t_weight = torch.rand(B)
    gate = torch.rand(B)
    out = {}
    gated, _ = calculate_token_loss(tokens, enc, share, table, sample_weight=t_weight * gate,
                                    metric_out=out, metric_sample_weight=t_weight)
    ungated, _ = calculate_token_loss(tokens, enc, share, table, sample_weight=t_weight)
    assert torch.allclose(out["value"], ungated) and not torch.allclose(gated, ungated)
    print("progress: gated reconstruction scored without the gate")


def test_aggregate_progress_substitutes_terms():
    one = lambda v: torch.tensor(float(v))
    levels = [{"a": one(1.0), "b": one(3.0), "metric_progress_b": one(2.0), "metric_x": one(9.0)},
              {"c": one(4.0)}]
    total, _ = aggregate_leveled_losses(levels)
    progress, per_level = aggregate_progress(levels)
    assert abs(float(total) - (2.0 + 4.0)) < 1e-6
    assert abs(float(progress) - (1.5 + 4.0)) < 1e-6 and abs(float(per_level[1]) - 4.0) < 1e-6
    print("loss/progress: same aggregation as loss/total with the progress terms substituted")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL TESTS PASSED")
