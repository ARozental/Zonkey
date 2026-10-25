"""FastSingleDeviceMuonWithAuxAdam matches SingleDeviceMuonWithAuxAdam (same updates up to bf16
rounding in the Newton-Schulz iteration, same state layout). GPU if available, else CPU.

Run: python tests/test_fast_optimizer.py
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from muon import FastSingleDeviceMuonWithAuxAdam, SingleDeviceMuonWithAuxAdam


def _groups(model):
    muon = [p for p in model.parameters() if p.ndim == 2]
    other = [p for p in model.parameters() if p.ndim != 2]
    return [dict(params=muon, lr=0.01, momentum=0.95, weight_decay=0.01, use_muon=True),
            dict(params=other, lr=1e-3, eps=1e-7, use_muon=False)]


def test_fast_optimizer_matches_reference():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    ref = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.GELU(), torch.nn.Linear(128, 128),
                              torch.nn.GELU(), torch.nn.Linear(128, 64), torch.nn.Linear(64, 128)).to(dev)
    fast = copy.deepcopy(ref)
    opt_ref = SingleDeviceMuonWithAuxAdam(_groups(ref))
    opt_fast = FastSingleDeviceMuonWithAuxAdam(_groups(fast))
    for step in range(5):
        x = torch.randn(32, 64, device=dev)
        for model, opt in ((ref, opt_ref), (fast, opt_fast)):
            torch.manual_seed(step)
            loss = model(x).pow(2).mean()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
    worst = max(float((a - b).abs().max() / a.abs().max()) for a, b in zip(ref.parameters(), fast.parameters()))
    assert worst < 2e-2, worst
    sd_ref, sd_fast = opt_ref.state_dict(), opt_fast.state_dict()
    assert sd_ref["state"].keys() == sd_fast["state"].keys()
    for k in sd_ref["state"]:
        assert sd_ref["state"][k].keys() == sd_fast["state"][k].keys()
    print(f"fast optimizer matches the reference (max relative parameter difference {worst:.1e}), same state layout")


if __name__ == "__main__":
    test_fast_optimizer_matches_reference()
    print("ALL TESTS PASSED")
