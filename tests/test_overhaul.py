"""CPU checks for the overhaul's new pieces: tokens per child, per-child normalization, splitter
boundaries only at child starts, the alignment assembly, per-level LR groups.

Run: python tests/test_overhaul.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default_config import Config

Config.DEVICE = "cpu"
Config.COMPILE_TRANSFORMER_STACKS = False

import torch
import torch.nn.functional as F


def _three_level_config():
    Config.AGENT_LEVELS = 3
    Config.D_MODEL = [32, 128, 128]
    Config.TOKEN_EMBEDDING_SIZE = 32
    Config.COMPRESSION_VECTORS = [4, 4, 4]
    Config.NUM_HEADS = [4, 4, 4]
    Config.MAX_SEQ_LENGTHS = [12, 12, 16]
    Config.MAX_DOC_LENGTHS = [64, 24, 8, 4]
    Config.MAX_SEQUENCES_PER_BATCH = [48, 16, 8]
    Config.NUM_UPWARD_LAYERS = [1, 1, 1]
    Config.NUM_DENOISER_LAYERS = [1, 1, 1]
    Config.NUM_DECOMPRESSOR_LAYERS = [1, 1, 1]
    for key in ("COVERAGE_WEIGHT", "COMPRESSION_PENALTY", "MLM_WEIGHT", "DIRTY_MLM_WEIGHT", "DIRTY_RECONSTRUCTION_WEIGHT",
                "CLEAN_RECONSTRUCTION_WEIGHT", "NOISE_STEP_SIZE", "NOISE_LAST_STEP_SIZE", "DRIFTING_QUEUE_SIZE",
                "EXISTS_WEIGHT", "FLOW_VELOCITY_WEIGHT", "INTERFACE_CONSISTENCY_WEIGHT", "NUM_FAKE_NEGATIVES"):
        v = list(getattr(Config, key))
        setattr(Config, key, (v + [v[-1]] * 3)[:3])
    Config.DRIFTING_QUEUE_SIZE = [64, 64, 64]
    Config.NUM_FAKE_NEGATIVES = [0, 0, 0]
    Config.STITCHER_REFINE = False


def test_child_tokens():
    from splitter.segment_splitter import child_tokens
    _three_level_config()
    assert [child_tokens(l) for l in range(3)] == [1, 1, 4]
    print("tokens per child: level 1 widens (1 token per child), level 2 keeps the width (4 tokens)")


def test_norm_tokens_keeps_relative_norms():
    from models.zonkey_layer import ZonkeyLayer
    from models.zonkey import Zonkey
    _three_level_config()
    torch.manual_seed(0)
    m = Zonkey()
    l2 = m.layers[2]
    child = torch.randn(5, 4 * 128) * torch.tensor([1.0, 2.0, 0.5, 1.5]).repeat_interleave(128)
    tokens = child.view(5, 4, 128)
    out = l2.norm_tokens(tokens.view(1, 20, 128)).view(5, 4, 128)
    cos = F.cosine_similarity(out.reshape(5, -1), child, dim=-1)
    assert torch.allclose(cos, torch.ones(5), atol=1e-5), cos
    assert abs(float(out.reshape(5, -1).norm(dim=-1).mean()) - l2.child_norm) < 1e-3
    print("per-child normalization keeps each child code's direction (its tokens' relative norms)")


def test_splitter_boundaries_only_at_child_starts():
    from models.zonkey import Zonkey
    _three_level_config()
    torch.manual_seed(1)
    m = Zonkey()
    sp = m.layers[2].segment_splitter
    seq = torch.randn(2, 8 * 4, 128)
    real = torch.ones(2, 32)
    out = sp(seq, real)
    starts = out[10]                        # all_bos_starts (token positions)
    assert bool((starts % 4 == 0).all()), starts
    print("level-2 windows start only at child starts:", starts.tolist())


def test_align_assembly_cuts_where_the_next_window_begins():
    from models.zonkey import Zonkey
    _three_level_config()
    torch.manual_seed(2)
    m = Zonkey()
    Config.STITCHER = False
    Config.ASSEMBLY = "align"
    d = Config.D_MODEL[0]
    text = torch.randn(20, d)                                   # a "document" of 20 positions
    L = Config.MAX_SEQ_LENGTHS[0]
    starts = [0, 5, 11]
    seq = torch.stack([F.pad(text[s:s + L], (0, 0, 0, max(0, L - len(text[s:s + L])))) for s in starts])
    # soft existence: certain on the owned span, 0.6 for one position past it, then ~0
    ends = [5, 11, 20]
    e = torch.full((3, L), 1e-3)
    for i, (s, t) in enumerate(zip(starts, ends)):
        e[i, :t - s] = 1.0
        if t - s < L:
            e[i, t - s] = 0.6
    em = (e > 0.1).float()
    doc = m.assemble(0, seq, em, e).squeeze(0)
    assert doc.shape[0] == 20 + 1, doc.shape                 # the last window keeps its 0.6 position
    assert torch.allclose(doc[:20], text), "the cuts must follow the next windows' starts"
    Config.STITCHER = True
    print("alignment assembly: soft ends cut where the next window begins")


def test_level_lr_groups():
    from models.zonkey import PlZonkey
    _three_level_config()
    Config.USE_MUON = True
    Config.LEVEL_LR_SCALE = [0.3, 0.3, 1.0]
    pl = PlZonkey()
    opt = pl.configure_optimizers()["optimizer"]
    scales = sorted({round(g["level_lr_scale"], 3) for g in opt.param_groups})
    assert scales == [0.3, 1.0], scales
    for g in opt.param_groups:
        base = Config.MUON_LR if g["use_muon"] else Config.LEARNING_RATE
        assert abs(g["base_lr"] - base * g["level_lr_scale"]) < 1e-12
    Config.LEVEL_LR_SCALE = None
    assert len(pl.configure_optimizers()["optimizer"].param_groups) == 2
    print("LEVEL_LR_SCALE: one Muon and one Adam group per level with scaled LRs; None keeps 2 groups")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL TESTS PASSED")
