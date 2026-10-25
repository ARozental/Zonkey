"""CPU checks for the overhaul's new pieces: tokens per child, per-child normalization, splitter
boundaries only at child starts, the alignment assembly, per-level LR groups, hard negatives per child,
losses reaching one level down only.

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


def test_level_freezing_schedule():
    from models.zonkey import PlZonkey
    _three_level_config()
    Config.USE_MUON = True
    Config.LEVEL_LR_SCALE = None
    Config.FREEZE_LEVELS_UNTIL = [10, 10, 0]
    torch.manual_seed(3)
    pl = PlZonkey()
    opt = pl.configure_optimizers()["optimizer"]
    ema_len = len(pl._ema_params())
    pl._apply_level_freezing(0)
    m = pl.model
    assert m.layers[0].frozen and m.layers[1].frozen and not m.layers[2].frozen
    l0 = m.layers[0].compressor.layers[0].ff[0].weight
    l2 = m.layers[2].compressor.layers[0].ff[0].weight
    assert not l0.requires_grad and l2.requires_grad
    assert not m.layers[0].compressor_cls_existence_probs.requires_grad
    before0, before2 = l0.detach().clone(), l2.detach().clone()
    for p in m.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p)
    opt.step()
    assert torch.equal(l0, before0), "a frozen level must not move (no momentum, no decay)"
    assert not torch.equal(l2, before2)
    assert len(pl._ema_params()) == ema_len, "the EMA list stays fixed"
    pl._apply_level_freezing(10)
    assert not m.layers[0].frozen and l0.requires_grad
    assert not m.layers[0].compressor_cls_existence_probs.requires_grad, "fixed parameters stay fixed"
    Config.FREEZE_LEVELS_UNTIL = None
    print("FREEZE_LEVELS_UNTIL: frozen levels do not move, unfreeze on schedule, EMA list fixed")


def test_gathered_dot_matches_gather_of_matmul():
    from losses.reconstruction import gathered_dot
    torch.manual_seed(6)
    a = torch.randn(10, 8, dtype=torch.float64, requires_grad=True)
    b = torch.randn(7, 8, dtype=torch.float64, requires_grad=True)
    index = torch.randint(0, 7, (10, 5))
    index[0] = 3                                                 # repeated picks accumulate
    up = torch.randn(10, 5, dtype=torch.float64)
    ref = torch.gather(a @ b.t(), 1, index)
    ga, gb = torch.autograd.grad((ref * up).sum(), (a, b))
    out = gathered_dot(a, b, index, chunk=3)                     # chunks of 3, 3, 3, 1 rows
    ha, hb = torch.autograd.grad((out * up).sum(), (a, b))
    assert torch.allclose(out, ref) and torch.allclose(ha, ga) and torch.allclose(hb, gb)
    assert torch.autograd.gradcheck(lambda x, y: gathered_dot(x, y, index, chunk=4), (a, b))
    assert torch.autograd.grad(gathered_dot(a, b.detach(), index).sum(), a)[0].shape == a.shape
    assert gathered_dot(a[:0], b, index[:0]).shape == (0, 5)
    print("gathered_dot: the values and gradients of gather(a @ b.T), without the full matrix")


def test_context_guesses_are_whole_children():
    from models.zonkey import Zonkey
    _three_level_config()
    Config.MLM_ALT_PASSES, Config.MLM_ALT_FRACTION = 1, 0.5
    torch.manual_seed(4)
    m = Zonkey()
    l2 = m.layers[2]
    k, D, B, C = l2.k, Config.D_MODEL[2], 3, 4
    seq = l2.norm_tokens(torch.randn(B, C * k, D))
    real = torch.ones(B, C * k)
    share = torch.ones(B, C * k)
    share[:, -k:] = 0.0                                          # the last child is owned by no window
    guess, has = l2._mlm_alternatives(seq, real, share)
    assert guess.shape == (B, C, k * D) and has.shape == (B, C), (guess.shape, has.shape)
    assert bool(has.any()) and not bool(has[:, -1].any()), has
    # One pass: the guess is the denoiser's output with exactly the guessed children masked (all k tokens).
    masked = has.repeat_interleave(k, dim=1).unsqueeze(-1)
    x = torch.where(masked, l2.mask_vector.view(1, 1, D).expand(B, C * k, D), seq)
    prefix = l2._mlm_denoiser_prefix(B, seq.dtype)
    P = prefix.shape[1]
    with torch.no_grad():
        out = l2.denoiser(torch.cat([prefix, x], dim=1), torch.cat([torch.ones(B, P), real], dim=1))[:, P:]
    expected = l2.as_children(l2.norm_tokens(out))
    assert torch.allclose(guess[has], expected[has], atol=1e-5)
    assert torch.allclose(guess[has].norm(dim=-1), torch.full((int(has.sum()),), l2.child_norm), atol=1e-3)
    g1, h1 = m.layers[1]._mlm_alternatives(torch.randn(B, 6, Config.D_MODEL[1]), torch.ones(B, 6), torch.ones(B, 6))
    assert g1.shape == (B, 6, Config.D_MODEL[1]) and h1.shape == (B, 6)
    Config.MLM_ALT_PASSES, Config.MLM_ALT_FRACTION = 2, 0.25
    print("context guesses at k = 4: whole children masked, guessed and normalized per child")


def test_hard_negatives_reach_every_level_above_0():
    from models.zonkey import Zonkey, aggregate_leveled_losses
    _three_level_config()
    # 4 complementary passes of 1/4: every owned child gets a guess (the default 2 x 1/4 guesses half).
    Config.MLM_ALT_PASSES, Config.MLM_ALT_FRACTION, Config.MINED_NEGATIVES_K = 4, 0.25, 4
    torch.manual_seed(5)
    m = Zonkey()
    seen = {}
    for lvl in (1, 2):
        layer = m.layers[lvl]

        def spy(den, share, extra, orig=layer._extra_negative_sims, lvl=lvl):
            sims = orig(den, share, extra)
            seen[lvl] = (den.shape[-1], sims.shape[-1], int((share > 0).sum()), int((sims[..., 0] > -1).sum()),
                         int((sims[..., 1:] > -1).sum()))
            return sims
        layer._extra_negative_sims = spy
    texts = torch.randint(1, Config.TOKENIZER_VOCAB_SIZE_CHARS, (2, Config.MAX_DOC_LENGTHS[0]))
    _, losses = m.forward({"full_texts": texts})
    aggregate_leveled_losses(losses)[0].backward()
    for lvl in (1, 2):
        assert lvl in seen, f"level {lvl}: no extra negatives"
        width, cols, owned, guessed, mined = seen[lvl]
        assert width == m.layers[lvl].k * Config.D_MODEL[lvl], (lvl, width)
        assert cols == 1 + 4 and guessed == owned > 0 and mined > 0, (lvl, seen[lvl])
    Config.MLM_ALT_PASSES, Config.MINED_NEGATIVES_K = 2, 16
    print("clean reconstruction at levels 1 and 2 gets a context guess and mined neighbours per child:", seen)


def test_window_bos_probs_match_the_splitter_layout():
    from models.zonkey import Zonkey
    _three_level_config()
    torch.manual_seed(9)
    m = Zonkey()
    for lvl, T in ((1, Config.MAX_DOC_LENGTHS[1]), (2, Config.MAX_DOC_LENGTHS[2] * 4)):
        sp = m.layers[lvl].segment_splitter
        seq = torch.randn(3, T, Config.D_MODEL[lvl])
        real = torch.ones(3, T)
        real[1, T // 2:] = 0                                     # windows that run past a document's end
        out = sp(seq, real)
        bos, _ = sp.bos_probabilities(seq, real)
        assert torch.equal(sp.window_bos_probs(bos, out[7], out[2]), out[1]), lvl
    print("window_bos_probs rebuilds the splitter's window BOS probabilities exactly (k = 1 and k = 4)")


def test_losses_reach_one_level_down():
    import random
    from models.zonkey import Zonkey, aggregate_leveled_losses
    _three_level_config()
    torch.manual_seed(7)
    m = Zonkey()
    texts = torch.randint(1, Config.TOKENIZER_VOCAB_SIZE_CHARS, (2, Config.MAX_DOC_LENGTHS[0]))
    level_of = {}                                   # a layer's parameters include the levels below it
    for lvl in range(3):
        for p in m.layers[lvl].parameters():
            level_of.setdefault(id(p), lvl)
    params = list(m.parameters())

    def run(one_level, levels):
        Config.ONE_LEVEL_GRADIENTS = one_level
        m.zero_grad(set_to_none=True)
        torch.manual_seed(8)
        random.seed(8)
        codes, losses = m.forward({"full_texts": texts})
        totals = aggregate_leveled_losses(losses)[1]
        sum(totals[lvl] for lvl in levels).backward()
        grads = {id(p): p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params}
        return [c.detach() for c in codes], torch.stack([t.detach() for t in totals]), grads

    def size(grads, lvl):
        return sum(float(g.norm()) for i, g in grads.items() if level_of[i] == lvl)

    codes_old, totals_old, g2_old = run(False, [2])
    codes_new, totals_new, g2_new = run(True, [2])
    assert torch.allclose(totals_new, totals_old) and all(torch.allclose(a, b) for a, b in zip(codes_new, codes_old))
    assert size(g2_old, 0) > 0 and size(g2_new, 0) == 0, "level 2's losses must not reach level 0"
    assert size(g2_new, 1) > 0
    for i, g in g2_new.items():
        if level_of[i] > 0:                         # levels 1 and 2 get exactly the gradients they got before
            assert torch.allclose(g, g2_old[i], rtol=1e-4, atol=1e-7)
    assert any(float(g2_new[id(p)].norm()) > 0 for p in m.layers[1].segment_splitter.parameters())
    _, _, g01_old = run(False, [0, 1])
    _, _, gall_new = run(True, [0, 1, 2])
    for i, g in gall_new.items():
        if level_of[i] == 0:                        # level 1's losses still reach level 0 in full
            assert torch.allclose(g, g01_old[i], rtol=1e-4, atol=1e-7)
    Config.ONE_LEVEL_GRADIENTS = True
    print("ONE_LEVEL_GRADIENTS: level 2's losses train levels 2 and 1 as before and give level 0 no gradient")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL TESTS PASSED")
