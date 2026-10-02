"""Pure-tensor checks for the changes specified on top of 659d2e9 (no dataset, CPU).

Run: python tests/test_changes_659d2e9.py   (or with pytest)
"""
import math
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default_config import Config

# Tiny CPU config before any model import (modules read Config at construction).
Config.DEVICE = "cpu"
Config.D_MODEL = [16, 64]
Config.TOKEN_EMBEDDING_SIZE = 16
Config.COMPRESSION_VECTORS = [4, 4]
Config.NUM_HEADS = [2, 2]
Config.MAX_SEQ_LENGTHS = [27, 27]
Config.MAX_DOC_LENGTHS = [1024, 128, 16]
Config.MAX_SEQUENCES_PER_BATCH = [384, 48]
Config.BATCH_SIZE = 3

import torch
import torch.nn.functional as F

from utils.content_id import prefix_hash, prefix_hash_tensor, pow_table, span_content_id, HASH_MOD
from splitter.segment_splitter import SegmentSplitter
from models.zonkey_layer import ZonkeyLayer


def _word_like_bos(T, real_len, gen):
    """Synthetic word-like BOS probabilities: 0.95 at boundaries (segments of 4 to 9
    characters), 0.02 inside, 0 past the real text."""
    p = torch.full((T,), 0.02)
    pos = 0
    while pos < real_len:
        p[pos] = 0.95
        pos += int(torch.randint(4, 10, (1,), generator=gen))
    p[real_len:] = 0.0
    return p


class _FixedBos(torch.nn.Module):
    """Stands in for the splitter's BOS classifier: returns fixed synthetic probabilities."""
    def __init__(self, probs):
        super().__init__()
        self.probs = probs

    def forward(self, x):
        return self.probs.clone()


def _run_splitter(T=1024, real_lens=(1024, 1024, 1024), seed=0):
    gen = torch.Generator().manual_seed(seed)
    sp = SegmentSplitter(level=0, max_num_sentences=384)
    B = len(real_lens)
    probs = torch.stack([_word_like_bos(T, rl, gen) for rl in real_lens])
    sp.bos_classifier = _FixedBos(probs)
    x = torch.randn(B, T, Config.D_MODEL[0])
    is_real = torch.zeros(B, T)
    for b, rl in enumerate(real_lens):
        is_real[b, :rl] = 1.0
    out = sp(x, is_real, deterministic=True)
    return sp, out, probs, is_real


def test_c1_share_tail_mass():
    """The last kept window of a capped document must not own the next (dropped) window's text."""
    sp, out, probs, is_real = _run_splitter()
    share = out[3]                       # all_p_exist_weight
    num = out[5]                         # num_sentences_per_doc
    starts, owned_end = out[10], out[11]
    L = Config.MAX_SEQ_LENGTHS[0]
    i, tail_masses = 0, []
    for b in range(num.shape[0]):
        n = int(num[b])
        last = i + n - 1
        offset = int(owned_end[last] - starts[last])         # first position past its own segment
        tail_masses.append(float(share[last, offset:].sum()))
        i += n
    assert int(num[0]) == Config.MAX_DOC_LENGTHS[1], "documents should hit the per-document cap"
    assert max(tail_masses) < 2.0, f"tail share mass too high: {tail_masses}"

    # The old normalization (kept windows only) for comparison: tail mass of about 20.
    b, n = 0, int(num[0])
    bos_all = torch.nonzero(probs[0] > 0.5).view(-1)
    kept = bos_all[:n]
    gpos = kept.unsqueeze(1) + torch.arange(L)
    valid = gpos < probs.shape[1]
    tb = probs[0][gpos.clamp(max=probs.shape[1] - 1)]
    lp = torch.log1p(-tb.clamp(1e-4, 0.9999)); lp[:, 0] = 0
    pe = torch.exp(torch.cumsum(lp, 1)).masked_fill(~valid, 0)
    tot = torch.zeros(probs.shape[1]).index_add_(0, gpos[valid], pe[valid])
    old_share = pe / (tot[gpos.clamp(max=probs.shape[1] - 1)] + 1e-10)
    old_share = old_share * is_real[0][gpos.clamp(max=probs.shape[1] - 1)]
    offset = int(owned_end[n - 1] - starts[n - 1])
    old_tail = float(old_share[-1, offset:].sum())
    assert old_tail > 5.0, f"replica of the old normalization should show the bug, got {old_tail}"
    print(f"C1: tail share mass per capped doc {[round(m, 3) for m in tail_masses]} (old normalization {old_tail:.1f})")


def test_c2_owned_end_and_shapes():
    sp, out, probs, is_real = _run_splitter(real_lens=(1024, 300, 37))
    starts, owned_end, num = out[10], out[11], out[5]
    assert (owned_end > starts).all(), "owned_end must be > start for every kept window"
    assert (owned_end - starts <= Config.MAX_SEQ_LENGTHS[0]).all()
    i = 0
    for b in range(num.shape[0]):
        n = int(num[b])
        real_len = int(is_real[b].sum())
        assert (owned_end[i:i + n] <= real_len).all()
        assert (owned_end[i:i + n - 1] <= starts[i + 1:i + n]).all(), "owned spans must not overlap"
        i += n
    # The calibration path (every position a BOS) still works.
    sp.force_max_segments = True
    out2 = sp(torch.randn(3, 1024, Config.D_MODEL[0]), torch.ones(3, 1024), deterministic=True)
    assert out2[0].shape[0] > 0
    print("C2: owned_end checks passed")


def test_c2_content_ids():
    gen = torch.Generator().manual_seed(1)
    T = 200
    docs = torch.randint(2, 256, (3, T), generator=gen)
    docs[1, 50:60] = docs[0, 10:20]                       # the same text in another doc
    docs[2, 100:110] = docs[0, 10:20]                     # and at another position
    ph = prefix_hash_tensor(docs)
    pw = pow_table(T)
    # (i) the same range gets the same id anywhere
    d = torch.tensor([0, 1, 2]); a = torch.tensor([10, 50, 100]); b = a + 10
    ids = span_content_id(ph, pw, d, a, b, T)
    assert ids[0] == ids[1] == ids[2]
    # length is part of the id, and empty ranges are -1
    assert span_content_id(ph, pw, torch.tensor([0]), torch.tensor([10]), torch.tensor([10]), T)[0] == -1
    # (ii) different ranges get different ids (no collisions in a batch-sized sample)
    rd = torch.randint(0, 3, (5000,), generator=gen)
    ra = torch.randint(0, T - 30, (5000,), generator=gen)
    rl = torch.randint(1, 28, (5000,), generator=gen)
    rid = span_content_id(ph, pw, rd, ra, ra + rl, T)
    texts = {}
    for k in range(5000):
        key = tuple(docs[rd[k], ra[k]:ra[k] + rl[k]].tolist())
        texts.setdefault(int(rid[k]), set()).add(key)
    assert all(len(v) == 1 for v in texts.values()), "hash collision between different texts"
    # the ids match a direct hash of the range
    direct = prefix_hash(docs[0, 10:20].tolist())[-1]
    assert int(ids[0]) == direct * (T + 1) + 10
    # (iii) at level >= 1 a window's range is the concatenation of its children's ranges
    cuts = torch.tensor([0, 4, 9, 15, 22, 30, 37, 45])
    child_spans = torch.stack([cuts[:-1], cuts[1:]], dim=-1).unsqueeze(0)       # [1, 7, 2]
    start, owned_end = torch.tensor([1]), torch.tensor([4])                        # children 1..3
    a1 = child_spans[0, start, 0]; b1 = child_spans[0, owned_end - 1, 1]
    assert int(a1) == 4 and int(b1) == 22
    id_l1 = span_content_id(ph, pw, torch.tensor([0]), a1, b1, T)
    id_chars = span_content_id(ph, pw, torch.tensor([0]), torch.tensor([4]), torch.tensor([22]), T)
    assert int(id_l1) == int(id_chars), "the id must not depend on how the range was split"
    print("C2: content id checks passed")


def test_c3_identifiability_groups_ids():
    n, D = 6, 32
    codes = F.normalize(torch.randn(n, D), dim=-1)
    ids = torch.tensor([7, 7, 3, 4, 5, -1])
    fake = types.SimpleNamespace()
    # Windows 0 and 1 own the same text (same id) and have the same code: an estimate on that
    # code is fully identifiable for both, whichever copy it is closest to.
    codes[1] = codes[0]
    p = ZonkeyLayer._identifiability(fake, codes.clone(), codes, ids)
    assert p[0] > 0.99 and p[1] > 0.99, p
    # Without the shared id, the two copies split the probability.
    p_split = ZonkeyLayer._identifiability(fake, codes.clone(), codes, torch.tensor([7, 8, 3, 4, 5, -1]))
    assert p_split[0] < 0.6, p_split
    print("C3: identifiability groups identical content")


def test_c5b_mined_negatives_exclude_target_content():
    D = 16
    layer = types.SimpleNamespace()
    Config.MINED_NEGATIVES_K = 3
    target = F.normalize(torch.randn(D), dim=0)
    copies = F.normalize(target + 0.01 * torch.randn(4, D), dim=-1)      # same text, nearly identical
    others = F.normalize(torch.randn(20, D), dim=-1)
    bank = torch.cat([copies, others])
    bank_ids = torch.cat([torch.full((4,), 11), torch.arange(100, 120)])
    denoised = target.view(1, 1, D).clone().requires_grad_(True)
    extra = {"bank": bank, "bank_ids": bank_ids, "target_ids": torch.tensor([[11]])}
    sims = ZonkeyLayer._extra_negative_sims(layer, denoised, torch.ones(1, 1), extra)
    expected = (others @ target).topk(3).values
    assert torch.allclose(sims.view(-1).sort(descending=True).values, expected, atol=1e-5), "copies of the target leaked in"
    sims.sum().backward()
    assert denoised.grad is not None and bank.grad is None
    print("C5b: mined negatives exclude the target's content id")


def test_c6_schedule_maps():
    def layer(D):
        return types.SimpleNamespace(upwards_d_model=D)
    u = torch.linspace(0, 1, 101, dtype=torch.float64)
    t = ZonkeyLayer.u_to_t(layer(1024), u)
    assert torch.allclose(t, u, atol=1e-9), "identity at D = 1024"
    for D in (256, 1024, 4096):
        t = ZonkeyLayer.u_to_t(layer(D), u)
        back = ZonkeyLayer.t_to_u(layer(D), t)
        assert torch.allclose(back, u, atol=1e-9)
        assert (t[1:] >= t[:-1]).all(), f"not monotone at D={D}"
        assert abs(float(t[0])) < 1e-12 and abs(float(t[-1]) - 1.0) < 1e-12, (float(t[0]), float(t[-1]))
    t_half = float(ZonkeyLayer.u_to_t(layer(4096), torch.tensor([0.5], dtype=torch.float64))[0])
    assert abs(t_half - 0.705) < 0.002, t_half
    print(f"C6: schedule maps ok (D=4096, u=0.5 -> t={t_half:.4f})")


def test_c4c_vocab_ce():
    D, n, L = 24, 3, 5
    V = F.normalize(torch.randn(10, D), dim=-1)
    V[1] = V[0]                                            # two entries with the same text
    V_ids = torch.tensor([5, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    grand = types.SimpleNamespace(_drifting_queue_count=0, _drifting_queue=torch.zeros(0, D),
                                  _drifting_queue_ids=torch.zeros(0, dtype=torch.long))
    lower = types.SimpleNamespace(_last_doc_codes=V, _last_doc_ids=V_ids, previous_layer=grand)
    teacher = V[torch.tensor([0, 2, 3])].view(n, 1, D).expand(n, L, D).clone()
    w = torch.ones(n, L)
    p = torch.ones(n)
    right = ZonkeyLayer._interface_vocab_ce(lower, teacher.clone().requires_grad_(True), teacher, w, p)
    wrong_student = V[torch.tensor([4, 5, 6])].view(n, 1, D).expand(n, L, D).clone().requires_grad_(True)
    wrong = ZonkeyLayer._interface_vocab_ce(lower, wrong_student, teacher, w, p)
    assert float(right) < float(wrong), (float(right), float(wrong))
    # Landing on the other copy of the same text is not penalized (merged class).
    twin = V[torch.tensor([1, 2, 3])].view(n, 1, D).expand(n, L, D).clone()
    twin_ce = ZonkeyLayer._interface_vocab_ce(lower, twin, teacher, w, p)
    assert abs(float(twin_ce) - float(right)) < 1e-4
    wrong.backward()
    assert wrong_student.grad is not None
    zero_p = ZonkeyLayer._interface_vocab_ce(lower, wrong_student.detach(), teacher, w, torch.zeros(n))
    assert float(zero_p) == 0.0
    print(f"C4c: vocab CE right {float(right):.3f} < wrong {float(wrong):.3f}")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL TESTS PASSED")
