"""Pure-tensor checks for the fused decoder attention, the masked coverage queue, negatives
that can never be the target child itself, and the stitcher's sequence-loss scale (CPU).

Run: python tests/test_attention_coverage_negatives.py   (or with pytest)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default_config import Config

Config.DEVICE = "cpu"

import torch
import torch.nn.functional as F

from models.transformer import DecoderAttention, TransformerDecoder
from losses.reconstruction import calculate_reconstruction_loss
from utils.helper_functions import compute_improved_coverage_loss
from utils.content_id import slot_source_ids
from splitter.stitcher import Stitcher


def test_decoder_attention_matches_explicit_causal_mask():
    torch.manual_seed(0)
    att = DecoderAttention(d_model=32, n_heads=4, max_seq_len=16)
    x = torch.randn(3, 10, 32)
    out = att(x)
    # The previous implementation: explicit -inf upper triangle, softmax, then XSA.
    qkv = att.W_qkv(x).reshape(3, 10, 3, 4, 8).permute(2, 0, 3, 1, 4)
    Q, K, V = att.pos(qkv[0], 10), att.pos(qkv[1], 10), qkv[2]
    scores = Q @ K.transpose(-2, -1) * att.scale
    scores = scores + torch.triu(torch.full((10, 10), float("-inf")), diagonal=1)
    o = torch.softmax(scores, dim=-1) @ V
    Vn = F.normalize(V, dim=-1)
    o = o - (o * Vn).sum(-1, keepdim=True) * Vn
    ref = att.W_o(o.transpose(1, 2).reshape(3, 10, 32))
    assert torch.allclose(out, ref, atol=1e-5), float((out - ref).abs().max())
    print("decoder attention: fused causal kernel == explicit mask")


def test_decoder_kv_cache_generation_still_consistent():
    torch.manual_seed(0)
    dec = TransformerDecoder(d_model=32, n_heads=4, d_ff=64, max_seq_len=16, num_layers=2)
    prompt = torch.randn(2, 5, 32)
    gen = dec.generate(prompt, 2)                      # prompt pass (fused) + cached steps

    def layers_only(x):                                # generate() has no final layer norm
        for layer in dec.layers:
            x, _ = layer.decode(x)
        return x
    assert torch.allclose(gen[:, :5], layers_only(prompt), atol=1e-5)
    full = layers_only(torch.cat([prompt, gen[:, 4:5]], dim=1))
    assert torch.allclose(gen[:, 5], full[:, 5], atol=1e-5)
    print("decoder: cached generation matches the full causal pass")


def test_coverage_full_queue_with_mask_matches_slice():
    torch.manual_seed(0)
    q = torch.zeros(10, 8)
    q[:7] = F.normalize(torch.randn(7, 8), dim=-1)
    doc_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    z1 = torch.randn(6, 2, 4, requires_grad=True)
    z2 = z1.detach().clone().requires_grad_(True)
    a = compute_improved_coverage_loss(z1, doc_ids=doc_ids, memory_queue=q[:7])
    b = compute_improved_coverage_loss(z2, doc_ids=doc_ids, memory_queue=q,
                                       memory_valid=q.abs().amax(dim=-1) > 0)
    a.backward()
    b.backward()
    assert torch.allclose(a, b) and torch.allclose(z1.grad, z2.grad, atol=1e-7)
    # no valid neighbour anywhere: 0, as before
    lone = compute_improved_coverage_loss(torch.randn(1, 2, 4), doc_ids=torch.tensor([0]))
    assert float(lone) == 0.0
    print("coverage: whole queue + filled mask == sliced queue (value and gradient)")


def test_negatives_never_the_same_child():
    """Slot 1 holds the same child as slot 0 (same document position, as overlapping windows
    do). Slot 0 predicts its target exactly; only slot 0 is weighted. If the copy were drawn as
    a negative its logit would equal the positive's, so the loss would be about log(#copies)."""
    D = 16
    t = torch.zeros(4, D)
    t[0, 0] = 1.0
    t[1] = t[0]                                   # the same vector in another window's slot
    t[2, 3] = 1.0                                 # unrelated children: zero similarity
    t[3, 5] = 1.0
    denoised = t.clone().view(1, 4, D)
    targets = t.view(1, 4, D)
    real = torch.ones(1, 4)
    share = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    orig = torch.tensor([[[0, 7], [0, 7], [0, 9], [0, 11]]])   # slot 1 = doc 0, position 7 again
    torch.manual_seed(1)
    with_ids, _ = calculate_reconstruction_loss(denoised, real, targets, share, num_negatives=63,
                                                source_ids=slot_source_ids(orig))
    torch.manual_seed(1)
    without, _ = calculate_reconstruction_loss(denoised, real, targets, share, num_negatives=63)
    assert float(with_ids) < 1e-3, float(with_ids)
    assert float(without) > 1.0, float(without)
    print(f"negatives: same-child copies redrawn (loss {float(with_ids):.2e}, "
          f"with the copy {float(without):.2f})")


def test_stitcher_sequence_loss_is_a_mean_not_divided_by_pairs():
    torch.manual_seed(0)
    V, D, L = Config.TOKENIZER_VOCAB_SIZE_CHARS, Config.D_MODEL[0], Config.MAX_SEQ_LENGTHS[0]
    emb = torch.nn.Embedding(V, D)
    saved = Config.MAX_SEQUENCES_PER_BATCH
    Config.MAX_SEQUENCES_PER_BATCH = [8, 8, 4]      # keeps the generation scratch buffer small
    try:
        st = Stitcher(level=0, token_embedding_layer=emb)
    finally:
        Config.MAX_SEQUENCES_PER_BATCH = saved
    n = 4                                           # one document, 4 windows -> 3 pairs
    starts = torch.tensor([0, 5, 9, 14])
    pos = starts[:, None] + torch.arange(L)[None, :]
    orig = torch.stack([torch.zeros_like(pos), pos], dim=-1)
    tokens = torch.randint(3, V, (n, L))
    seqs = F.normalize(emb(tokens).detach() + 0.3 * torch.randn(n, L, D), dim=-1) * (D ** 0.5)
    real = torch.linspace(1, 0.1, L).expand(n, L).contiguous()
    share = real.clone()
    _, pos_loss, seq_loss = st(seqs, real, torch.tensor([n]), original_position=orig,
                               original_input_sequences=seqs, all_p_exist_share=share,
                               all_tokens=tokens, assemble=False)
    left, right = torch.arange(3), torch.arange(1, 4)
    _, p_direct, s_direct, _ = st.merge_2_sequences(
        seqs[left], real[left], seqs[right], real[right], (starts[right] - starts[left]).float(),
        seqs[left], share[left], all_tokens=tokens[left])
    assert torch.allclose(seq_loss, s_direct.mean(), atol=1e-6), (float(seq_loss), float(s_direct))
    assert torch.allclose(pos_loss, p_direct.mean(), atol=1e-6)
    print(f"stitcher: sequence loss {float(seq_loss):.3f} = the pairs' mean CE (not /3)")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL TESTS PASSED")
