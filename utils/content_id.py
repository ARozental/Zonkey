"""Data-defined identity of the text a window owns (its "content id").

Two windows, at any level, own the same text iff the character ranges they own hash equal.
The id is a polynomial rolling hash of that character range with its length folded in, so it
does not depend on how the range was split into children, nor on any learned vector.
"""
import torch

HASH_MOD = 2**31 - 1          # prime; every product stays below 2**62 in int64
HASH_BASE = 1_000_003


def prefix_hash(tokens):
    """len(tokens) + 1 values; h[i] = hash of tokens[:i]."""
    h = [0]
    for c in tokens:
        h.append((h[-1] * HASH_BASE + int(c)) % HASH_MOD)
    return h


def prefix_hash_tensor(token_batch: torch.Tensor) -> torch.Tensor:
    """[B, T] token ids -> [B, T + 1] int64 prefix hashes (CPU loop; fallback path only)."""
    rows = [prefix_hash(row.tolist()) for row in token_batch.detach().cpu()]
    return torch.tensor(rows, dtype=torch.long, device=token_batch.device)


def pow_table(max_len: int, device=None) -> torch.Tensor:
    """POW[m] = HASH_BASE**m mod HASH_MOD for m in [0, max_len]."""
    p = [1]
    for _ in range(max_len):
        p.append((p[-1] * HASH_BASE) % HASH_MOD)
    return torch.tensor(p, dtype=torch.long, device=device)


def span_content_id(prefix_hash_b: torch.Tensor, pow_tab: torch.Tensor, doc: torch.Tensor,
                    a: torch.Tensor, b: torch.Tensor, max_len: int) -> torch.Tensor:
    """Content id of the character range [a, b) of document `doc`.

    prefix_hash_b: [B, T+1] int64; doc, a, b: int64 tensors of equal shape.
    Returns ids >= 0 for 0 <= a < b <= max_len, and -1 for empty or invalid ranges.
    """
    ok = b > a
    a_c = a.clamp(0, max_len)
    b_c = torch.maximum(b, a).clamp(0, max_len)
    ha = torch.remainder(prefix_hash_b[doc, a_c] * pow_tab[b_c - a_c], HASH_MOD)
    h = torch.remainder(prefix_hash_b[doc, b_c] - ha, HASH_MOD)
    ids = h * (max_len + 1) + (b_c - a_c)                 # length folded in; below 2**41
    return torch.where(ok, ids, torch.full_like(ids, -1))
