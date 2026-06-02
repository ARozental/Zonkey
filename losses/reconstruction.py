import torch
import torch.nn.functional as F
from configs.default_config import Config
from utils.helper_functions import arc_cosine_similarity_seq


def calculate_token_loss(
    original_tokens,
    encoded_sequences,
    splitter_existence_share,
    token_embedding_layer,
    previous_denoised=None,
    full_reconstruction_ratio=0.85,
):
    B, L, D = encoded_sequences.shape
    V = token_embedding_layer.weight.shape[0]
    BL = B * L
    device = encoded_sequences.device

    W = token_embedding_layer.weight  # (V, D)
    true_ids = original_tokens.reshape(BL)

    if previous_denoised is not None:
        # Get target embeddings for optimal_t computation
        target_embeddings = token_embedding_layer.weight[original_tokens]  # (B, L, D)

        # Sequence-level optimal t (shared across whole sequence)
        pos_sim, _, optimal_t = arc_cosine_similarity_seq(
            previous_denoised,
            encoded_sequences,
            target_embeddings,
            splitter_existence_share
        )

        # === SEGMENT LOSS (full-vocab using shared optimal_t) ===
        t = optimal_t.view(B, 1, 1)
        one_minus_t = 1.0 - t

        E_norm = F.normalize(encoded_sequences, p=2, dim=-1)

        E_dot_prev = (E_norm * previous_denoised).sum(-1, keepdim=True)
        E_dot_W = torch.matmul(E_norm, W.t())

        numer = one_minus_t * E_dot_prev + t * E_dot_W

        prev_norm_sq = (previous_denoised * previous_denoised).sum(-1, keepdim=True)
        prev_dot_W = torch.matmul(previous_denoised, W.t())
        W_norm_sq = (W * W).sum(-1)

        denom_sq = (one_minus_t ** 2) * prev_norm_sq + 2 * one_minus_t * t * prev_dot_W + (t ** 2) * W_norm_sq
        denom = torch.sqrt(denom_sq.clamp(min=Config.EPS ** 2))

        all_sims = numer / denom

        logits = 2 * torch.atanh(torch.clamp(all_sims, min=Config.EPS-1, max=1-Config.EPS))
        logits = logits.reshape(BL, V)

        segment_loss = F.cross_entropy(logits, true_ids, reduction='none')
        segment_loss = (segment_loss.reshape(B, L) * splitter_existence_share).sum() / (splitter_existence_share.sum() + Config.EPS)

        # === DIRECT FULL RECONSTRUCTION LOSS (no segment) ===
        E = encoded_sequences.reshape(BL, D)
        W_norm = F.normalize(W, p=2, dim=1)
        E_norm = F.normalize(E, p=2, dim=1)
        exact_sims = torch.matmul(E_norm, W_norm.t())
        direct_sims = 2 * torch.atanh(torch.clamp(exact_sims, min=Config.EPS-1, max=1-Config.EPS))
        direct_loss = F.cross_entropy(direct_sims, true_ids, reduction='none')
        direct_loss = (direct_loss.reshape(B, L) * splitter_existence_share).sum() / (splitter_existence_share.sum() + Config.EPS)

        # Always weighted average (when ratio=0 → pure segment, when ratio=1 → pure direct)
        total_loss = (1.0 - full_reconstruction_ratio) * segment_loss + full_reconstruction_ratio * direct_loss

        return total_loss, optimal_t

    else:
        # Clean full-vocab path (no previous_denoised) - unchanged
        E = encoded_sequences.reshape(BL, D)
        W_norm = F.normalize(W, p=2, dim=1)
        E_norm = F.normalize(E, p=2, dim=1)
        exact_sims = torch.matmul(E_norm, W_norm.t())
        sims = 2 * torch.atanh(torch.clamp(exact_sims, min=Config.EPS-1, max=1-Config.EPS))
        logits = sims
        loss = F.cross_entropy(logits, true_ids, reduction='none')
        loss = (loss.reshape(B, L) * splitter_existence_share).sum() / (splitter_existence_share.sum() + Config.EPS)
        return loss, None


def calculate_reconstruction_loss(
    denoised,
    is_real_inferred,
    target_sequences,
    splitter_existence_share,
    previous_denoised=None,
    sequence_weight=1,
    num_negatives=63,
    fake_negatives=None,
    full_reconstruction_ratio=0.85,
):
    batch, seq_len, hidden = denoised.shape
    device = denoised.device
    
    denoised_flat = denoised.reshape(-1, hidden)
    target_flat = target_sequences.reshape(-1, hidden)
    is_real_flat = is_real_inferred.reshape(-1)
    
    N = batch * seq_len
    
    if N <= 1:
        return torch.tensor(0.0, device=device, requires_grad=True), None
    
    if previous_denoised is not None:
        # === SEGMENT LOSS using sequence-level optimal_t for BOTH positive AND negatives ===
        pos_sim, _, optimal_t = arc_cosine_similarity_seq(
            previous_denoised,
            denoised,
            target_sequences,
            splitter_existence_share
        )
        pos_sim = pos_sim.reshape(-1)  # (N,)

        previous_flat = previous_denoised.reshape(-1, hidden)
        
        # Sample negatives
        sample_probs = is_real_flat.float() / (is_real_flat.float().sum() + Config.EPS)
        sampled_indices = torch.multinomial(
            sample_probs, N * num_negatives, replacement=True
        ).reshape(N, num_negatives)
        
        # Fix self-collisions
        self_indices = torch.arange(N, device=device).unsqueeze(1)
        collision_mask = (sampled_indices == self_indices)
        while collision_mask.any():
            num_collisions = collision_mask.sum().item()
            sampled_indices[collision_mask] = torch.multinomial(sample_probs, num_collisions, replacement=True)
            collision_mask = (sampled_indices == self_indices)
        
        # --- Memory-efficient dot products via matmul+gather ---
        # Instead of gathering (N, num_neg, hidden) tensors for neg_previous and neg_target,
        # precompute per-position scalars and use matmul for denoised dot products.
        denoised_norm_2d = F.normalize(denoised_flat, p=2, dim=-1)  # (N, hidden)

        # Per-position scalars (all shape (N_total,))
        prev_norms_sq_all = (previous_flat * previous_flat).sum(-1)  # ||prev_i||^2
        targ_norms_sq_all = (target_flat * target_flat).sum(-1)      # ||targ_i||^2
        prev_dot_targ_all = (previous_flat * target_flat).sum(-1)    # prev_i . targ_i

        # Denoised dot products via matmul + gather (avoids (N, num_neg, hidden) tensors)
        d_sim_prev_full = torch.matmul(denoised_norm_2d, previous_flat.T)  # (N, N)
        d_sim_targ_full = torch.matmul(denoised_norm_2d, target_flat.T)    # (N, N)

        d_dot_prev = torch.gather(d_sim_prev_full, 1, sampled_indices)     # (N, num_neg)
        d_dot_targ = torch.gather(d_sim_targ_full, 1, sampled_indices)     # (N, num_neg)
        del d_sim_prev_full, d_sim_targ_full

        # Gather scalar properties at sampled positions
        prev_norm_sq = prev_norms_sq_all[sampled_indices]    # (N, num_neg)
        targ_norm_sq = targ_norms_sq_all[sampled_indices]    # (N, num_neg)
        prev_dot_targ = prev_dot_targ_all[sampled_indices]   # (N, num_neg)

        # Append fake negatives
        if fake_negatives is not None and fake_negatives.shape[0] > 0:
            K = fake_negatives.shape[0]
            # For fakes: previous=fake, target=fake, so prev_dot_targ=||fake||^2, norms same
            fake_d_dot = torch.matmul(denoised_norm_2d, fake_negatives.T)  # (N, K)
            fake_norm_sq = (fake_negatives * fake_negatives).sum(-1)       # (K,)
            d_dot_prev = torch.cat([d_dot_prev, fake_d_dot], dim=1)
            d_dot_targ = torch.cat([d_dot_targ, fake_d_dot], dim=1)
            prev_norm_sq = torch.cat([prev_norm_sq, fake_norm_sq.unsqueeze(0).expand(N, -1)], dim=1)
            targ_norm_sq = torch.cat([targ_norm_sq, fake_norm_sq.unsqueeze(0).expand(N, -1)], dim=1)
            prev_dot_targ = torch.cat([prev_dot_targ, fake_norm_sq.unsqueeze(0).expand(N, -1)], dim=1)

        # Shared sequence-level optimal_t for all negatives
        t = optimal_t.view(batch, 1).repeat_interleave(seq_len, dim=0)  # (N, 1)
        one_minus_t = 1.0 - t

        numer = one_minus_t * d_dot_prev + t * d_dot_targ
        denom_sq = (one_minus_t ** 2) * prev_norm_sq + 2 * one_minus_t * t * prev_dot_targ + (t ** 2) * targ_norm_sq
        denom = torch.sqrt(denom_sq.clamp(min=Config.EPS ** 2))

        neg_sim = numer / denom  # (N, total_neg)

        # Segment-based contrastive loss
        all_sim = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        logits = 2 * torch.atanh(torch.clamp(all_sim, min=Config.EPS-1, max=1-Config.EPS))
        ce_loss = F.cross_entropy(logits, torch.zeros(N, dtype=torch.long, device=device), reduction='none')
        ce_loss = ce_loss.reshape(batch, seq_len)
        weighted_loss = ce_loss * splitter_existence_share * sequence_weight
        segment_loss = weighted_loss.sum() / (splitter_existence_share.sum() + Config.EPS)

        # === DIRECT (non-segment) contrastive loss ===
        # Only the positive changes to direct cosine; negatives reuse segment-interpolated neg_sim
        target_norm = F.normalize(target_flat, p=2, dim=-1)
        direct_pos_sim = torch.sum(denoised_norm_2d * target_norm, dim=-1)
        direct_all_sim = torch.cat([direct_pos_sim.unsqueeze(1), neg_sim], dim=1)
        direct_logits = 2 * torch.atanh(torch.clamp(direct_all_sim, min=Config.EPS-1, max=1-Config.EPS))
        direct_ce = F.cross_entropy(direct_logits, torch.zeros(N, dtype=torch.long, device=device), reduction='none')
        direct_ce = direct_ce.reshape(batch, seq_len)
        direct_weighted = direct_ce * splitter_existence_share * sequence_weight
        direct_loss = direct_weighted.sum() / (splitter_existence_share.sum() + Config.EPS)

        # Always weighted average (ratio=0 → pure segment with shared t)
        total_loss = (1.0 - full_reconstruction_ratio) * segment_loss + full_reconstruction_ratio * direct_loss

        return total_loss, optimal_t

    else:
        # No previous_denoised → regular contrastive (unchanged)
        denoised_norm = F.normalize(denoised_flat, p=2, dim=-1)
        target_norm = F.normalize(target_flat, p=2, dim=-1)
        
        pos_sim = torch.sum(denoised_norm * target_norm, dim=-1)
        
        sample_probs = is_real_flat.float() / (is_real_flat.float().sum() + Config.EPS)
        sampled_indices = torch.multinomial(
            sample_probs, N * num_negatives, replacement=True
        ).reshape(N, num_negatives)
        
        self_indices = torch.arange(N, device=device).unsqueeze(1)
        collision_mask = (sampled_indices == self_indices)
        while collision_mask.any():
            num_collisions = collision_mask.sum().item()
            sampled_indices[collision_mask] = torch.multinomial(sample_probs, num_collisions, replacement=True)
            collision_mask = (sampled_indices == self_indices)
        
        # Compute full similarity matrix: (N, N) — much smaller than
        # gathering (N, num_neg, hidden) and doing elementwise multiply+sum.
        full_sim = torch.matmul(denoised_norm, target_norm.T)
        neg_sim = torch.gather(full_sim, 1, sampled_indices)
        del full_sim
        
        if fake_negatives is not None and fake_negatives.shape[0] > 0:
            fake_neg_norm = F.normalize(fake_negatives, p=2, dim=-1)
            fake_sim = torch.matmul(denoised_norm, fake_neg_norm.T)  # (N, K)
            neg_sim = torch.cat([neg_sim, fake_sim], dim=1)
        
        all_sim = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        logits = 2 * torch.atanh(torch.clamp(all_sim, min=Config.EPS-1, max=1-Config.EPS))
        
        ce_loss = F.cross_entropy(logits, torch.zeros(N, dtype=torch.long, device=device), reduction='none')
        ce_loss = ce_loss.reshape(batch, seq_len)
        weighted_loss = ce_loss * splitter_existence_share * sequence_weight
        total_loss = weighted_loss.sum() / (splitter_existence_share.sum() + Config.EPS)
        
        return total_loss, None