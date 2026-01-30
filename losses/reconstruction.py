import torch
import torch.nn.functional as F
from configs.default_config import Config
from utils.helper_functions import segment_cosine_similarity, segment_cosine_similarity_seq

def calculate_token_loss(
    original_tokens,
    encoded_sequences,
    is_real_position,
    token_embedding_layer
    ):
    B, L, D = encoded_sequences.shape
    V = token_embedding_layer.weight.shape[0]
    BL = B * L
    device = encoded_sequences.device
    
    E = encoded_sequences.reshape(BL, D)
    true_ids = original_tokens.reshape(BL)
    
    W_norm = F.normalize(token_embedding_layer.weight, p=2, dim=1)
    E_norm = F.normalize(E, p=2, dim=1)
    
    # Compute exact similarities for all tokens
    exact_sims = torch.matmul(E_norm, W_norm.t())  # (BL, V)
    
    sims = 2 * torch.atanh(torch.clamp(exact_sims, min=Config.EPS-1, max=1-Config.EPS))
    logits = sims
    
    loss = F.cross_entropy(logits, true_ids, reduction='none')
    loss = loss.reshape(B, L) * is_real_position
    
    return loss.sum() / is_real_position.sum()


def calculate_reconstruction_loss(
    denoised,
    is_real_inferred,
    target_sequences,
    splitter_existence_share,
    previous_denoised=None,
    sequence_weight=1,
    num_negatives=63
    ):
    batch, seq_len, hidden = denoised.shape
    device = denoised.device
    
    denoised_flat = denoised.reshape(-1, hidden)
    target_flat = target_sequences.reshape(-1, hidden)
    is_real_flat = is_real_inferred.reshape(-1)
    
    N = batch * seq_len
    
    if previous_denoised is not None:
        # Use sequence-level optimal t with importance weighting
        pos_sim, _, optimal_t = segment_cosine_similarity_seq(
            previous_denoised,      # (batch, seq_len, hidden)
            denoised,               # (batch, seq_len, hidden)
            target_sequences,       # (batch, seq_len, hidden)
            splitter_existence_share  # (batch, seq_len)
        )
        # Flatten to match downstream processing
        pos_sim = pos_sim.reshape(-1)  # (N,) where N = batch * seq_len
        
        previous_flat = previous_denoised.reshape(-1, hidden)
        
        sample_probs = is_real_flat.float()
        sample_probs = sample_probs / (sample_probs.sum() + Config.EPS)
        
        # Sample num_negatives indices for each position (with replacement)
        sampled_indices = torch.multinomial(
            sample_probs.unsqueeze(0).expand(N, -1),
            num_negatives,
            replacement=True
        )
        
        neg_previous = previous_flat[sampled_indices]
        neg_target = target_flat[sampled_indices]
        
        denoised_exp = denoised_flat.unsqueeze(1).expand(-1, num_negatives, -1).reshape(N * num_negatives, hidden)
        neg_previous_flat = neg_previous.reshape(N * num_negatives, hidden)
        neg_target_flat = neg_target.reshape(N * num_negatives, hidden)
        
        neg_sim_flat, _ = segment_cosine_similarity(
            neg_previous_flat.unsqueeze(0),  # <1, N*num_negatives, hidden>
            denoised_exp.unsqueeze(0),       # <1, N*num_negatives, hidden>
            neg_target_flat.unsqueeze(0)     # <1, N*num_negatives, hidden>
        )
        neg_sim = neg_sim_flat.squeeze(0).reshape(N, num_negatives)  # <N, num_negatives>
        
    else:
        # Regular cosine similarity (no segments)
        denoised_norm = F.normalize(denoised_flat, p=2, dim=-1)  # <N, hidden>
        target_norm = F.normalize(target_flat, p=2, dim=-1)  # <N, hidden>
        
        # Positive: element-wise
        pos_sim = torch.sum(denoised_norm * target_norm, dim=-1)  # <N>
        
        # Sample negatives
        sample_probs = is_real_flat.float()
        sample_probs = sample_probs / (sample_probs.sum() + Config.EPS)
        
        sampled_indices = torch.multinomial(
            sample_probs.unsqueeze(0).expand(N, -1),
            num_negatives,
            replacement=True
        )  # <N, num_negatives>
        
        neg_target_norm = target_norm[sampled_indices]
        
        # Compute negative similarities
        neg_sim = torch.sum(
            denoised_norm.unsqueeze(1) * neg_target_norm,
            dim=-1
        )  # <N, num_negatives>
    
    all_sim = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    
    logits = 2 * torch.atanh(torch.clamp(all_sim, min=Config.EPS-1, max=1-Config.EPS))
    
    labels = torch.zeros(N, dtype=torch.long, device=device)
    
    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    ce_loss = ce_loss.reshape(batch, seq_len)
    
    weighted_loss = ce_loss * splitter_existence_share
    weighted_loss = weighted_loss*sequence_weight
    total_loss = weighted_loss.sum() / (splitter_existence_share.sum() + Config.EPS)
    
    # Return optimal_t if it was computed (when previous_denoised exists)
    if previous_denoised is not None:
        return total_loss, optimal_t
    else:
        return total_loss, None
