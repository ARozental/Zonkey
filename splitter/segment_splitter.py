import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models.transformer as transformer
from models.transformer import TransformerEncoder, EfficientLocalAttention
from utils.helper_functions import inverse_sigmoid

from configs.default_config import Config

class LinearEncoder(nn.Module):
    def __init__(self, d_model, level):
        super().__init__()
        self.level = level
        self.cnn_features = 8
        
        # Local attention transformer
        self.transformer = TransformerEncoder(
            d_model=d_model,
            n_heads=Config.NUM_HEADS[level],
            d_ff=d_model * Config.FF_DIM_RATIO,
            max_seq_len=Config.MAX_DOC_LENGTHS[level],
            dropout=Config.DROPOUT,
            num_layers=1,
            window_size=Config.MAX_SEQ_LENGTHS[level]
        )
        
        # Multi-Scale Convolutional Encoder
        # We use multiple kernel sizes to capture both:
        # 1. Immediate local boundaries (punctuation, spacing) -> Small kernel
        # 2. Broader context (sentence length, recent splits) -> Large kernel derived from config
                
        
        self.conv_local = nn.Conv1d(d_model, self.cnn_features, kernel_size=3, padding=1)
        self.conv_mid = nn.Conv1d(d_model, self.cnn_features, kernel_size=5, padding=2)
        self.conv_long = nn.Conv1d(d_model, self.cnn_features, kernel_size=7, padding=3)
        self.conv_longer = nn.Conv1d(d_model, self.cnn_features, kernel_size=9, padding=4)


        # self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()
        
        # Mixer layer to combine branches
        # self.mixer = nn.Conv1d(d_model, d_model, kernel_size=1)
        
        self.proj = nn.Linear(d_model+4*self.cnn_features, 1)
        self.bias = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.initial_bias = nn.Parameter(inverse_sigmoid(torch.tensor(2/Config.MAX_SEQ_LENGTHS[self.level],device=Config.DEVICE)),requires_grad=False)
        self.sqrt_d = math.sqrt(Config.D_MODEL[level])
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        
        # Apply transformer with linear attention (mask-based)
        # Create existence mask (all True for now, can be modified if needed)
        existence_mask = torch.ones(x.shape[0], x.shape[1], device=x.device, dtype=torch.bool)
        x = self.transformer(x, existence_mask, is_linear=True)
        
        # Multi-scale block
        residual = x
        x_in = x.transpose(1, 2) # [B, D, L]
        
        # Parallel branches with 10 output channels each
        out_local = self.conv_local(x_in)   # [B, 10, L]
        out_mid = self.conv_mid(x_in)       # [B, 10, L]
        out_long = self.conv_long(x_in)     # [B, 10, L]
        out_longer = self.conv_longer(x_in) # [B, 10, L]
        
        # Concatenate all CNN features per position
        cnn_features = torch.cat([out_local, out_mid, out_long, out_longer], dim=1)  # [B, 40, L]
        cnn_features = self.activation(cnn_features)
        cnn_features_per_position = cnn_features.transpose(1, 2)  # [B, L, 40]
        
        # Concatenate CNN features to x
        x = torch.cat([x, cnn_features_per_position], dim=-1)  # [B, L, d_model + 40]
        
        x = self.proj(x)
        logits = self.initial_bias+(x+self.bias)/self.sqrt_d
        x = torch.sigmoid(logits)*(1-Config.EPS) + Config.EPS
        x = x.squeeze(-1)
        return x

class SegmentSplitter(nn.Module):
    def __init__(self, level, max_num_sentences=70):
        super().__init__()
        self.d_model = Config.D_MODEL[level]
        self.max_seq_len = Config.MAX_DOC_LENGTHS[level]
        self.max_sentence_length = Config.MAX_SEQ_LENGTHS[level]
        self.min_sentence_length = Config.COMPRESSION_VECTORS[level]
        self.max_num_sentences = max_num_sentences #per batch
        self.force_max_segments = False

        self.bos_classifier = LinearEncoder(self.d_model, level)

    def compute_mean_p_no_bos_patch(self,bos_probs, is_real_position, max_patch_length):
        """
        Computes the mean probability over specific starting positions that a patch of `patch_length` positions
        has no 'bos' selected. The starting positions must satisfy:
        - Position s >= 0
        - is_real_position[s] == 1
        - Number of real positions from s onwards >= patch_length + 1

        Given that is_real_position is a prefix of 1s followed by 0s starting with 1.

        Args:
            bos_probs (torch.Tensor): Tensor of shape (B, L) with probabilities 0 < p < 1.
            is_real_position (torch.Tensor): Float tensor of shape (B, L) with 0 or 1 indicating real positions.

        Returns:
            torch.Tensor: Scalar mean probability over all eligible starting positions
            in the entire batch.
        """
        # Force float32 for log1p+cumsum+exp to avoid float16 underflow
        log_x = torch.log1p(-bos_probs.float().clamp(max=1-Config.EPS))  # log(1 - p), shape (B, L)
        B, L = log_x.shape
        pl = max_patch_length #patch length
        
        valid_len = L - pl + 1
        # Assumed valid_len > 0
        
        prefix_log = torch.zeros(B, L + 1, dtype=log_x.dtype, device=log_x.device)
        prefix_log[:, 1:] = torch.cumsum(log_x, dim=1)
        
        log_num = prefix_log[:, pl : pl + valid_len]
        log_den = prefix_log[:, :valid_len]
        log_p = log_num - log_den
        p = torch.exp(log_p)  # (B, valid_len)
        
        # Compute sequence lengths
        seq_lens = torch.sum(is_real_position, dim=1)  # (B,)
        
        # Indices for starting positions
        device = bos_probs.device
        s_indices = torch.arange(valid_len, device=device)  # (valid_len,)
        
        # Number of real positions from s onwards: max(0, seq_len - s)
        sum_ge_s = (seq_lens[:, None] - s_indices[None, :]).clamp(min=0)  # (B, valid_len)
        
        # Mask for eligible starting positions
        mask = (s_indices[None, :] >= 0) & (s_indices[None, :] < seq_lens[:, None]) & (sum_ge_s >= pl + 1)
        
        # Compute mean per batch
        masked_p = p * mask.float()
        sum_selected = torch.sum(masked_p)
        counts = torch.sum(mask.float())
        mean_p = sum_selected / counts.clamp(min=1e-9)  # Avoid division by zero
        
        return mean_p
    
    def compute_mean_p_short_sentence(self, bos_probs, is_real_position, min_length):
        """
        Computes the mean probability over specific starting positions that a sentence
        is shorter than `min_length`.

        This is implemented as 1 - P(no BOS in the next `min_length - 1`
        positions), using the same sliding-window log-space product as
        `compute_mean_p_no_bos_patch`.

        Args:
            bos_probs (torch.Tensor): Tensor of shape (B, L) with probabilities 0 < p < 1.
            is_real_position (torch.Tensor): Float tensor of shape (B, L) with 0 or 1.

        Returns:
            torch.Tensor: Scalar mean probability over all eligible starting positions.
        """
        # Force float32 for log1p+cumsum+exp to avoid float16 underflow
        log_x = torch.log1p(-bos_probs.float().clamp(max=1-Config.EPS))  # log(1 - p), shape (B, L)
        B, L = log_x.shape

        # We look at the next (min_sentence_length - 1) positions *after* a start,
        # so the window length here is min_sentence_length - 1 and starts at s+1.
        pl = max(min_length - 1, 1)

        # For window [s+1, s+pl], we need s+pl <= L-1 -> s <= L-pl-1
        valid_len = L - pl
        if valid_len <= 0:
            return torch.zeros((), dtype=log_x.dtype, device=log_x.device)

        prefix_log = torch.zeros(B, L + 1, dtype=log_x.dtype, device=log_x.device)
        prefix_log[:, 1:] = torch.cumsum(log_x, dim=1)

        # Window [s+1, s+pl] -> prefix indices [s+1, s+pl+1)
        s_indices = torch.arange(valid_len, device=bos_probs.device)  # (valid_len,)
        start_pos = s_indices + 1
        end_pos = start_pos + pl

        log_num = prefix_log[:, end_pos]
        log_den = prefix_log[:, start_pos]
        log_p_no_bos = log_num - log_den
        p_no_bos = torch.exp(log_p_no_bos)  # (B, valid_len)

        # Probability that there *is* a BOS in the window (sentence shorter than min length)
        p_short = 1.0 - p_no_bos

        # Compute sequence lengths from is_real_position prefix
        seq_lens = torch.sum(is_real_position, dim=1)  # (B,)

        # Number of real positions from s onwards: max(0, seq_len - s)
        sum_ge_s = (seq_lens[:, None] - s_indices[None, :]).clamp(min=0)  # (B, valid_len)

        # Mask for eligible starting positions: s >= 0, within real prefix, and with
        # enough real tokens to cover the window plus at least one extra token
        mask = (s_indices[None, :] >= 0) & (s_indices[None, :] < seq_lens[:, None]) & (sum_ge_s >= pl + 1)
        
        # Weight by probability of starting a sentence at s
        # We only care if a sentence is short IF a sentence actually starts there
        start_probs = bos_probs[:, :valid_len]
        
        masked_p = p_short * start_probs * mask.float()
        sum_selected = torch.sum(masked_p)
        
        counts = torch.sum(start_probs * mask.float())
        mean_p_short = sum_selected / counts.clamp(min=1e-9)

        return mean_p_short
    

    def compute_is_bos_extra(self,is_bos_by_random, is_real_position):
        """
        Computes is_bos_extra <B, L> to add extra sentence starts using a greedy approach
        to ensure that there are no runs of more than `patch_length - 1` consecutive real positions
        without a sentence start in the final is_bos = is_bos_by_random | is_bos_extra.

        Assumes is_real_position is a prefix of 1s followed by 0s starting with 1 in each batch element,
        and that position 0 is already a BOS in is_bos_by_random where applicable.

        This implementation uses one Python loop over the batch dimension with vectorized operations inside,
        minimizes CPU/GPU syncs to one upfront transfer, and avoids all per-iteration syncs.

        Args:
            is_bos_by_random (torch.Tensor): Boolean tensor <B, L> of randomly selected starts.
            is_real_position (torch.Tensor): Float tensor <B, L> with 0 or 1 for real positions.
            patch_length (int): Maximum allowed patch length (default: 16).

        Returns:
            torch.Tensor: Boolean tensor <B, L> indicating extra starts added.
        """
        B, L = is_bos_by_random.shape
        device = is_bos_by_random.device
        dtype = is_bos_by_random.dtype
        is_bos_extra = torch.zeros(B, L, dtype=dtype, device=device)
        
        seq_lens = torch.sum(is_real_position, dim=1).long()
        seq_lens_cpu = seq_lens.cpu().tolist()
        
        pl = self.max_sentence_length
        pl1 = pl  # Adjusted to pl for max consecutive False = pl - 1
        fixed_max_k = (L // pl1) + 1  # Safe upper bound
        
        for b in range(B):
            sl = seq_lens_cpu[b]
            if sl == 0:
                continue
            
            random_bos = is_bos_by_random[b, :sl]
            
            # Bos positions within real prefix
            bos_pos = torch.nonzero(random_bos).flatten()
            
            # Extend with sl to handle last gap
            extended_pos = torch.cat([bos_pos, torch.tensor([sl], dtype=torch.long, device=device)])
            
            prevs = extended_pos[:-1]
            nexts = extended_pos[1:]
            
            dists = nexts - prevs
            
            # Compute number of extras per gap
            ks = (dists.float() / pl1).ceil().long() - 1
            ks = ks.clamp(min=0)
            
            # Generate add positions (mask handles if none needed)
            arange_k = torch.arange(1, fixed_max_k + 1, device=device)  # (fixed_max_k,)
            offsets = pl1 * arange_k[None, :]  # (1, fixed_max_k)
            prevs_rep = prevs[:, None]  # (num_gaps, 1)
            add_pos = prevs_rep + offsets  # (num_gaps, fixed_max_k)
            mask = arange_k[None, :] <= ks[:, None]  # (num_gaps, fixed_max_k)
            valid_add_pos = add_pos[mask]  # (total_adds,)
            # Filter out positions that would be out of bounds (beyond actual sequence length)
            valid_add_pos = valid_add_pos[valid_add_pos < sl]
            is_bos_extra[b, valid_add_pos] = True
        
        return is_bos_extra

    def forward(self, input_sequence, is_real_position, deterministic=False, input_tokens=None):
        batch_size, full_seq_len, _ = input_sequence.shape
        
        bos_probs = self.bos_classifier(input_sequence)   # [batch, full_seq_len]
        if self.force_max_segments:
            bos_probs = torch.ones_like(bos_probs)
        bos_probs[:, 0] = 1
        bos_probs = torch.clip(bos_probs, Config.EPS, 1 - Config.EPS)
        bos_probs = bos_probs * is_real_position
        
        device = input_sequence.device
        max_sentence_length = self.max_sentence_length
        
        # Random sampling for BOS
        if deterministic:
            my_random = torch.zeros_like(bos_probs) + 0.5
        else:
            my_random = torch.rand_like(bos_probs)
        my_random[:, 0] = 0  # Forces the first token to be BOS
        is_bos_by_random = my_random < bos_probs

        patch_loss = self.compute_mean_p_no_bos_patch(bos_probs, is_real_position, max_sentence_length)
        patch_loss = patch_loss + self.compute_mean_p_no_bos_patch(bos_probs, is_real_position, max_sentence_length+1)
        patch_loss = patch_loss + self.compute_mean_p_no_bos_patch(bos_probs, is_real_position, max_sentence_length+2)
        is_bos_extra = self.compute_is_bos_extra(is_bos_by_random, is_real_position)

        # Mean probability that a sentence is shorter than self.min_sentence_length
        short_sentence_loss = self.compute_mean_p_short_sentence(bos_probs, is_real_position, self.min_sentence_length)
        short_sentence_loss += self.compute_mean_p_short_sentence(bos_probs, is_real_position, self.min_sentence_length-1)
        short_sentence_loss += self.compute_mean_p_short_sentence(bos_probs, is_real_position, self.min_sentence_length-2)

        # Combine BOS markers
        attention_mask = is_real_position > 0.5
        is_bos = (is_bos_by_random | is_bos_extra) & attention_mask
        is_bos[:, -max_sentence_length:] = False
        is_bos[:, 0] = True  
        
        # Initialize outputs
        num_sentences_per_doc = torch.zeros(batch_size, dtype=torch.long, device=device)
        num_main_per_doc = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # PASS 1: Count sentences and plan
        # Calculate max sentences allowed per document
        max_sentences_per_doc_limit = max(1, self.max_num_sentences // 2)
        
        planned_docs = [] # List of tuples: (batch_index, bos_positions_subset)
        total_num_sentences = 0
        
        for b in range(batch_size):
            # Find BOS positions
            bos_positions = torch.nonzero(is_bos[b], as_tuple=False).view(-1)
            num_sent_b = len(bos_positions)
            
            if num_sent_b == 0:
                continue
                
            # Enforce per-document cap
            # Take the first K sentences as requested ("first ... of the first doc")
            num_to_take = min(num_sent_b, max_sentences_per_doc_limit)
            
            # Check global cap
            if total_num_sentences + num_to_take > self.max_num_sentences:
                # If we can't fit the full truncated doc, we stop completely
                # as per user request: "as we have reached max_docs, we do not continue"
                break
                
            planned_docs.append((b, bos_positions[:num_to_take]))
            num_sentences_per_doc[b] = num_to_take
            num_main_per_doc[b] = num_to_take
            total_num_sentences += num_to_take

        # Pre-allocate output tensors
        # We avoid torch.cat by allocating exactly what we need
        if total_num_sentences > 0:
            all_sentence_vectors = torch.empty(total_num_sentences, max_sentence_length, self.d_model, device=device, dtype=input_sequence.dtype)
            all_sentence_bos_probs = torch.empty(total_num_sentences, max_sentence_length, device=device, dtype=bos_probs.dtype)
            all_sentence_is_real = torch.empty(total_num_sentences, max_sentence_length, device=device, dtype=is_real_position.dtype)
            all_p_exist_share = torch.empty(total_num_sentences, max_sentence_length, device=device, dtype=bos_probs.dtype)
            all_bos_starts = torch.empty(total_num_sentences, dtype=torch.long, device=device)
            original_position = torch.empty(total_num_sentences, max_sentence_length, 2, dtype=torch.long, device=device)
            
            if input_tokens is not None:
                all_tokens_out = torch.empty(total_num_sentences, max_sentence_length, dtype=torch.long, device=device)
            else:
                all_tokens_out = None
        else:
            # Handle empty case
            all_sentence_vectors = torch.empty(0, max_sentence_length, self.d_model, device=device)
            all_sentence_bos_probs = torch.empty(0, max_sentence_length, device=device)
            all_sentence_is_real = torch.empty(0, max_sentence_length, device=device)
            all_p_exist_share = torch.empty(0, max_sentence_length, device=device)
            all_bos_starts = torch.empty(0, dtype=torch.long, device=device)
            all_tokens_out = torch.empty(0, max_sentence_length, dtype=torch.long, device=device) if input_tokens is not None else None
            original_position = torch.empty(0, max_sentence_length, 2, dtype=torch.long, device=device)
            bos_per_position = torch.tensor(0.0, device=device)

        # PASS 2: Fill tensors
        current_idx = 0
        for b, bos_positions in planned_docs:
            num_sent_b = len(bos_positions)
            
            # Extract sentences using advanced indexing
            local_indices = torch.arange(max_sentence_length, device=device)
            global_positions = bos_positions.unsqueeze(1) + local_indices  # [num_sent, max_len]
            valid = global_positions < full_seq_len
            pos_clamped = global_positions.clamp(0, full_seq_len - 1)
            
            # Slice for direct assignment
            idx_slice = slice(current_idx, current_idx + num_sent_b)
            
            # Fill pre-allocated tensors directly
            all_sentence_vectors[idx_slice] = input_sequence[b][pos_clamped, :]
            
            # Use temporary variables to avoid in-place modification of tensors used in gradients
            temp_bos_probs = bos_probs[b][pos_clamped]
            temp_is_real = is_real_position[b][pos_clamped]
            
            if input_tokens is not None:
                sentence_tokens = input_tokens[b][pos_clamped]
                sentence_tokens = sentence_tokens.masked_fill(~valid, 0)
                all_tokens_out[idx_slice] = sentence_tokens
            
            # Calculate p_exist in log space using temp variables
            # Force float32 for log1p+cumsum+exp to avoid float16 underflow
            log_one_minus_probs = torch.log1p(-temp_bos_probs.float().clamp(max=0.9999, min=0.0001))
            log_one_minus_probs[:, 0] = 0
            log_p_exist = torch.cumsum(log_one_minus_probs, dim=1)
            p_exist = torch.exp(log_p_exist)
            p_exist = p_exist.masked_fill(~valid, 0)
            
            # Vectorized p_exist_share
            p_exist_sum = torch.zeros(full_seq_len, device=device, dtype=p_exist.dtype)
            flat_pos = global_positions.view(-1)
            flat_p = p_exist.view(-1)
            flat_valid = valid.view(-1)
            
            # This uses atomic adds, safe for overlapping positions
            p_exist_sum.index_add_(0, flat_pos[flat_valid], flat_p[flat_valid])
            
            p_exist_totals = p_exist_sum[pos_clamped]
            p_exist_share = p_exist / (p_exist_totals + 1e-10)
            p_exist_share = p_exist_share.masked_fill(~valid, 0)
            
            all_p_exist_share[idx_slice] = p_exist_share
            
            # Apply masks out-of-place and then assign to the final tensor
            # This avoids modifying tensors that might be needed for backward pass
            all_sentence_bos_probs[idx_slice] = temp_bos_probs.masked_fill(~valid, 0)
            all_sentence_is_real[idx_slice] = temp_is_real.masked_fill(~valid, 0)
            
            # Create original_position tensor
            batch_ids = torch.full((num_sent_b, max_sentence_length), b, dtype=torch.long, device=device)
            # stack creates new tensor, then we assign
            original_position[idx_slice] = torch.stack([batch_ids, global_positions], dim=2)
            
            all_bos_starts[idx_slice] = bos_positions
            
            current_idx += num_sent_b

        if total_num_sentences > 0:
            # bos_per_position = all_sentence_bos_probs.sum() / all_sentence_is_real.sum()
            bos_per_position = (bos_probs[:,1:] * is_real_position[:,1:]).sum() / is_real_position[:,1:].sum()
        
        all_p_exist_weight = all_p_exist_share * all_sentence_is_real
        # num_main_total = total_num_sentences
        # num_alt_total = 0

        
        return (
            all_sentence_vectors,
            all_sentence_bos_probs,
            all_sentence_is_real,
            all_p_exist_weight,
            bos_per_position,
            num_sentences_per_doc,
            all_tokens_out,
            original_position,
            patch_loss,
            short_sentence_loss,
        )

