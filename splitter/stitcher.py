import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.transformer import TransformerEncoderLayer
from configs.default_config import Config
from losses.reconstruction import calculate_reconstruction_loss, calculate_token_loss
from utils.helper_functions import expected_l2_norm
"""
This is a module that stitches together the denoised sequences into a single sequence.
It takes denoised_input_sequences <sentences_batch, sentence_len, d_model> and returns <doc_batch, doc_len, d_model>
It also computes a loss IF in training mode (if labels are available)
"""

class Stitcher(nn.Module):
    def __init__(self,level=0,token_embedding_layer=None):
        super().__init__()
        self.level = level
        self.max_doc_len = Config.MAX_DOC_LENGTHS[level]
        self.max_seq_len = Config.MAX_SEQ_LENGTHS[level]
        self.d_model = Config.D_MODEL[level]
        self.max_sequences = Config.MAX_SEQUENCES_PER_BATCH[level]
        
        self.score_linear = nn.Linear(2 * self.d_model + 2, 1)
        self.correction_linear = nn.Linear(self.d_model, self.d_model)
        self.refiner = TransformerEncoderLayer(d_model = self.d_model,n_heads = Config.NUM_HEADS[level],d_ff = self.d_model*Config.FF_DIM_RATIO)
        self.refiner_portion = nn.Parameter(torch.tensor([0.1],device=Config.DEVICE,requires_grad=True))
        self.proj = nn.Linear(self.d_model, self.d_model//4)
        self.similarity_weight = nn.Parameter(torch.tensor([1.0],device=Config.DEVICE,requires_grad=True))
        self.pos_elu = lambda x: F.elu(x-1) + 1
        self.token_embedding_layer = token_embedding_layer
        self.dim_norm = expected_l2_norm(self.d_model)
        
        self.register_buffer('_output_buffer', torch.zeros(self.max_sequences, self.max_doc_len, self.d_model))

    def infer_start_position(
        self,
        seq1: torch.Tensor,
        seq2: torch.Tensor,
        seq1_exist_probs: torch.Tensor,
        seq2_exist_probs: torch.Tensor,
        similarity_weight: float = 1.0
    ) -> torch.Tensor:
        """
        Infer the start position (float) where seq2[0] aligns with seq1[N], or 64 for no overlap.
        Fully vectorized, differentiable for backprop.
        Now supports batch > 1.
        Returns: [batch] tensor of inferred positions
        """
        batch_size = seq1.size(0)
        seq_len = seq1.size(1)
        
        # Project sequences: [batch, seq_len, d_model] -> [batch, seq_len, d_model//4]
        seq1_proj = self.proj(seq1)
        seq2_proj = self.proj(seq2)
        
        # Compute similarity: [batch, seq_len, seq_len]
        sim = torch.bmm(seq2_proj, seq1_proj.transpose(1, 2))
        
        # Compute p_start from differences: [batch, seq_len+1]
        p_start = torch.zeros(batch_size, seq_len + 1, device=seq1.device)
        p_start[:, 1:seq_len] = seq1_exist_probs[:, :-1] - seq1_exist_probs[:, 1:]
        p_start = torch.clamp(p_start, min=0.0)
        p_start[:, seq_len] = seq1_exist_probs[:, -1]
        
        # Vectorized match_scores for n=0 to (seq_len-1): [batch, seq_len]
        M = 4
        j = torch.arange(M, device=seq1.device).view(M, 1, 1).expand(M, batch_size, seq_len)
        n_offsets = torch.arange(seq_len, device=seq1.device).view(1, 1, seq_len).expand(M, batch_size, seq_len)
        q = j
        k = n_offsets + j
        mask = k < seq_len
        
        # Gather scores using advanced indexing
        batch_idx = torch.arange(batch_size, device=seq1.device).view(1, batch_size, 1).expand(M, batch_size, seq_len)
        scores = torch.zeros(M, batch_size, seq_len, dtype=sim.dtype, device=seq1.device)
        scores[mask] = sim[batch_idx[mask], q[mask], k[mask]]
        
        counts = mask.float().sum(dim=0).clamp(min=1.0)  # [batch, seq_len]
        match_scores = scores.sum(dim=0) / counts  # [batch, seq_len]
        
        # Adjust p_start for overlap positions (exclude 64)
        p_start_overlap = p_start[:, :seq_len]  # [batch, seq_len]
        log_p_start_overlap = torch.log(p_start_overlap.clamp(min=1e-10))
        # Handle similarity_weight as tensor (Parameter) or scalar
        if isinstance(similarity_weight, torch.Tensor):
            sim_weight = self.pos_elu(similarity_weight.squeeze())
        else:
            sim_weight = self.pos_elu(torch.tensor(similarity_weight, device=seq1.device))
        log_adjusted = log_p_start_overlap + sim_weight * match_scores
        probs = torch.softmax(log_adjusted, dim=1)  # [batch, seq_len]
        
        positions_overlap = torch.arange(0, seq_len, dtype=torch.float, device=seq1.device).unsqueeze(0)  # [1, seq_len]
        soft_overlap_n = (probs * positions_overlap).sum(dim=1)  # [batch]
        
        prob_overlap = 1 - seq1_exist_probs[:, -1]  # [batch]
        
        # Inferred N: [batch]
        inferred_n = prob_overlap * soft_overlap_n + (1 - prob_overlap) * seq_len
        return inferred_n

    def to_original1(self, denoised_input_sequence1, denoised_is_real_position1, denoised_input_sequence2, denoised_is_real_position2):
        res = denoised_input_sequence1 + self.refiner_portion * self.refiner.cross_encode(denoised_input_sequence1, denoised_input_sequence2, denoised_is_real_position1, denoised_is_real_position2)
        res = F.normalize(res,p=2,dim=-1) * self.dim_norm
        return res

    def get_position_loss(self, denoised2_start_position, inferred_denoised2_start_position, p_exist_share1):
        """
        Your original weighted interval loss + per-sentence normalization (your idea 1).
        Sum before true start = 1, sum after true start = 1, then scale back by total A.
        """
        batch_size = p_exist_share1.shape[0]
        seq_len = p_exist_share1.shape[1]
        
        true_pos = denoised2_start_position.float().view(batch_size)
        inf_pos = inferred_denoised2_start_position.float().view(batch_size)
        
        # A = total importance per sentence
        A = p_exist_share1.sum(dim=1, keepdim=True)
        
        # Masks for before and after the true start position
        indices = torch.arange(seq_len, dtype=torch.float32, device=p_exist_share1.device).unsqueeze(0).expand(batch_size, -1)
        
        mask_before = (indices < true_pos.unsqueeze(1)).float()
        mask_after  = (indices >= true_pos.unsqueeze(1)).float()
        
        # Normalize before and after separately to sum to 1
        sum_before = (p_exist_share1 * mask_before).sum(dim=1, keepdim=True) + Config.EPS
        sum_after  = (p_exist_share1 * mask_after ).sum(dim=1, keepdim=True) + Config.EPS
        
        normalized_p = p_exist_share1.clone()
        normalized_p = normalized_p * mask_before / sum_before + normalized_p * mask_after / sum_after
        
        # Scale back by total A
        normalized_p = normalized_p * A
        
        # Now compute the error interval with the normalized weights (same logic as your original)
        min_pos = torch.min(true_pos, inf_pos)
        max_pos = torch.max(true_pos, inf_pos)
        
        weights = torch.zeros_like(indices)
        
        same_bucket = (torch.floor(min_pos).unsqueeze(1) == torch.floor(max_pos).unsqueeze(1))
        
        same_bucket_mask = same_bucket & (indices == torch.floor(min_pos).unsqueeze(1))
        weights = torch.where(same_bucket_mask, (max_pos - min_pos).unsqueeze(1), weights)
        
        mask_full = ~same_bucket & (indices > torch.floor(min_pos).unsqueeze(1)) & (indices < torch.floor(max_pos).unsqueeze(1))
        weights = torch.where(mask_full, torch.ones_like(weights), weights)
        
        min_bucket_mask = ~same_bucket & (indices == torch.floor(min_pos).unsqueeze(1))
        weights = torch.where(min_bucket_mask, 1.0 - (min_pos.unsqueeze(1) - torch.floor(min_pos).unsqueeze(1)), weights)
        
        max_bucket_mask = ~same_bucket & (indices == torch.floor(max_pos).unsqueeze(1))
        weights = torch.where(max_bucket_mask, max_pos.unsqueeze(1) - torch.floor(max_pos).unsqueeze(1), weights)
        
        position_sum = (normalized_p * weights).sum(dim=1)
        interval_loss = position_sum / (A.squeeze(1) + Config.EPS)
        
        return interval_loss
    
    def get_sequence_loss(self, inferred_input_sequence1, original_input_sequence1, p_exist_share1, all_tokens=None, is_real_inferred1=None, previous_denoised1=None):
        """
        This works at a single doc level now. 
        """
        
        if self.level > 0:
            loss,_ = calculate_reconstruction_loss(
                denoised=inferred_input_sequence1, 
                is_real_inferred=is_real_inferred1,
                target_sequences=original_input_sequence1, 
                splitter_existence_share=p_exist_share1, 
                previous_denoised=previous_denoised1, 
                num_negatives=63
            )
        else:
            loss, _ = calculate_token_loss(all_tokens, inferred_input_sequence1, p_exist_share1, self.token_embedding_layer)
        
        return loss


    def merge_2_sequences(
        self, 
        denoised_input_sequence1, 
        denoised_is_real_position1, 
        denoised_input_sequence2, 
        denoised_is_real_position2, 
        denoised2_start_position=None, 
        original_input_sequence1=None, 
        p_exist_share1=None,
        all_tokens=None,
        previous_denoised1=None
        ):
        """
        input_sequences are of shape [batch, seq_len, d_model]
        is_real_position is of shape [batch, seq_len]
        denoised2_start_position is the position where the second denoised sequence starts, it is always <= seq_len and there is usually an overlap
        original_input_sequence1 is the target sequence
        p_exist_share1 [batch, seq_len] will be used to determine how important each position is for the loss function
        
        Now supports batch > 1 for vectorized processing.
        """
        batch_size = denoised_input_sequence1.shape[0]
        inferred_denoised2_start_position = self.infer_start_position(denoised_input_sequence1, denoised_input_sequence2, denoised_is_real_position1, denoised_is_real_position2,self.similarity_weight)
        inferred_input_sequence1 = self.to_original1(denoised_input_sequence1, denoised_is_real_position1, denoised_input_sequence2, denoised_is_real_position2)
        if denoised2_start_position is None:
            k = inferred_denoised2_start_position.round().clamp(0, self.max_seq_len).long()
            return inferred_input_sequence1, torch.zeros(batch_size, device=inferred_input_sequence1.device), torch.zeros(1, device=inferred_input_sequence1.device).sum(), k
        
        k = denoised2_start_position.round().clamp(0, self.max_seq_len).long().view(batch_size)
        position_loss = self.get_position_loss(denoised2_start_position, inferred_denoised2_start_position, p_exist_share1)
        sequence_loss = self.get_sequence_loss(inferred_input_sequence1, original_input_sequence1, p_exist_share1, all_tokens=all_tokens, is_real_inferred1=denoised_is_real_position1, previous_denoised1=previous_denoised1)

        
       
        return inferred_input_sequence1, position_loss, sequence_loss, k

    def forward(
        self, 
        denoised_input_sequences, 
        denoised_is_real_position,
        num_seq_per_doc, 
        original_position=None, 
        original_input_sequences=None, 
        all_p_exist_share=None, 
        all_tokens=None,
        previous_denoised=None):

        # Extract start_positions from original_position if provided
        # original_position shape: [num_sequences, seq_len, 2] where [:,:,0]=doc_id, [:,:,1]=position_in_doc
        start_positions = None
        if original_position is not None:
            # Extract start position from the first position of each sequence
            start_positions = original_position[:, 0, 1]  # [num_sequences]

        # Clone denoised_input_sequences to avoid graph conflicts
        denoised_input_sequences = denoised_input_sequences.clone()
        
        # Initialize losses
        total_position_loss = torch.tensor(0.0, device=denoised_input_sequences.device, dtype=denoised_input_sequences.dtype)
        total_sequence_loss = torch.tensor(0.0, device=denoised_input_sequences.device, dtype=denoised_input_sequences.dtype)
        
        # Filter documents with num_seq > 0
        mask = num_seq_per_doc > 0
        num_seq_per_doc_filtered = num_seq_per_doc[mask]
        num_docs = mask.sum().item()
        
        # Output buffer handling
        d_model = denoised_input_sequences.shape[2]
        seq_len = denoised_input_sequences.shape[1]
        
        if num_docs <= self.max_sequences:
            all_patches = self._output_buffer[:num_docs, :self.max_doc_len, :d_model].detach()
            all_patches.fill_(0)
        else:
            all_patches = torch.zeros(num_docs, self.max_doc_len, d_model, 
                                     device=denoised_input_sequences.device, dtype=denoised_input_sequences.dtype)
        
        stitched_len = torch.full((num_docs,), seq_len, device=denoised_input_sequences.device, dtype=torch.long)

        # Precompute document start indices in the flat input
        doc_starts = torch.cat([torch.tensor([0], device=num_seq_per_doc_filtered.device), 
                                num_seq_per_doc_filtered.cumsum(0)[:-1]])
        
        # Identify single vs multi sentence docs
        single_sent_mask = num_seq_per_doc_filtered == 1
        multi_sent_mask = ~single_sent_mask
        
        # --- 1. Handle Single Sentence Documents ---
        if single_sent_mask.any():
            single_doc_indices = torch.nonzero(single_sent_mask).squeeze(1)
            flat_indices = doc_starts[single_doc_indices]
            single_docs = denoised_input_sequences[flat_indices]
            padded_len = min(self.max_seq_len, self.max_doc_len)
            all_patches[single_doc_indices, :padded_len] = single_docs[:, :padded_len]
            
        # --- 2. Handle Multi Sentence Documents ---
        if multi_sent_mask.any():
            multi_doc_indices = torch.nonzero(multi_sent_mask).squeeze(1)
            multi_doc_starts = doc_starts[multi_doc_indices]
            multi_doc_counts = num_seq_per_doc_filtered[multi_doc_indices]
            
            # Generate indices for consecutive pairs (i, i+1) within each document
            pair_counts = multi_doc_counts - 1
            
            # Valid ranges: for each doc, sentences from start to start+count-2 are "left" sentences
            range_starts = multi_doc_starts
            range_ends = multi_doc_starts + multi_doc_counts - 1 
            
            # Generate flattened indices for "left" side of all pairs
            left_indices_list = [torch.arange(s, e, device=denoised_input_sequences.device) 
                               for s, e in zip(range_starts, range_ends)]
            left_indices = torch.cat(left_indices_list)
            right_indices = left_indices + 1
            
            # Gather Batches
            seq1_batch = denoised_input_sequences[left_indices]
            seq2_batch = denoised_input_sequences[right_indices]
            is_real1_batch = denoised_is_real_position[left_indices]
            is_real2_batch = denoised_is_real_position[right_indices]
            
            # Optional Inputs
            delta_batch = None
            if start_positions is not None:
                starts_i = start_positions[left_indices].float()
                starts_i_plus_1 = start_positions[right_indices].float()
                delta_batch = starts_i_plus_1 - starts_i
            
            original_batch = original_input_sequences[left_indices] if original_input_sequences is not None else None
            p_exist_batch = all_p_exist_share[left_indices] if all_p_exist_share is not None else None
            tokens_batch = all_tokens[left_indices] if all_tokens is not None else None
            previous_denoised1_batch = previous_denoised[left_indices] if previous_denoised is not None else None
            
            # Call Merge
            inferred_batch, pos_losses, seq_losses, k_batch = self.merge_2_sequences(
                seq1_batch, is_real1_batch, seq2_batch, is_real2_batch,
                delta_batch, original_batch, p_exist_batch, all_tokens=tokens_batch,
                previous_denoised1=previous_denoised1_batch
            )
            
            # Accumulate Losses
            total_position_loss = pos_losses.sum()
            if seq_losses.numel() > 1:
                total_sequence_loss = seq_losses.sum()
            else:
                total_sequence_loss = seq_losses
            
            # --- Reconstruct Output ---
            current_pair_idx = 0
            num_multi_docs = len(multi_doc_indices)
            
            # Iterate over unique multi-sentence docs to stitch them back
            for i in range(num_multi_docs):
                doc_idx = multi_doc_indices[i].item()
                count = pair_counts[i].item()
                
                # Slices for this document
                doc_k = k_batch[current_pair_idx : current_pair_idx + count]
                doc_inferred = inferred_batch[current_pair_idx : current_pair_idx + count]
                
                # Get the very last sentence (which was never a 'left' side)
                last_sent_idx = multi_doc_starts[i] + multi_doc_counts[i] - 1
                last_sent = denoised_input_sequences[last_sent_idx]
                
                current_pos = 0
                
                # Place merged segments
                for j in range(count):
                    length = int(doc_k[j].item())
                    if current_pos + length > self.max_doc_len:
                        length = self.max_doc_len - current_pos
                    
                    if length > 0:
                        all_patches[doc_idx, current_pos : current_pos + length] = doc_inferred[j, :length]
                        current_pos += length
                        
                    if current_pos >= self.max_doc_len:
                        break
                
                # Place last sentence
                if current_pos < self.max_doc_len:
                    remaining = self.max_doc_len - current_pos
                    take = min(self.max_seq_len, remaining)
                    all_patches[doc_idx, current_pos : current_pos + take] = last_sent[:take]
                    current_pos += take
                
                stitched_len[doc_idx] = current_pos
                current_pair_idx += count


        total_pairs = 0
        if multi_sent_mask.any():
            total_pairs = (num_seq_per_doc_filtered[multi_sent_mask] - 1).sum().item()
            
        if total_pairs > 0:
            total_position_loss = total_position_loss / total_pairs
            total_sequence_loss = total_sequence_loss / total_pairs # Assuming seq loss is per-pair
            
        return all_patches, total_position_loss, total_sequence_loss



