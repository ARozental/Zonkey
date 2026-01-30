import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Literal, Callable
from configs.default_config import Config

class RotaryPositionalEncoding(nn.Module):
    """Rotary Positional Encoding (RoPE) - Optimized version."""
    
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin for max length
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> torch.Tensor:
        """
        Apply rotary embeddings to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, n_heads, seq_len, head_dim)
            seq_len: Sequence length
            offset: Position offset for incremental generation
            
        Returns:
            Tensor with rotary embeddings applied
        """
        batch_size, n_heads, seq_len, head_dim = x.shape
        
        # Get cached cos and sin values with offset for generation
        cos = self.cos_cached[offset:offset+seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim//2)
        sin = self.sin_cached[offset:offset+seq_len].unsqueeze(0).unsqueeze(0)
        
        # Reshape x to separate pairs of dimensions (matching original implementation)
        x = x.reshape(batch_size, n_heads, seq_len, head_dim // 2, 2)
        x0, x1 = x[..., 0], x[..., 1]
        
        # Apply rotation formula
        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos
        
        # Combine back
        out = torch.stack([out0, out1], dim=-1)
        return out.reshape(batch_size, n_heads, seq_len, head_dim)

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding from 'Attention is All You Need'.
    Applied to Q and K tensors (not embeddings directly)."""
    
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute position encodings
        position = torch.arange(max_seq_len).unsqueeze(1).float()  # (max_seq_len, 1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(base) / dim))  # (dim//2,)
        
        # Create position encodings: (max_seq_len, dim)
        pe = torch.zeros(max_seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> torch.Tensor:
        """
        Apply sinusoidal positional encodings to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, n_heads, seq_len, head_dim)
            seq_len: Sequence length
            offset: Position offset for incremental generation
            
        Returns:
            Tensor with positional encodings added
        """
        batch_size, n_heads, seq_len, head_dim = x.shape
        
        # Get position encodings with offset
        # Shape: (seq_len, head_dim)
        pos_enc = self.pe[offset:offset+seq_len, :head_dim]
        
        # Reshape to match input dimensions: (1, 1, seq_len, head_dim)
        pos_enc = pos_enc.unsqueeze(0).unsqueeze(0)
        
        # Add positional encodings to input
        return x + pos_enc
if Config.POS == "rope":
    POS = RotaryPositionalEncoding
else:
    POS = SinusoidalPositionalEncoding

class DecoderAttention(nn.Module):
    """Multi-head attention with RoPE, optional soft existence masking, and KV cache support."""
    
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        # Linear projections - using single projection for efficiency
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.pos = POS(self.d_k, max_seq_len)
        self.q_pos = torch.arange(max_seq_len+5, device=Config.DEVICE) #+5 to time and compressed vectors, we can get it from config later
        self.k_pos = torch.arange(max_seq_len+5, device=Config.DEVICE)
        self.ones = torch.ones(1,1,device=Config.DEVICE)
        self.zeros = torch.zeros(1,1,device=Config.DEVICE)

        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        return_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape
        
        # Single projection for Q, K, V - more efficient
        qkv = self.W_qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, n_heads, seq_len, d_k)
        Q, K, V = qkv[0], qkv[1], qkv[2]


        if kv_cache is not None:
            # We're in generation mode - x is just the new token(s)
            cache_len = kv_cache['k'].shape[2]
            Q = self.pos(Q, seq_len, offset=cache_len)
            K = self.pos(K, seq_len, offset=cache_len)
            
            # Concatenate with cached K, V
            K = torch.cat([kv_cache['k'], K], dim=2)
            V = torch.cat([kv_cache['v'], V], dim=2)
            
            # Update total sequence length for masking
            total_seq_len = K.shape[2]
        else:
            # Normal forward pass
            Q = self.pos(Q, seq_len)
            K = self.pos(K, seq_len)
            total_seq_len = seq_len
                
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        
        # Pre-allocate mask if we'll need it
        mask = None
        if mask is None:
            mask = self.zeros.expand_as(scores)
            
        if kv_cache is not None:
            # Only mask the new query positions against future keys
            causal = torch.triu(
                self.ones.expand(seq_len, total_seq_len)  * float('-inf'),
                # torch.ones(seq_len, total_seq_len, device=x.device, dtype=scores.dtype) * float('-inf'),
                diagonal=cache_len + 1
            )
        else:
            causal = torch.triu(
                self.ones.expand(seq_len, total_seq_len)  * float('-inf'),
                # torch.ones(total_seq_len, total_seq_len, device=x.device, dtype=scores.dtype) * float('-inf'),
                diagonal=1
            )
        mask = mask + causal.unsqueeze(0).unsqueeze(0)
        
        # Apply combined mask if any
        if mask is not None:
            scores = scores + mask
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        
        # attn_weights = torch.nan_to_num(attn_weights, 0.0)  # More efficient than where + isfinite
        # attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Transpose back and reshape
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(attn_output)
        
        if return_cache:
            new_cache = {'k': K, 'v': V}
            return output, new_cache
        return output      

class ProbabilisticAttention(nn.Module):    
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        # Linear projections - using single projection for efficiency
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Rotary positional encoding
        self.pos = POS(self.d_k, max_seq_len)
        self.q_pos = torch.arange(max_seq_len+5, device=Config.DEVICE) #+5 to time and compressed vectors, we can get it from config later
        self.k_pos = torch.arange(max_seq_len+5, device=Config.DEVICE)
        self.ones = torch.ones(1,1,device=Config.DEVICE)
        self.zeros = torch.zeros(1,1,device=Config.DEVICE)

        
        self.dropout = nn.Dropout(dropout)
        if Config.GATED_ATTENTION:
            self.gate_proj = nn.Linear(d_model, self.n_heads * self.d_k)
        else:
            self.gate_proj = None

    def forward(
        self, 
        x: torch.Tensor, 
        existence_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        """
        Apply multi-head attention with optional KV caching.
        Optimized version with asymmetric existence probability scaling using FlashAttention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            existence_probs: Optional existence probabilities             
        Returns:
            output
        """
        batch_size, seq_len, _ = x.shape
        
        # Single projection for Q, K, V - more efficient
        qkv = self.W_qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, n_heads, seq_len, d_k)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        Q = self.pos(Q, seq_len)
        K = self.pos(K, seq_len)
                
        # Construct efficient mask for SDPA if existence_probs provided
        attn_bias = None
        if existence_probs is not None:
            # exist_probs shape: (B, T)
            # Work in log space for numerical stability
            log_exist = torch.log(existence_probs.clamp(min=1e-20))
            
            # Compute log(p[k]) - log(p[q])
            # Broadcast to (B, T, T)
            log_exist_k = log_exist.unsqueeze(1)  # (B, 1, T) keys
            log_exist_q = log_exist.unsqueeze(2)  # (B, T, 1) queries
            diff = log_exist_k - log_exist_q      # (B, T, T)
            
            # Apply asymmetric scaling: key at position k affects query at position q only if k > q
            # This corresponds to the upper triangle (diagonal=1)
            attn_bias = torch.triu(diff, diagonal=1)
            
            # Mask invalid keys (where existence_probs == 0) with -inf
            valid_mask = (existence_probs > 0)
            attn_bias = attn_bias.masked_fill(~valid_mask.unsqueeze(1), float('-inf'))
            
            # Expand for heads: (B, 1, T, T)
            attn_bias = attn_bias.unsqueeze(1)

        # Apply Scaled Dot Product Attention (uses FlashAttention/MemoryEfficient kernels if available)
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_bias,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=self.scale
        )

        # Zero out attention output for invalid queries (rows)
        # We do this post-hoc to avoid NaN issues from masking rows with -inf in SDPA
        if existence_probs is not None:
            q_valid = (existence_probs > 0).view(batch_size, 1, seq_len, 1)
            attn_output = attn_output * q_valid

        if Config.GATED_ATTENTION: 
            Y_heads = attn_output  # [B, nhead, T, d_k] : SDPA output per head

            # G1 Gating: query-dependent sigmoid gate (from pre-norm input X)
            gate_input = self.gate_proj(x)  
            gate_input = gate_input.view(batch_size, seq_len, self.n_heads, self.d_k)  
            gate = torch.sigmoid(gate_input).transpose(1, 2)  # Elementwise sigmoid
            Y_gated = Y_heads * gate  # [B, nhead, T, d_k] : Head-specific multiplicative gating

            # Concatenate heads and output projection
            attn_output = Y_gated.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # [B, T, d_model]
        else: 
            # Transpose back and reshape
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(attn_output)
        
        return output      

    def cross_forward(
        self,
        seq1: torch.Tensor,
        seq2: torch.Tensor,
        seq1_exist_probs: Optional[torch.Tensor] = None,
        seq2_exist_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply multi-head cross-attention with existence probabilities.
        seq2 provides queries, seq1 provides keys and values.
        
        Args:
            seq1: Input tensor of shape (batch_size, seq_len1, d_model)
            seq2: Input tensor of shape (batch_size, seq_len2, d_model)
            seq1_exist_probs: Existence probabilities for seq1 (batch_size, seq_len1)
            seq2_exist_probs: Existence probabilities for seq2 (batch_size, seq_len2)
        Returns:
            output: Tensor of shape (batch_size, seq_len2, d_model)
        """
        batch_size, seq_len1, _ = seq1.shape
        _, seq_len2, _ = seq2.shape
        
        # Compute Q, K, V using single projection
        qkv_seq1 = self.W_qkv(seq1)  # (batch_size, seq_len1, 3 * d_model)
        qkv_seq2 = self.W_qkv(seq2)  # (batch_size, seq_len2, 3 * d_model)
        
        # Reshape and split for seq1 (K, V)
        qkv_seq1 = qkv_seq1.reshape(batch_size, seq_len1, 3, self.n_heads, self.d_k)
        qkv_seq1 = qkv_seq1.permute(2, 0, 3, 1, 4)  # (3, batch_size, n_heads, seq_len1, d_k)
        _, K, V = qkv_seq1[0], qkv_seq1[1], qkv_seq1[2]  # Ignore Q from seq1
        
        # Reshape and split for seq2 (Q)
        qkv_seq2 = qkv_seq2.reshape(batch_size, seq_len2, 3, self.n_heads, self.d_k)
        qkv_seq2 = qkv_seq2.permute(2, 0, 3, 1, 4)  # (3, batch_size, n_heads, seq_len2, d_k)
        Q = qkv_seq2[0]  # (batch_size, n_heads, seq_len2, d_k)
        
        # Apply rotary positional encoding
        Q = self.pos(Q, seq_len2)
        K = self.pos(K, seq_len1)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (batch_size, n_heads, seq_len2, seq_len1)
        
        # Apply existence probability masking
        if seq1_exist_probs is not None and seq2_exist_probs is not None:
            seq2_exist = seq2_exist_probs.view(batch_size, 1, seq_len2, 1)  # (batch_size, 1, seq_len2, 1)
            seq1_exist = seq1_exist_probs.view(batch_size, 1, 1, seq_len1)  # (batch_size, 1, 1, seq_len1)
            
            # Create validity mask for hard masking zeros
            valid_q = seq2_exist > 0  # (batch_size, 1, seq_len2, 1)
            valid_k = seq1_exist > 0  # (batch_size, 1, 1, seq_len1)
            
            mask = self.zeros.expand_as(scores)  # (batch_size, n_heads, seq_len2, seq_len1)
            neg_inf = torch.finfo(scores.dtype).min
            
            # Hard mask: positions with existence=0 can't attend or be attended to
            mask = mask.masked_fill(~valid_q, neg_inf)  # Mask invalid queries
            mask = mask.masked_fill(~valid_k, neg_inf)  # Mask invalid keys
            
            # Asymmetric scaling: key at position k affects query at position q
            # If k > q: multiply by P(k exists | q exists) = seq1_exist_probs[k] / seq2_exist_probs[q]
            log_seq2_exist = torch.log(seq2_exist_probs.clamp(min=1e-20))  # (batch_size, seq_len2)
            log_seq1_exist = torch.log(seq1_exist_probs.clamp(min=1e-20))  # (batch_size, seq_len1)
            
            # Create scaling matrix
            log_seq1_keys = log_seq1_exist.unsqueeze(1).expand(-1, seq_len2, -1)  # (batch_size, seq_len2, seq_len1)
            log_seq2_queries = log_seq2_exist.unsqueeze(2).expand(-1, -1, seq_len1)  # (batch_size, seq_len2, seq_len1)
            
            scale_matrix = log_seq1_keys - log_seq2_queries  # (batch_size, seq_len2, seq_len1)
            
            # Apply scaling only when key position > query position
            key_after_query = (self.k_pos[:seq_len1].view(1, -1) > self.q_pos[:seq_len2].view(-1, 1))  # (seq_len2, seq_len1)
            scale_matrix = torch.where(
                key_after_query.unsqueeze(0),
                scale_matrix,
                self.zeros.expand_as(scale_matrix)
            )
            mask = mask + scale_matrix.unsqueeze(1)  # Add head dimension
            
            scores = scores + mask
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        
        # Handle numerical issues
        q_invalid = ~(seq2_exist_probs > 0).view(batch_size, 1, seq_len2, 1)
        attn_weights = attn_weights.masked_fill(q_invalid, 0.0)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, n_heads, seq_len2, d_k)
        
        # Transpose back and reshape
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch_size, seq_len2, n_heads, d_k)
        attn_output = attn_output.view(batch_size, seq_len2, self.d_model)  # (batch_size, seq_len2, d_model)
        
        # Final linear projection
        output = self.W_o(attn_output)  # (batch_size, seq_len2, d_model)
        
        return output

class EfficientLocalAttention(nn.Module):
    """Optimized local attention using unfold for sliding windows: lower memory, faster for long seq."""
    def __init__(self, d_model: int, n_heads: int = 8, window_size: int = 100, 
                 max_seq_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.half_window = window_size // 2
        self.effective_window_size = 2 * self.half_window + 1  # Matches original (e.g., 101 for window_size=100)
        self.scale = 1.0 / math.sqrt(self.d_head)
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.pos = POS(self.d_head, max_seq_len)

        if Config.GATED_ATTENTION:
            self.gate_proj = nn.Linear(d_model, self.n_heads * self.d_head)
        else:
            self.gate_proj = None

    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Project QKV
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape: [B, L, H, D] -> [B, H, L, D]
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Pad K and V (before applying RoPE)
        pad_left = self.half_window
        pad_right = self.half_window
        k_padded = F.pad(k, (0, 0, pad_left, pad_right), mode='constant', value=0)
        v_padded = F.pad(v, (0, 0, pad_left, pad_right), mode='constant', value=0)
        
        # Unfold: B H L D W
        k_unfolded = k_padded.unfold(dimension=2, size=self.effective_window_size, step=1)
        v_unfolded = v_padded.unfold(dimension=2, size=self.effective_window_size, step=1)
        
        # Apply window-relative positional encoding
        # Q uses center position of window (half_window), K uses positions [0, 1, ..., window_size-1]
        # This makes positions relative within the window, not absolute in the document
        
        if isinstance(self.pos, RotaryPositionalEncoding):
            # RoPE: use cached cos/sin for center position
            cos_q = self.pos.cos_cached[self.half_window:self.half_window+1].unsqueeze(0).unsqueeze(0)  # (1, 1, 1, dim//2)
            sin_q = self.pos.sin_cached[self.half_window:self.half_window+1].unsqueeze(0).unsqueeze(0)
            
            # Apply RoPE to Q using center position
            q_reshaped = q.reshape(batch_size, self.n_heads, seq_len, self.d_head // 2, 2)
            q0, q1 = q_reshaped[..., 0], q_reshaped[..., 1]
            q_out0 = q0 * cos_q - q1 * sin_q
            q_out1 = q0 * sin_q + q1 * cos_q
            q = torch.stack([q_out0, q_out1], dim=-1).reshape(batch_size, self.n_heads, seq_len, self.d_head)
            
            # Apply RoPE to unfolded K with window-relative positions [0, 1, ..., window_size-1]
            k_unfolded = k_unfolded.transpose(-1, -2)  # [B, H, L, D, W] -> [B, H, L, W, D]
            BHL = batch_size * self.n_heads * seq_len
            k_for_rope = k_unfolded.reshape(BHL, self.effective_window_size, self.d_head).unsqueeze(1)  # [B*H*L, 1, W, D]
            k_with_rope = self.pos(k_for_rope, self.effective_window_size, offset=0)  # [B*H*L, 1, W, D]
            k_unfolded = k_with_rope.squeeze(1).reshape(batch_size, self.n_heads, seq_len, self.effective_window_size, self.d_head)
            k_unfolded = k_unfolded.transpose(-1, -2)  # [B, H, L, W, D] -> [B, H, L, D, W]
        else:
            # Sinusoidal: apply relative positional encoding
            # Q gets center position encoding, K gets window-relative positions
            # Apply to Q: use center position (half_window) for all queries
            q_for_pos = q.unsqueeze(2)  # [B, H, L, 1, D] - treat each query as seq_len=1
            q = self.pos(q_for_pos.reshape(batch_size * self.n_heads * seq_len, 1, 1, self.d_head), 
                         seq_len=1, offset=self.half_window)
            q = q.reshape(batch_size, self.n_heads, seq_len, self.d_head)
            
            # Apply to K: use window-relative positions [0, 1, ..., window_size-1]
            k_unfolded = k_unfolded.transpose(-1, -2)  # [B, H, L, D, W] -> [B, H, L, W, D]
            BHL = batch_size * self.n_heads * seq_len
            k_for_pos = k_unfolded.reshape(BHL, 1, self.effective_window_size, self.d_head)  # [B*H*L, 1, W, D]
            k_with_pos = self.pos(k_for_pos, self.effective_window_size, offset=0)  # [B*H*L, 1, W, D]
            k_unfolded = k_with_pos.squeeze(1).reshape(batch_size, self.n_heads, seq_len, self.effective_window_size, self.d_head)
            k_unfolded = k_unfolded.transpose(-1, -2)  # [B, H, L, W, D] -> [B, H, L, D, W]
        
        # Scores
        scores = torch.einsum('bhqd,bhqdw->bhqw', q, k_unfolded) * self.scale
        
        # Clamp scores for stability
        scores = torch.clamp(scores, min=-60.0, max=60.0)
        
        # Mask handling
        if mask is not None:
            mask = mask.bool() if mask.dtype != torch.bool else mask
            mask_padded = F.pad(mask, (pad_left, pad_right), mode='constant', value=False)
            mask_unfolded = mask_padded.unfold(dimension=1, size=self.effective_window_size, step=1)  # B L W
            mask_unfolded = mask_unfolded.unsqueeze(1)  # B 1 L W
            local_attn_mask = torch.full_like(scores, float('-inf'))
            local_attn_mask = local_attn_mask.masked_fill(mask_unfolded, 0.0)
            scores = scores + local_attn_mask
        
        # Improved softmax handling to avoid NaNs and stabilize gradients
        max_scores = scores.max(dim=-1, keepdim=True)[0]
        all_masked = (max_scores == float('-inf'))
        attn_weights = torch.zeros_like(scores)
        non_masked = ~all_masked.squeeze(-1)
        if non_masked.any():
            attn_weights[non_masked] = F.softmax(scores[non_masked], dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)
        
        # Output
        attn_output = torch.einsum('bhqw,bhqdw->bhqd', attn_weights, v_unfolded)
        
        if Config.GATED_ATTENTION:
            Y_heads = attn_output  # [B, nhead, T, d_head] : SDPA output per head

            gate_input = self.gate_proj(x)
            gate_input = gate_input.view(batch_size, seq_len, self.n_heads, self.d_head)
            gate = torch.sigmoid(gate_input).transpose(1, 2)
            Y_gated = Y_heads * gate

            attn_output = Y_gated.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        else:
            # Reshape
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.out_proj(attn_output)
        
        # Apply mask to output (matches original)
        if mask is not None:
            output = output * mask.unsqueeze(-1).float()
        
        return output
        

class TransformerLayer(nn.Module):
    """Unified Transformer layer with conditional attention based on is_linear flag."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0, 
                 max_seq_len: int = 2048, window_size: int = None, decoder: bool = False,
                 compute_eos_fn: Optional[Callable] = None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.compute_eos_fn = compute_eos_fn
        
        # Conditionally instantiate only one attention module based on is_linear
        if decoder:
            self.self_attn = DecoderAttention(d_model, n_heads, max_seq_len, dropout)
        elif window_size is not None:
            self.self_attn = EfficientLocalAttention(
                d_model=d_model, n_heads=n_heads, window_size=window_size, 
                max_seq_len=max_seq_len, dropout=dropout
            )
        else:
            self.self_attn = ProbabilisticAttention(d_model, n_heads, max_seq_len, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def encode(self, x: torch.Tensor, existence_probs: torch.Tensor, is_linear: bool = False) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Encoder forward pass, with optional EoS computation."""
        residual = x
        x = self.norm1(x)
        
        if is_linear:
            mask = existence_probs.bool() if is_linear else None
            attn_out = self.self_attn(x, mask=mask)
        else:
            attn_out = self.self_attn(x, existence_probs=existence_probs)


        # Use self_attn with mask from existence_probs (boolean for linear mode)
        
        # EoS computation if enabled (only for non-linear mode)
        if not is_linear and self.compute_eos_fn is not None:
            existence_probs, attn_out = self.compute_eos_fn(attn_out)
        
        x = residual + attn_out
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + x
        return existence_probs, x

    def cross_encode(
        self,
        seq1: torch.Tensor,
        seq2: torch.Tensor,
        seq1_exist_probs: torch.Tensor = None,
        seq2_exist_probs: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Apply cross-attention encoding where seq1 provides queries and seq2 provides keys/values.
        Mirrors the TransformerLayer forward pass but uses cross-attention and the full feedforward model.
        
        Args:
            seq1: Input tensor of shape (batch_size, seq_len1, d_model)
            seq2: Input tensor of shape (batch_size, seq_len2, d_model)
            seq1_exist_probs: Existence probabilities for seq1 (batch_size, seq_len1)
            seq2_exist_probs: Existence probabilities for seq2 (batch_size, seq_len2)
        Returns:
            output: Tensor of shape (batch_size, seq_len1, d_model)
        """
        # Cross-attention: seq1 queries attend to seq2 keys/values
        attn_output = self.self_attn.cross_forward(seq2, seq1, seq2_exist_probs, seq1_exist_probs)
        
        # Residual connection and layer normalization
        output = self.norm1(seq1 + attn_output)
        
        # Feedforward with full sequential model (including GELU)
        ff_output = self.ff(output)
        
        # Second residual connection and layer normalization
        output = self.norm2(output + ff_output)
        
        return output
    
    def decode(
        self, 
        x: torch.Tensor,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        return_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Decoder mode: uses causal mask, supports KV caching."""
        # Self attention with causal masking
        residual = x
        x = self.norm1(x)
        
        if return_cache:
            x, new_cache = self.self_attn(
                x, 
                kv_cache=kv_cache, return_cache=True
            )
        else:
            x = self.self_attn(x)
            new_cache = None
            
        x = residual + x
        
        # Feedforward
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + x
        
        if return_cache:
            return x, new_cache
        return x, None


class Transformer(nn.Module):
    """Unified Transformer with conditional layer instantiation."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int = 2048, 
                 dropout: float = 0.0, num_layers: int = 6, window_size: int = None,
                 compute_eos_fn: Optional[Callable] = None, decoder: bool = False,
                 use_rezero: bool = False, init_scale: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Conditionally create layers with the same is_linear flag (no dual modes per model)
        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout,
                max_seq_len=max_seq_len, window_size=window_size,
                compute_eos_fn=compute_eos_fn, decoder=decoder
            ) for _ in range(num_layers)
        ])
        
        self.last_layer_norm = nn.LayerNorm(d_model)
        
        # ReZero scales if enabled
        if use_rezero:
            self.residual_scales = nn.Parameter(torch.ones(num_layers) * init_scale)
        else:
            self.residual_scales = None
    
    def initialize_weights(self, layer_scale: float = 1.0):
        """Initialize weights for stable training."""
        for layer in self.layers:
            attn = layer.self_attn
            nn.init.normal_(attn.qkv_proj.weight, mean=0, std=layer_scale * 0.1)
            nn.init.zeros_(attn.qkv_proj.bias)
            nn.init.normal_(attn.out_proj.weight, mean=0, std=layer_scale * 0.01)
            nn.init.zeros_(attn.out_proj.bias)
            
            # FFN init
            nn.init.normal_(layer.ff[0].weight, mean=0, std=layer_scale * 0.5)
            nn.init.zeros_(layer.ff[0].bias)
            nn.init.normal_(layer.ff[3].weight, mean=0, std=layer_scale * 0.01)
            nn.init.zeros_(layer.ff[3].bias)
    
    def encode(self, x: torch.Tensor, existence_probs: torch.Tensor, is_linear: bool = False, noise_level: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode method - matches TransformerEncoder behavior exactly.
        
        Args:
            x: Input tensor
            existence_probs: Either float probabilities (normal mode) or boolean mask (linear mode)
            is_linear: If True, uses linear attention and treats existence_probs as boolean mask
        """
        if noise_level is None:
            for i, layer in enumerate(self.layers):
                # Store residual for scaling
                residual = x
                
                if not is_linear and layer.compute_eos_fn is not None:
                    existence_probs, layer_out = layer.encode(x, existence_probs, is_linear=is_linear)
                    x = layer_out
                else:
                    existence_probs, x = layer.encode(x, existence_probs, is_linear=is_linear)
                
                # Apply ReZero scaling if enabled (makes changes very small initially)
                # if self.residual_scales is not None:
                #     # x = residual + scale * (x - residual)
                #     # This ensures the layer starts as near-identity
                #     x = residual + self.residual_scales[i] * (x - residual)
        else:
            i=0
            while i < len(self.layers)-1:
                existence_probs, layer_out_clean = self.layers[i].encode(x, existence_probs, is_linear=is_linear)
                existence_probs, layer_out_noisy = self.layers[i+1].encode(x, existence_probs, is_linear=is_linear)
                
                noise_level = noise_level.view(-1,1,1)
                x = layer_out_clean*(1-noise_level) + layer_out_noisy*noise_level
                i += 2



        x = self.last_layer_norm(x)
        return x
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode method - matches TransformerDecoder behavior exactly."""
        for i, layer in enumerate(self.layers):
            # Store residual for scaling
            residual = x
            x, _ = layer.decode(x)  # Ignore cache for full decode
            
            # Apply ReZero scaling if enabled
            # if self.residual_scales is not None:
            #     x = residual + self.residual_scales[i] * (x - residual)
        x = self.last_layer_norm(x)
        return x
    
    def generate(self, prompt: torch.Tensor, n: int):
        """
        Generate vectors with optional KV caching for efficiency.
        Matches TransformerDecoder.generate behavior exactly.
        
        Args:
            prompt: Initial vectors (batch_size, prompt_len, d_model)
            n: Number of vectors to generate
            use_cache: Whether to use KV caching (much faster but uses more memory)
        """        
        # Efficient generation with KV caching
        generated = prompt
        kv_caches = [None] * self.num_layers
        
        # First pass: process the prompt and initialize caches
        x = prompt
        for i, layer in enumerate(self.layers):
            residual = x
            x, kv_caches[i] = layer.decode(x, kv_cache=None, return_cache=True)
            
            # Apply ReZero scaling if enabled
            # if self.residual_scales is not None:
            #     x = residual + self.residual_scales[i] * (x - residual)
        
        generated = x
        count = 0
        
        while count < n:
            # Only process the last vector
            next_input = generated[:, -1:, :]
            
            # Pass through layers with cached KV
            for i, layer in enumerate(self.layers):
                residual = next_input
                next_input, kv_caches[i] = layer.decode(
                    next_input, kv_cache=kv_caches[i], return_cache=True
                )
                
                # Apply ReZero scaling if enabled
                # if self.residual_scales is not None:
                #     next_input = residual + self.residual_scales[i] * (next_input - residual)
            
            generated = torch.cat([generated, next_input], dim=1)
            count += 1
        
        return generated

# Backward compatibility classes that use the unified Transformer
class TransformerEncoder(Transformer):
    """Backward compatible TransformerEncoder using unified Transformer."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int = 2048, 
                 dropout: float = 0.0, num_layers: int = 6, window_size: int = None,
                 compute_eos_fn: Optional[Callable] = None, 
                 use_rezero: bool = False, init_scale: float = 1.0):
        super().__init__(d_model, n_heads, d_ff, max_seq_len, dropout, num_layers, 
                        window_size, compute_eos_fn, use_rezero=use_rezero, init_scale=init_scale)
    
    def forward(self, x: torch.Tensor, existence_probs: torch.Tensor, is_linear: bool = False, noise_level: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encode(x, existence_probs,is_linear=is_linear,noise_level=noise_level)

class TransformerDecoder(Transformer):
    """Backward compatible TransformerDecoder using unified Transformer."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int = 2048, 
                 dropout: float = 0.0, num_layers: int = 6, window_size: int = None,
                 compute_eos_fn: Optional[Callable] = None, decoder: bool = True,
                 use_rezero: bool = False, init_scale: float = 1.0):
        super().__init__(d_model, n_heads, d_ff, max_seq_len, dropout, num_layers, 
                        window_size, compute_eos_fn, decoder=decoder, use_rezero=use_rezero, init_scale=init_scale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(x)

# Keep the original separate layer classes for backward compatibility if needed
class TransformerEncoderLayer(TransformerLayer):
    """Backward compatible encoder layer."""
    def forward(self, x: torch.Tensor, existence_probs: torch.Tensor, is_linear: bool = False) -> torch.Tensor:
        existence_probs, output = self.encode(x, existence_probs, is_linear)
        return output

class TransformerDecoderLayer(TransformerLayer):
    """Backward compatible decoder layer."""
    def forward(
        self, 
        x: torch.Tensor,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        return_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        return self.decode(x, kv_cache, return_cache)