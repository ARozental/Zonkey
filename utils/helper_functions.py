from configs.default_config import Config
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins.io import TorchCheckpointIO
from torch.utils.tensorboard import SummaryWriter
import os as _os


class SameDirCheckpointIO(TorchCheckpointIO):
    """Write a checkpoint to a temp file in the SAME directory as the target,
    then atomically rename it into place.

    The default Lightning/fsspec path stages the entire checkpoint via
    tempfile.mkstemp() in the system temp dir (/tmp) and then shutil.move()s it.
    On WSL that means a multi-GB write to a small tmpfs /tmp plus a cross-device
    move to /mnt/c — which both leaks /tmp and can stall. torch.save() here
    streams straight to a sibling .tmp on the destination filesystem, then
    os.replace() is an atomic same-device rename. No /tmp, no cross-device copy.
    """
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        path = str(path)
        _os.makedirs(_os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp.{_os.getpid()}"
        torch.save(checkpoint, tmp)
        _os.replace(tmp, path)
import torch
import torch.nn.functional as F
from typing import Dict, Optional
import math

def expected_l2_norm(d):
    num = math.lgamma((d + 1) / 2.0)
    den = math.lgamma(d / 2.0)
    res = math.sqrt(2.0) * math.exp(num - den)
    return res


def inverse_sigmoid(y):
    return torch.log(y / (1 - y))

def segment_cosine_similarity_seq(input_sequence, high_noise_input, low_noise_input, splitter_existence_share):
    """
    Finds the maximum weighted mean cosine similarity between high_noise_input and points 
    on the segment between input_sequence and low_noise_input, with the constraint
    that all positions in a sequence must use the same t parameter.
    
    Positions are weighted by their importance (splitter_existence_share) when computing
    the mean similarity to optimize.
    
    Args:
        input_sequence: (batch, seq_len, hidden_dim)
        high_noise_input: (batch, seq_len, hidden_dim)
        low_noise_input: (batch, seq_len, hidden_dim)
        splitter_existence_share: (batch, seq_len) importance weights for each position
    
    Returns:
        cosine_similarities: (batch, seq_len) cosine similarities for each position using optimal per-sequence t
        optimal_point: (batch, seq_len, hidden_dim) points on segment using optimal per-sequence t
        optimal_t: (batch,) the optimal interpolation parameter for each sequence
    """
    eps = 1e-6
    batch, seq_len, hidden_dim = input_sequence.shape
    
    # Normalize weights per sequence
    weights = splitter_existence_share / (splitter_existence_share.sum(dim=1, keepdim=True) + eps)  # (batch, seq_len)
    
    # Direction vector from input_sequence to low_noise_input
    d = low_noise_input - input_sequence
    
    # Precompute dot products for all positions: (batch, seq_len)
    h_dot_a = torch.sum(high_noise_input * input_sequence, dim=-1)
    h_dot_d = torch.sum(high_noise_input * d, dim=-1)
    a_dot_a = torch.sum(input_sequence * input_sequence, dim=-1)
    a_dot_d = torch.sum(input_sequence * d, dim=-1)
    d_dot_d = torch.sum(d * d, dim=-1)
    h_norm = torch.sqrt(torch.sum(high_noise_input * high_noise_input, dim=-1) + eps)
    
    # Analytical solution: use weighted average of per-position optimal t values
    # Weight by splitter_existence_share (importance) and |h_dot_d| (directional contribution)
    numerator = h_dot_d * a_dot_a - h_dot_a * a_dot_d
    denominator = h_dot_a * d_dot_d - h_dot_d * a_dot_d
    t_per_position = numerator / (denominator + eps)
    t_per_position = torch.nan_to_num(t_per_position, nan=0.5, posinf=0.5, neginf=0.5)
    t_per_position = torch.clamp(t_per_position, 0.0, 1.0)
    
    # Combine importance weights with directional contribution
    combined_weights = weights * (torch.abs(h_dot_d) + eps)
    combined_weights = combined_weights / (combined_weights.sum(dim=1, keepdim=True) + eps)
    t_interior = (t_per_position * combined_weights).sum(dim=1)  # (batch,)
    t_interior = torch.clamp(t_interior, 0.0, 1.0)
    
    # Evaluate at three candidates: t=0, t=1, t=weighted_mean
    t_candidates = torch.stack([
        torch.zeros(batch, device=input_sequence.device),
        torch.ones(batch, device=input_sequence.device),
        t_interior
    ], dim=0)  # (3, batch)
    
    # Vectorized evaluation for all candidates at once
    t_exp = t_candidates.unsqueeze(-1)  # (3, batch, 1)
    
    # Expand dimensions for broadcasting
    h_dot_a_exp = h_dot_a.unsqueeze(0)  # (1, batch, seq_len)
    h_dot_d_exp = h_dot_d.unsqueeze(0)
    a_dot_a_exp = a_dot_a.unsqueeze(0)
    a_dot_d_exp = a_dot_d.unsqueeze(0)
    d_dot_d_exp = d_dot_d.unsqueeze(0)
    h_norm_exp = h_norm.unsqueeze(0)
    
    # Compute for all candidates simultaneously: (3, batch, seq_len)
    norm_sq = a_dot_a_exp + 2 * t_exp * a_dot_d_exp + (t_exp ** 2) * d_dot_d_exp
    norm = torch.sqrt(norm_sq + eps)
    num = h_dot_a_exp + t_exp * h_dot_d_exp
    cos_sims = num / (h_norm_exp * norm + eps)
    
    # Weighted mean over positions using importance weights: (3, batch)
    weights_exp = weights.unsqueeze(0)  # (1, batch, seq_len)
    mean_sims = (cos_sims * weights_exp).sum(dim=2)  # (3, batch)
    
    # Find best candidate per sequence
    best_idx = torch.argmax(mean_sims, dim=0)  # (batch,)
    best_t = t_candidates[best_idx, torch.arange(batch, device=input_sequence.device)]
    
    # Compute final optimal point and cosine similarities using best_t
    best_t_exp = best_t.view(batch, 1, 1)
    optimal_point = input_sequence + best_t_exp * d
    final_cos_sim = F.cosine_similarity(high_noise_input, optimal_point, dim=-1, eps=eps)
    final_cos_sim = torch.clamp(final_cos_sim, -1.0, 1.0)
    
    return final_cos_sim, optimal_point, best_t

def segment_cosine_similarity(input_sequence, high_noise_input, low_noise_input):
    """
    Finds the maximum cosine similarity between high_noise_input and any point 
    on the segment between input_sequence and low_noise_input.
    
    Args:
        input_sequence: (batch, seq_len, hidden_dim)
        high_noise_input: (batch, seq_len, hidden_dim)
        low_noise_input: (batch, seq_len, hidden_dim)
    
    Returns:
        cosine_similarities: (batch, seq_len) max cosine similarities for each position
    """
    eps = 1e-6
    
    # Direction vector from input_sequence to low_noise_input
    d = low_noise_input - input_sequence
    
    # Compute dot products needed for the analytical solution
    h_dot_a = torch.sum(high_noise_input * input_sequence, dim=-1)
    h_dot_d = torch.sum(high_noise_input * d, dim=-1)
    a_dot_a = torch.sum(input_sequence * input_sequence, dim=-1)
    a_dot_d = torch.sum(input_sequence * d, dim=-1)
    d_dot_d = torch.sum(d * d, dim=-1)
    
    # Analytical solution for the critical point
    numerator = h_dot_d * a_dot_a - h_dot_a * a_dot_d
    denominator = h_dot_a * d_dot_d - h_dot_d * a_dot_d
    
    # Compute t_optimal with safe division
    t_optimal = numerator / (denominator + eps)
    
    # Replace NaN/inf with 0.5 (midpoint)
    t_optimal = torch.nan_to_num(t_optimal, nan=0.5, posinf=0.5, neginf=0.5)
    
    # Compute all three candidates at once for better vectorization
    # 1. Endpoints
    cos_sim_0 = F.cosine_similarity(high_noise_input, input_sequence, dim=-1, eps=eps)
    cos_sim_1 = F.cosine_similarity(high_noise_input, low_noise_input, dim=-1, eps=eps)
    
    # 2. Interior point (compute for all, then mask)
    t_clamped = torch.clamp(t_optimal, 0.0, 1.0)
    t_expanded = t_clamped.unsqueeze(-1)
    optimal_point = input_sequence + t_expanded * d  # Equivalent but faster than (1-t)*a + t*b
    cos_sim_optimal = F.cosine_similarity(high_noise_input, optimal_point, dim=-1, eps=eps)
    
    # Stack all candidates and take max
    # Only consider interior point if t_optimal is actually in (0, 1)
    in_interior = (t_optimal > 0) & (t_optimal < 1)
    
    # Use where to select between including or excluding the interior point
    max_cos_sim = torch.where(
        in_interior,
        torch.maximum(torch.maximum(cos_sim_0, cos_sim_1), cos_sim_optimal),
        torch.maximum(cos_sim_0, cos_sim_1)
    )
    
    # Final clamp to ensure valid range (should rarely be needed with eps in F.cosine_similarity)
    max_cos_sim = torch.clamp(max_cos_sim, -1.0, 1.0)
    
    return max_cos_sim,optimal_point

def make_trainer_config(time_now):
    project_root = Path(__file__).resolve().parents[1]
    # Checkpoints land in <repo>/checkpoints by default (override with ZONKEY_CKPT_DIR).
    # Reliable writes are handled by SameDirCheckpointIO below (no /tmp, no cross-device copy).
    ckpt_root = os.environ.get("ZONKEY_CKPT_DIR")
    root_log_dir = Path(ckpt_root).expanduser() if ckpt_root else (project_root / "checkpoints")
    run_id = time_now.strftime("%Y%m%d-%H%M%S")
    run_root = root_log_dir / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=Config.SAVE_EVERY_N_STEPS,
        save_top_k=getattr(Config, "SAVE_TOP_K", -1),
        dirpath=str(run_root),
    )
    # Map Config.DEVICE to Lightning accelerator/devices
    if Config.DEVICE == "cuda" and torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
    elif Config.DEVICE == "cpu":
        accelerator = "cpu"
        devices = 1
    else:
        # Fallback (e.g., mps)
        accelerator = Config.DEVICE
        devices = 1

    trainer_config = {
        "accelerator": accelerator,
        "devices": devices,
        "logger": False,
        "precision": Config.PRECISION,
        "plugins": [SameDirCheckpointIO()],
    }
    if Config.MAX_EPOCHS and Config.MAX_EPOCHS > 0 and not Config.MAX_STEPS:
        trainer_config["max_epochs"] = Config.MAX_EPOCHS
    else:
        trainer_config["max_steps"] = Config.MAX_STEPS
    if Config.SAVE_EVERY_N_STEPS and Config.SAVE_EVERY_N_STEPS > 0:
        trainer_config["callbacks"] = [checkpoint_callback]
    return trainer_config

def calculate_spherical_uniformity_loss(
    compressed: torch.Tensor,
    doc_ids: torch.Tensor = None,
    positions: torch.Tensor = None
) -> torch.Tensor:
    """
    Calculate loss that encourages uniform distribution on the sphere.
    Uses Gram matrix to penalize pairwise similarity.
    
    Args:
        compressed: Tensor of shape (batch_size, ...) - already normalized to fixed L2 norm
        doc_ids: Optional (batch_size,) - exclude same-doc pairs from loss
        positions: Optional (batch_size,) - exclude same-position pairs from loss
        
    Returns:
        Loss encouraging uniform spherical distribution
    """
    batch_size = compressed.shape[0]
    if batch_size <= 1:
        return torch.tensor(0.0, device=compressed.device, requires_grad=True)
    
    # Flatten each batch element into a single vector
    compressed_flat = compressed.view(batch_size, -1)
    
    # Normalize to unit vectors
    compressed_norm = F.normalize(compressed_flat, p=2, dim=-1)
    
    # 1. Mean should be zero (no angular bias)
    mean_vec = compressed_norm.mean(dim=0)
    mean_loss = (mean_vec ** 2).mean()
    
    # 2. Gram matrix: (batch, batch) - penalize pairwise similarity
    gram = compressed_norm @ compressed_norm.T
    
    # Create mask for valid pairs (True = include in loss)
    # Start with all off-diagonal pairs
    valid_mask = ~torch.eye(batch_size, dtype=torch.bool, device=compressed.device)
    
    # Exclude same-doc pairs
    if doc_ids is not None:
        same_doc = doc_ids.unsqueeze(0) == doc_ids.unsqueeze(1)
        valid_mask = valid_mask & ~same_doc
    
    # Exclude same-position pairs (e.g., first word across docs)
    if positions is not None:
        same_pos = positions.unsqueeze(0) == positions.unsqueeze(1)
        valid_mask = valid_mask & ~same_pos
    
    num_valid = valid_mask.sum()
    if num_valid == 0:
        return mean_loss
    
    # Only penalize valid pairs
    off_diag_loss = (gram[valid_mask].pow(2)).mean()
    
    return mean_loss + off_diag_loss

# def compute_improved_coverage_loss(
#     z: torch.Tensor,
#     use_koleo: bool = True          # set False to use pure atanh version
# ) -> torch.Tensor:
#     """SOTA 2025 coverage loss: KoLeo (entropy maximization) + optional temperature-free uniformity.
#     Much stronger and more stable than the original Wang & Isola version.
#     """
#     # Flatten and normalize to unit sphere (same as your old loss)
#     z_flat = z.view(z.shape[0], -1)
#     z_norm = F.normalize(z_flat, dim=-1)

#     if not use_koleo:
#         # Pure atanh version (your MLM trick applied to uniformity)
#         sim = torch.mm(z_norm, z_norm.t())
#         mask = ~torch.eye(z_norm.shape[0], dtype=torch.bool, device=z.device)
#         scaled = 2 * torch.atanh(torch.clamp(sim[mask], min=Config.EPS - 1, max=1 - Config.EPS))
#         return torch.log(torch.mean(torch.exp(scaled)))   # temperature-free

#     # === KoLeo (the main upgrade used in DINOv2 and 2025 FMs) ===
#     dist = torch.cdist(z_norm, z_norm, p=2) ** 2
#     mask = ~torch.eye(z_norm.shape[0], dtype=torch.bool, device=z.device)
#     min_dist = torch.min(dist + (~mask).float() * 1e9, dim=1)[0]  # nearest neighbor (exclude self)

#     koleo = torch.mean(torch.logsumexp(-0.5 * min_dist, dim=0))
#     return koleo


def compute_improved_coverage_loss(
    z: torch.Tensor,
    doc_ids: Optional[torch.Tensor] = None,
    memory_queue: Optional[torch.Tensor] = None,   # (Q, flat_dim) — already flattened + normalized
) -> torch.Tensor:
    """Document-aware KoLeo coverage loss (SOTA 2025 style).
    
    - z shape: (B, C, D) → flattened to (B, flat_dim = C*D) exactly as you already do
    - Nearest-neighbor search **only across different documents** (same-doc vectors are ignored)
    - Optional memory queue (_drifting_queue) for global coverage on small batches
    - Pure dot-product on the sphere, no cdist, no extra arguments
    """
    B = z.shape[0]
    z_flat = z.view(B, -1)                                      # (B, flat_dim)
    z_norm = F.normalize(z_flat, dim=-1)                        # unit sphere

    # === Build candidates: current batch + memory queue ===
    if memory_queue is not None and memory_queue.shape[0] > 0:
        candidates = torch.cat([memory_queue, z_norm], dim=0)   # (Q + B, flat_dim)
        mem_size = memory_queue.shape[0]
    else:
        candidates = z_norm
        mem_size = 0

    # Cosine similarity: every current vector vs all candidates
    sim = torch.matmul(z_norm, candidates.T)                    # (B, Q + B)

    # === Build mask for invalid neighbors (same doc or self) ===
    invalid = torch.zeros_like(sim, dtype=torch.bool, device=z.device)

    # 1. Self-similarity within the current batch
    if mem_size > 0:
        eye = torch.eye(B, dtype=torch.bool, device=z.device)
        invalid[:, mem_size:] = eye
    else:
        invalid |= torch.eye(B, dtype=torch.bool, device=z.device)

    # 2. Same-document vectors (only applied to the current-batch portion)
    if doc_ids is not None:
        doc_ids = doc_ids.view(-1)                              # (B,)
        same_doc = doc_ids.unsqueeze(1) == doc_ids.unsqueeze(0) # (B, B)
        if mem_size > 0:
            # queue part is always valid (previous batches = different docs)
            same_doc_mask = torch.cat([
                torch.zeros((B, mem_size), dtype=torch.bool, device=z.device),
                same_doc
            ], dim=1)
            invalid |= same_doc_mask
        else:
            invalid |= same_doc

    # Apply mask
    sim = sim.masked_fill(invalid, -1e9)

    # === KoLeo surrogate: soft-max of the worst (largest) cosine ===
    max_sim_nn = torch.max(sim, dim=1)[0]                       # largest cosine = smallest angle
    koleo = torch.mean(torch.logsumexp(max_sim_nn, dim=0))

    return koleo


def compute_drifting_loss(
    generated: torch.Tensor,
    real: torch.Tensor,
    temperature = 0.1,
    num_real: Optional[int] = None
) -> torch.Tensor:
    """
    Compute drifting loss that pulls generated vectors toward real distribution
    while repelling from other generated vectors.
    
    Matches the paper "Generative Modeling via Drifting" (Deng et al. 2026) Algorithm 2:
    - Multi-scale kernel: sum doubly-stochastic attention across temperatures (Section 3.3)
    - Double softmax: geometric mean of row and column normalization
    - Weight cross-multiplication for balanced attraction/repulsion
    - Cosine similarity / temperature as kernel logits
    - Unit-sphere inputs make ||V|| dimension-invariant (no drift normalization needed)
    
    Args:
        generated: Tensor of shape (N, ...) - compressed vectors (noisy and/or random path)
        real: Tensor of shape (N_real, ...) - clean compressed vectors (target distribution)
        temperature: Single float or list of floats for multi-scale kernel
        num_real: If provided, subsample this many real vectors. If None, use all.
        
    Returns:
        MSE loss toward drifted target (decreases as generated matches real)
    """
    device = generated.device
    N = generated.shape[0]
    N_real = real.shape[0]
    
    if N <= 1 or N_real == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Flatten to (N, D) - detach real to make it a frozen target
    gen_flat = generated.view(N, -1)
    real_flat = real.detach().view(N_real, -1)

    if num_real is not None and num_real > 0 and num_real < N_real:
        sampled = torch.randperm(N_real, device=device)[:num_real]
        real_flat = real_flat.index_select(0, sampled)
        N_real = real_flat.shape[0]
    
    # Normalize to unit sphere for cosine similarity kernel
    gen_norm = F.normalize(gen_flat, p=2, dim=-1)
    real_norm = F.normalize(real_flat, p=2, dim=-1)

    # Support single or multi-scale temperature
    temperatures = temperature if isinstance(temperature, (list, tuple)) else [temperature]

    # Compute drifting field V under full stop-gradient (paper Section 3.2)
    with torch.no_grad():
        gen_ng = gen_norm.detach()

        # Cosine similarities (computed once, reused across temperatures)
        sim_to_real = gen_ng @ real_norm.T  # (N, N_real)
        sim_to_gen = gen_ng @ gen_ng.T      # (N, N)

        # Mask self-similarity (paper: dist_neg += eye * 1e6)
        diag = torch.eye(N, dtype=torch.bool, device=device)
        sim_to_gen_masked = sim_to_gen.masked_fill(diag, float('-inf'))

        # Multi-scale kernel: average doubly-stochastic attention across temperatures
        A = torch.zeros(N, N_real + N, device=device)
        for T in temperatures:
            logit_real = sim_to_real / T
            logit_gen = sim_to_gen_masked / T
            logits = torch.cat([logit_real, logit_gen], dim=1)  # (N, N_real + N)

            # Double softmax (paper Algorithm 2): geometric mean of row and column normalization
            A_row = F.softmax(logits, dim=-1)   # normalize over candidates
            A_col = F.softmax(logits, dim=-2)   # normalize over queries
            A = A + torch.sqrt(A_row * A_col)
        A = A / len(temperatures)

        # Split back to positive (real) and negative (generated) attention
        A_pos = A[:, :N_real]   # (N, N_real)
        A_neg = A[:, N_real:]   # (N, N)

        # Weight cross-multiplication (paper Algorithm 2 / Eq. 11)
        W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)  # (N, N_real)
        W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)  # (N, N)

        # Compute drift: V = weighted_sum(y_pos) - weighted_sum(y_neg)
        drift_pos = W_pos @ real_norm  # (N, D)
        drift_neg = W_neg @ gen_ng     # (N, D)
        V = drift_pos - drift_neg

        # Drifted target (fully stop-grad including the x term)
        # No drift normalization: since inputs are unit-normalized, ||V|| is
        # dimension-invariant and naturally decreases as generated matches real
        target = gen_ng + V

    # Per-sample squared L2 toward drifted target ≈ ||V||², decreases with training
    diff = gen_norm - target
    loss = (diff ** 2).sum(dim=-1).mean()

    return loss


def calculate_mean_similarity(compressed: torch.Tensor, doc_ids: torch.Tensor) -> torch.Tensor:
    """
    Calculate mean cosine similarity between vectors from different documents.
    
    Args:
        compressed: Tensor of shape (batch_size, ...) where ... can be any dimensions
        doc_ids: Tensor of shape (batch_size,) containing document IDs for each vector
        
    Returns:
        Mean cosine similarity over pairs from different documents, or 0 if all same doc
    """
    batch_size = compressed.shape[0]
    
    # Flatten each batch element into a single vector
    compressed_flat = compressed.view(batch_size, -1)
    
    # Normalize
    compressed_normalized = F.normalize(compressed_flat, p=2, dim=-1)
    
    # Compute pairwise cosine similarity matrix
    cosine_sim_matrix = torch.matmul(compressed_normalized, compressed_normalized.transpose(0, 1))
    
    # Create mask for pairs from different documents (exclude same doc and diagonal)
    # doc_ids: (batch_size,) -> expand to (batch_size, batch_size) for comparison
    doc_ids_col = doc_ids.unsqueeze(1)  # (batch_size, 1)
    doc_ids_row = doc_ids.unsqueeze(0)  # (1, batch_size)
    different_doc_mask = (doc_ids_col != doc_ids_row)  # (batch_size, batch_size)
    
    # Count how many valid pairs exist
    num_valid_pairs = different_doc_mask.sum()
    
    # If all vectors are from the same document, return 0
    if num_valid_pairs == 0:
        return torch.tensor(0.0, device=compressed.device, dtype=compressed.dtype)
    
    # Calculate mean over pairs from different documents
    mean_cosine_sim = cosine_sim_matrix[different_doc_mask].mean()
    
    return mean_cosine_sim


# -----------------------------------------------------------------------------
# TensorBoard utility
# -----------------------------------------------------------------------------
from pathlib import Path
import os, socket, subprocess, sys

def make_tb_writer(time_now):
    project_root = Path(__file__).resolve().parents[1]
    root_log_dir = project_root / "tensorboard_logs"
    start_tensorboard(root_log_dir, port=6006)
    run_id = time_now.strftime("%Y%m%d-%H%M%S")
    run_root = root_log_dir / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_root))
    Config.TB_WRITER = writer
    return writer

def start_tensorboard(logdir, port):
    """Launch TensorBoard on `logdir` (defaults to project-root/tensorboard_logs) if not running."""
    if logdir is None:
        logdir_path = Path(__file__).resolve().parent.parent / "tensorboard_logs"
    else:
        logdir_path = Path(logdir).expanduser().resolve()
    logdir_path.mkdir(parents=True, exist_ok=True)

    def _port_in_use(p: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", p)) == 0

    if _port_in_use(port):
        print(f"TensorBoard already running on http://localhost:{port}")
        return

    import time
    print("Starting TensorBoard...")

    # Start TensorBoard, redirect output to a log file so we can inspect errors if any
    log_file = logdir_path / "tensorboard_stdout.log"
    with open(log_file, "a") as lf:
        subprocess.Popen([
            sys.executable, "-m", "tensorboard.main", "--logdir", str(logdir_path), "--port", str(port), "--host", "0.0.0.0"
        ], stdout=lf, stderr=lf, env={**os.environ, "LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}, start_new_session=True)

    # Wait a few seconds to ensure server starts
    for _ in range(5):
        time.sleep(1)
        if _port_in_use(port):
            print(f"TensorBoard available at http://localhost:{port}")
            break
    else:
        print(f"Warning: TensorBoard did not start within expected time. See {log_file} for details.")

def slerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Spherical Linear Interpolation (great-circle arc).
    Clean, explicit broadcasting — no loops, no dynamic dim checks.
    """
    a_unit = F.normalize(a, p=2, dim=-1)
    b_unit = F.normalize(b, p=2, dim=-1)

    cos_omega = torch.sum(a_unit * b_unit, dim=-1, keepdim=True).clamp_(-1.0 + eps, 1.0 - eps)
    omega = torch.acos(cos_omega)
    sin_omega = torch.sin(omega) + eps

    # Explicit broadcasting: add trailing singleton dimensions to match a.shape[:-1]
    t = t.view(*t.shape, *[1] * (a.dim() - t.dim()))

    sin_onem_t = torch.sin((1.0 - t) * omega)
    sin_t      = torch.sin(t * omega)

    result_unit = (sin_onem_t * a_unit + sin_t * b_unit) / sin_omega

    # Safe radius extraction (works on expanded tensors)
    radius = torch.norm(a.flatten(start_dim=0, end_dim=-2)[0], p=2)

    return result_unit * radius


def arc_cosine_similarity_seq(input_sequence, high_noise_input, low_noise_input, splitter_existence_share):
    """
    Exact same API and return shapes as segment_cosine_similarity_seq,
    but uses the true great-circle arc (SLERP) instead of the chord.
    """
    eps = 1e-6
    batch, seq_len, hidden_dim = input_sequence.shape

    weights = splitter_existence_share / (splitter_existence_share.sum(dim=1, keepdim=True) + eps)

    # Fast chord-based candidates (excellent heuristic for the arc)
    d = low_noise_input - input_sequence
    h_dot_a = torch.sum(high_noise_input * input_sequence, dim=-1)
    h_dot_d = torch.sum(high_noise_input * d, dim=-1)
    a_dot_a = torch.sum(input_sequence * input_sequence, dim=-1)
    a_dot_d = torch.sum(input_sequence * d, dim=-1)
    d_dot_d = torch.sum(d * d, dim=-1)

    numerator = h_dot_d * a_dot_a - h_dot_a * a_dot_d
    denominator = h_dot_a * d_dot_d - h_dot_d * a_dot_d
    t_per_position = numerator / (denominator + eps)
    t_per_position = torch.nan_to_num(t_per_position, nan=0.5, posinf=0.5, neginf=0.5)
    t_per_position = torch.clamp(t_per_position, 0.0, 1.0)

    combined_weights = weights * (torch.abs(h_dot_d) + eps)
    combined_weights = combined_weights / (combined_weights.sum(dim=1, keepdim=True) + eps)
    t_interior = (t_per_position * combined_weights).sum(dim=1)
    t_interior = torch.clamp(t_interior, 0.0, 1.0)

    t_candidates = torch.stack([
        torch.zeros(batch, device=input_sequence.device),
        torch.ones(batch, device=input_sequence.device),
        t_interior
    ], dim=0)  # (3, batch)

    # === Candidates on the arc ===
    t_exp = t_candidates.view(3, batch, 1).expand(3, batch, seq_len)   # explicit (3, batch, seq_len)
    arc_points = slerp(
        input_sequence.unsqueeze(0).expand(3, -1, -1, -1),
        low_noise_input.unsqueeze(0).expand(3, -1, -1, -1),
        t_exp
    )

    cos_sims = F.cosine_similarity(
        high_noise_input.unsqueeze(0).expand(3, -1, -1, -1),
        arc_points,
        dim=-1
    )

    weights_exp = weights.unsqueeze(0)
    mean_sims = (cos_sims * weights_exp).sum(dim=2)  # (3, batch)

    best_idx = torch.argmax(mean_sims, dim=0)
    best_t = t_candidates[best_idx, torch.arange(batch, device=input_sequence.device)]

    # === Final optimal point ===
    best_t_exp = best_t.view(batch, 1, 1)   # explicit (batch, 1, 1)
    optimal_point = slerp(input_sequence, low_noise_input, best_t_exp)

    final_cos_sim = F.cosine_similarity(high_noise_input, optimal_point, dim=-1)
    final_cos_sim = torch.clamp(final_cos_sim, -1.0, 1.0)

    return final_cos_sim, optimal_point, best_t