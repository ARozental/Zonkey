import torch
import torch.nn as nn
from configs.default_config import Config
from models.transformer import TransformerDecoder,TransformerEncoder,AutoregressiveDecoder
from typing import Tuple, Optional
import math
import torch.nn.functional as F
from losses.reconstruction import calculate_reconstruction_loss,calculate_token_loss
from utils.helper_functions import calculate_mean_similarity,expected_l2_norm,calculate_spherical_uniformity_loss,compute_improved_coverage_loss,arc_cosine_similarity_seq
from splitter.segment_splitter import SegmentSplitter
from splitter.stitcher import Stitcher
from torch.distributions import Beta


bce = F.binary_cross_entropy

class ResidualConv1d(nn.Module):
    def __init__(self, d_model, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size//2)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, L, D)
        residual = x
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.activation(x)
        x = self.norm(x)
        return x + residual


class TimeEmbedding(nn.Module):
    """Sinusoidal time/flow-position embedding + MLP.

    Flow matching samples the flow time t densely over [0, 1], so the
    conditioning signal must vary smoothly and expressively across the whole
    continuum. The old 2-vector lerp could only represent a 1-D family of
    conditioning vectors; this gives the network a proper time embedding.
    """
    def __init__(self, d_model: int, max_period: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) in [0, 1]
        t = t.view(-1).float()
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half, 1)
        )
        args = t[:, None] * freqs[None] * 1000.0  # spread t over the embedding's useful range
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.d_model:
            emb = F.pad(emb, (0, self.d_model - emb.shape[-1]))
        return self.mlp(emb)

class ZonkeyLayer(nn.Module):
    def __init__(self, level: int, previous_layer: Optional[nn.Module] = None):
        super().__init__()
        self.level = level
        self.previous_layer = previous_layer
        d_model = Config.D_MODEL[level]
        self.max_seq_len = Config.MAX_SEQ_LENGTHS[level]
        self.d_model = d_model
        self.upwards_d_model = d_model*Config.COMPRESSION_VECTORS[level]
        self.upwards_norm = expected_l2_norm(self.upwards_d_model)
        self.dim_norm = expected_l2_norm(d_model)

        # EoS prediction components
        self.eos_vector = nn.Parameter(torch.randn(d_model),requires_grad=True)
        self.eos_weight = nn.Parameter(torch.ones(1),requires_grad=True)
        self.bos_scaling = math.sqrt(d_model)
        self.eos_bias_per_position = nn.Parameter(torch.zeros(1,self.max_seq_len+Config.COMPRESSION_VECTORS[level]+1),requires_grad=True)
        self.eos_overall_bias = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.eos_target_bias = Config.EOS_TARGET_BIAS[level] #so the regularization pushes us to this instead of 0
        self.eos_temprature = nn.Parameter(torch.zeros(1),requires_grad=True)

        self.bos_layer = nn.Linear(self.d_model, 1)
        self.signal_coherence = nn.Linear(self.upwards_d_model, 1)

        self.mask_vector = nn.Parameter(torch.randn(d_model),requires_grad=True)
        self.classification_head = nn.Linear(d_model, 3)


        
        
        # compressor
        self.compressor = TransformerEncoder(
            d_model = Config.D_MODEL[level],
            n_heads = Config.NUM_HEADS[level],
            d_ff = Config.D_MODEL[level]*Config.FF_DIM_RATIO,
            max_seq_len = Config.MAX_SEQ_LENGTHS[level]+Config.COMPRESSION_VECTORS[level]+1, 
            dropout = Config.DROPOUT,
            num_layers = Config.NUM_UPWARD_LAYERS[level]
            )
        self.compressor_cls_emd = nn.Parameter(torch.randn(1,Config.COMPRESSION_VECTORS[level],d_model,device=Config.DEVICE),requires_grad=True)
        self.compressor_cls_existence_probs = nn.Parameter(torch.ones(1,Config.COMPRESSION_VECTORS[level],device=Config.DEVICE),requires_grad=False)

        self.local_feature_extractor = ResidualConv1d(d_model, kernel_size=3)

        self.segment_splitter = SegmentSplitter(
            level=level,
            max_num_sentences=Config.MAX_SEQUENCES_PER_BATCH[level])
        self.stitcher = Stitcher(level=level,token_embedding_layer=self.previous_layer)


        # Flow-time conditioning embedding (replaces the old 2-vector lerp).
        self.time_embedding = TimeEmbedding(d_model)
        


        # self.ones = torch.ones(1,Config.COMPRESSION_VECTORS[level],device=Config.DEVICE)
        self.ones = torch.ones(1,Config.COMPRESSION_VECTORS[level]+1,device=Config.DEVICE)
        self.denoiser = TransformerEncoder(
            d_model = Config.D_MODEL[level],
            n_heads = Config.NUM_HEADS[level],
            d_ff = Config.D_MODEL[level]*Config.FF_DIM_RATIO,
            max_seq_len = self.max_seq_len+Config.COMPRESSION_VECTORS[level]+1,
            dropout = Config.DROPOUT,
            num_layers = Config.NUM_DENOISER_LAYERS[level]
            )
        self.alpha = torch.tensor(1.0,device=Config.DEVICE,requires_grad=False)
        self.beta = torch.tensor(3.0,device=Config.DEVICE,requires_grad=False)
        self.beta_dist_mean = self.alpha / (self.alpha + self.beta)
        self.beta_dist = Beta(self.alpha, self.beta)
        self.noise_step_size = torch.tensor(Config.NOISE_STEP_SIZE[self.level],device=Config.DEVICE,requires_grad=False)
        self.noise_last_step_size = torch.tensor(Config.NOISE_LAST_STEP_SIZE[self.level],device=Config.DEVICE,requires_grad=False) 
        self.decompressor = TransformerDecoder(
            d_model = Config.D_MODEL[level],
            n_heads = Config.NUM_HEADS[level],
            d_ff = Config.D_MODEL[level]*Config.FF_DIM_RATIO,
            max_seq_len = self.max_seq_len+Config.COMPRESSION_VECTORS[level]+1,
            dropout = Config.DROPOUT,
            use_rezero = False,
            decoder = True,
            init_scale = 0.5,
            num_layers = Config.NUM_DECOMPRESSOR_LAYERS[level]
            )
        self.ar_decoder = AutoregressiveDecoder(decompressor=self.decompressor, denoiser=self.denoiser) #no new weights here



        if self.level != Config.AGENT_LEVELS - 1:
            self.coherence_generator = nn.Sequential(
                nn.Linear(self.upwards_d_model, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.upwards_d_model)
            )    
        
        # Drifting loss queue: FIFO of detached clean_compressed vectors from recent batches
        # persistent=False -> excluded from state_dict, no checkpoint mismatch
        queue_size = Config.DRIFTING_QUEUE_SIZE[level]
        flat_dim = Config.COMPRESSION_VECTORS[level] * d_model
        self.register_buffer('_drifting_queue', torch.zeros(queue_size, flat_dim), persistent=False)
        self._drifting_queue_ptr = 0
        self._drifting_queue_count = 0

        if Config.USE_GRADIENT_CHECKPOINTING:
            self.denoise_and_reconstruct = lambda *args, **kwargs: torch.utils.checkpoint.checkpoint(
                self._denoise_and_reconstruct, *args, **kwargs, use_reentrant=False
            )
        else:
            self.denoise_and_reconstruct = self._denoise_and_reconstruct
    
    def noise_schedule(self, t: torch.Tensor) -> torch.Tensor:
        return t
    
    # def add_noise(self, x: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
    #     """
    #     noise_level means "1 - expected cosine similarity".
    #     So noise_level=0.1 → expected cosine ≈ 0.9
    #     noise_level=0.04 → expected cosine ≈ 0.96
        
    #     Internally converts to signal_strength = (1 - noise_level)^2
    #     to achieve approximately the desired expected cosine after renormalization.
    #     """
    #     # Clamp noise_level to valid range [0, 1]
    #     nl = noise_level.clamp_(0.0, 1.0 - 1e-6)
        
    #     # Desired expected cosine similarity
    #     target_cosine = 1.0 - nl
        
    #     # Convert to signal & noise strengths (variance-preserving)
    #     # signal_strength = target_cosine ** 2   → gives E[cos] ≈ target_cosine
    #     signal_strength = target_cosine * target_cosine
    #     noise_strength = 1.0 - signal_strength

    #     # Generate isotropic Gaussian noise
    #     noise = torch.randn_like(x)

    #     # Forward diffusion step
    #     noisy = (
    #         torch.sqrt(signal_strength).view(-1, 1, 1) * x +
    #         torch.sqrt(noise_strength).view(-1, 1, 1) * noise
    #     )

    #     # Renormalize to your fixed expected norm
    #     batch_size = noisy.shape[0]
    #     noisy_flat = noisy.view(batch_size, -1)
    #     noisy_flat = F.normalize(noisy_flat, p=2, dim=-1) * self.upwards_norm

    #     return noisy_flat.view_as(noisy)
    
    @staticmethod
    def _slerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between unit vectors a and b.

        a, b: (B, D) unit vectors. t: (B,) or scalar in [0, 1].
        Returns unit vectors: t=0 -> a, t=1 -> b. Falls back to (renormalized)
        linear interp when a and b are nearly colinear (sin(omega) ~ 0).
        """
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=a.device, dtype=a.dtype)
        t = t.reshape(-1, 1).to(a.dtype)
        dot = (a * b).sum(-1, keepdim=True).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega)
        small = sin_omega.abs() < 1e-4
        coef_a = torch.sin((1.0 - t) * omega) / sin_omega
        coef_b = torch.sin(t * omega) / sin_omega
        out = coef_a * a + coef_b * b
        lerp = (1.0 - t) * a + t * b
        out = torch.where(small, lerp, out)
        return F.normalize(out, p=2, dim=-1)

    def flow_interpolate(self, x: torch.Tensor, noise_level: torch.Tensor,
                         noise: Optional[torch.Tensor] = None):
        """Flow-matching interpolation along the geodesic from data (noise_level=0)
        toward a random noise sample (noise_level=1), on the sphere of radius
        ``upwards_norm``. Returns (x_t, noise) so callers can reuse the noise sample.
        """
        batch = x.shape[0]
        x_flat = x.reshape(batch, -1)
        x_unit = F.normalize(x_flat, p=2, dim=-1)
        if noise is None:
            noise = torch.randn_like(x_flat)
        noise_unit = F.normalize(noise.reshape(batch, -1), p=2, dim=-1)
        nl = torch.clamp(noise_level, min=0.0, max=1.0) if torch.is_tensor(noise_level) else noise_level
        x_t = self._slerp(x_unit, noise_unit, nl) * self.upwards_norm
        return x_t.view_as(x), (noise_unit * self.upwards_norm).view_as(x)

    def add_noise(self, x: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        """Interpolate x toward a fresh random noise sample by ``noise_level`` along
        the geodesic (noise_level=0 -> x, noise_level=1 -> noise). Stays on the sphere."""
        return self.flow_interpolate(x, noise_level)[0]
    
    def compress(self, x: torch.Tensor, existence_probs: torch.Tensor) -> torch.Tensor:
        x0 = x
        x0 = self.local_feature_extractor(x0)

        x0 = torch.cat([self.compressor_cls_emd.expand(x0.shape[0], -1, -1), x0], dim=1)
        existence_probs = torch.cat([self.compressor_cls_existence_probs.expand(x0.shape[0], -1), existence_probs], dim=1)

        x = self.compressor(x0, existence_probs)

        compressed = x[:, :Config.COMPRESSION_VECTORS[self.level]]
        batch_size = compressed.shape[0]
        compressed_flat = compressed.view(batch_size, -1)

        cls_flat = self.compressor_cls_emd.view(1, -1)
        compressed_flat = compressed_flat - cls_flat * (1 - Config.EPS)
        compressed_flat = F.normalize(compressed_flat, p=2, dim=-1) * self.upwards_norm

        return compressed_flat.view(batch_size, Config.COMPRESSION_VECTORS[self.level], -1)

    def compute_bos_probability_splitter(self, vectors: torch.Tensor) -> torch.Tensor:
        #this uses the splitter's bos classifier, we don't actually have to do so and I'm not sure if this using it is good.
        #in fact why would we even use the same bos classifier for decompressed and denoised
        return self.segment_splitter.bos_classifier(vectors)

    
    
    def compute_bos_probability(self, vectors: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.clamp(self.bos_layer(vectors).squeeze(-1)/self.bos_scaling+Config.EOS_TARGET_BIAS[self.level],min=-13.8155,max=13.8155))
    
    def compute_bos_probability_logits(self, vectors: torch.Tensor) -> torch.Tensor:
        return self.bos_layer(vectors).squeeze(-1)+Config.EOS_TARGET_BIAS[self.level]
    
    def compressed_to_denoised(self, compressed: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        batch_size = compressed.shape[0]
        # Explicit flow-time conditioning (no cosine inference; t is always known under FM).
        if not torch.is_tensor(noise_level):
            noise_level = torch.full((batch_size,), float(noise_level), device=compressed.device, dtype=compressed.dtype)
        elif noise_level.dim() == 0:
            noise_level = noise_level.expand(batch_size)
        time_vec = self.time_embedding(noise_level).to(compressed.dtype)

        prompt = torch.cat([
            time_vec.view(batch_size, 1, compressed.shape[2]),
            compressed
        ], dim=1)

        decompressed = self.decompressor.generate(prompt, Config.MAX_SEQ_LENGTHS[self.level])
        
        vectors = decompressed[:, (1+Config.COMPRESSION_VECTORS[self.level]):]
        # vectors = decompressed[:, Config.COMPRESSION_VECTORS[self.level]:]
        bos_probability = self.compute_bos_probability(vectors)
        is_real_inferred = self.bos_probs_to_inferred_real_position(bos_probability)
        
        cum_not_eos_expanded = torch.cat([
            self.ones.expand(decompressed.shape[0], -1), 
            is_real_inferred
        ], dim=1)
        denoised = self.denoiser(decompressed, cum_not_eos_expanded)
        denoised = denoised[:, (1+Config.COMPRESSION_VECTORS[self.level]):, :]
        # denoised = denoised[:, Config.COMPRESSION_VECTORS[self.level]:, :]
        denoised = F.normalize(denoised, p=2, dim=-1) * self.dim_norm
        dbos_probability = self.compute_bos_probability(denoised)
        is_real_inferred = self.bos_probs_to_inferred_real_position(dbos_probability)
        return denoised, is_real_inferred



    def calculated_coherence_score(self, compressed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute coherence score (now as probability after sigmoid) and lower-level vectors.
        Sigmoid moved inside as requested.
        """

        decompressed = self.decompressor.generate(compressed, Config.MAX_SEQ_LENGTHS[self.level])
        vectors = decompressed[:, Config.COMPRESSION_VECTORS[self.level]:]

        bos_probability = self.compute_bos_probability(vectors)
        is_real_inferred = self.bos_probs_to_inferred_real_position(bos_probability)

        cum_not_eos_expanded = torch.cat([
            self.ones[:, :Config.COMPRESSION_VECTORS[self.level]].expand(decompressed.shape[0], -1),
            is_real_inferred
        ], dim=1)

        denoised = self.denoiser(decompressed, cum_not_eos_expanded)

        # Raw logit
        d1 = F.normalize(denoised[:, :Config.COMPRESSION_VECTORS[self.level], :].view(denoised.shape[0], -1), p=2, dim=-1) * self.upwards_norm
        d2 = F.normalize(compressed.view(denoised.shape[0], -1), p=2, dim=-1) * self.upwards_norm #should alread be normalized like this
        signal = d1-d2
        raw_score = self.signal_coherence(signal / math.sqrt(self.upwards_d_model))
        
        # Sigmoid moved inside → now returns probability (as you requested)
        coherence_prob = torch.sigmoid(raw_score)

        lower_level_vectors = denoised[:, Config.COMPRESSION_VECTORS[self.level]:, :]
        lower_level_vectors = F.normalize(lower_level_vectors, p=2, dim=-1) * self.dim_norm

        return coherence_prob, lower_level_vectors    
    
    # def get_level_coherence_loss(self, compressed: torch.Tensor, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Coherence regularizer for compressed vectors at this level.
    #     - In-level: regression predicting cleanliness (1.0 = clean / real-word-like vector)
    #       • 25% pure noise → target = 0.0
    #       • 25% clean compressed vectors → target = 1.0
    #       • 50% clean + Uniform(0,1) added noise → target = 1.0 - noise_level
    #     - Cross-level penalty: kept EXACTLY as in your current file (including your stop-gradient formula)
    #       Goal remains: push lower_mean_coherence → 1.0 so level-1 produces vectors that level-0 sees as clean words.
    #     """
    #     batch_size = compressed.shape[0]
    #     num_samples = min(N, batch_size)

    #     # Level 1: skip in-level coherence loss entirely, only compute cross-level penalty
    #     if self.level == Config.AGENT_LEVELS - 1:
    #         in_level_coherence_loss = compressed.new_zeros(())
    #         sample_indices = torch.randint(0, batch_size, (num_samples,), device=compressed.device)
    #         _, random_lower_level_vectors = self.calculated_coherence_score(compressed[sample_indices])
    #     else:
    #         device = compressed.device
    #         dtype = compressed.dtype

    #         # 25% pure noise → target = 0.0
    #         num_pure = max(1, int(num_samples * 0.25))
    #         pure_noise = torch.randn(num_pure, Config.COMPRESSION_VECTORS[self.level], self.d_model,
    #                                  device=device, dtype=dtype)
    #         pure_noise = F.normalize(pure_noise.view(num_pure, -1), p=2, dim=-1) * self.upwards_norm
    #         pure_noise = pure_noise.view(num_pure, Config.COMPRESSION_VECTORS[self.level], self.d_model)

    #         # 25% clean → target = 1.0
    #         num_clean = max(1, int(num_samples * 0.25))
    #         clean_indices = torch.randint(0, batch_size, (num_clean,), device=device)
    #         clean_part = compressed[clean_indices]

    #         # 50% clean + Uniform(0,1) noise → target = 1.0 - noise_level
    #         num_noisy = num_samples - num_pure - num_clean
    #         noise_levels = torch.rand(num_noisy, device=device)   # Uniform[0,1]
    #         noisy_indices = torch.randint(0, batch_size, (num_noisy,), device=device)
    #         noisy_part = compressed[noisy_indices].clone()
    #         noisy_part = self.add_noise(noisy_part, noise_levels)

    #         # Concatenate for one forward pass
    #         all_samples = torch.cat([pure_noise, clean_part, noisy_part], dim=0) #maybe todo later: have clean-clean_mean

    #         # Target cleanliness (this is the label we want the head to predict accurately)
    #         target_clean = torch.cat([
    #             torch.zeros(num_pure, device=device),      # pure noise
    #             torch.ones(num_clean, device=device),      # clean
    #             1.0 - noise_levels                         # noisy-clean
    #         ], dim=0)

    #         # Forward through your existing calculated_coherence_score (returns probability after sigmoid)
    #         pred_clean, random_lower_level_vectors = self.calculated_coherence_score(all_samples)

    #         # LK loss
    #         in_level_coherence_loss = F.binary_cross_entropy(pred_clean.squeeze(-1), target_clean) - F.binary_cross_entropy(target_clean, target_clean)

    #         if self.level == 0:
    #             return in_level_coherence_loss, compressed.new_zeros(())

    #     # === CROSS-LEVEL PENALTY — EXACTLY as you have it now (unchanged) ===
    #     prev_comp = Config.COMPRESSION_VECTORS[self.level - 1]
    #     prev_d_model = Config.D_MODEL[self.level - 1]

    #     lower_bos_probs = self.compute_bos_probability(random_lower_level_vectors)
    #     lower_is_real = self.bos_probs_to_inferred_real_position(lower_bos_probs)

    #     sample_weights = lower_is_real.reshape(-1).clamp_min(Config.EPS)
    #     sample_probs = sample_weights / sample_weights.sum()

    #     num_selected = max(1, num_samples * Config.COMPRESSION_VECTORS[self.level])
    #     sampled_indices = torch.multinomial(sample_probs, num_selected, replacement=True)

    #     flat_lower_vectors = random_lower_level_vectors.reshape(-1, self.d_model)
    #     selected_flat = flat_lower_vectors[sampled_indices]
    #     selected_compressed = selected_flat.view(num_selected, prev_comp, prev_d_model)

    #     lower_scores, _ = self.previous_layer.calculated_coherence_score(selected_compressed)
    #     lower_scores_detached, _ = self.previous_layer.calculated_coherence_score(selected_compressed.detach())
    #     lower_scores = (lower_scores - lower_scores_detached) + lower_scores.detach()

    #     lower_mean_coherence = lower_scores.mean()
    #     cross_level_penalty = 1.0 - lower_mean_coherence

    #     return in_level_coherence_loss, cross_level_penalty    

    # def get_level_coherence_loss(self, compressed: torch.Tensor, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     In-level loss is now purely discriminator-based (optimized).
    #     Returns (in_level_coherence_loss, cross_level_penalty)
    #     - in_level_coherence_loss : value == full BCE loss
    #                                 gradients: -d(full_loss)/d(generator/compressor/coherence_discriminator)
    #                                            +d(full_loss)/d(discriminator)
    #     """
    #     batch_size = compressed.shape[0]
    #     device = compressed.device
    #     dtype = compressed.dtype
    #     num_samples = min(N, batch_size)

    #     if self.level == Config.AGENT_LEVELS - 1:
    #         # Level 1: only cross-level (exactly as before)
    #         in_level_coherence_loss = compressed.new_zeros(())
    #         sample_indices = torch.randint(0, batch_size, (num_samples,), device=device)
    #         _, random_lower_level_vectors = self.calculated_coherence_score(compressed[sample_indices])
    #     else:
    #         # Calculate counts (prefer clean division, reduce pure if necessary)
    #         num_real = int(num_samples * 0.35)
    #         num_hard = int(num_samples * 0.35)
    #         num_pure = num_samples - num_real - num_hard   

    #         real_indices = torch.randint(0, batch_size, (num_real,), device=device)
    #         real_samples = compressed[real_indices]

    #         pure_noise = torch.randn(num_pure, Config.COMPRESSION_VECTORS[self.level], self.d_model,
    #                                  device=device, dtype=dtype)
    #         pure_noise = F.normalize(pure_noise.view(num_pure, -1), p=2, dim=-1) * self.upwards_norm
    #         pure_noise = pure_noise.view(num_pure, Config.COMPRESSION_VECTORS[self.level], self.d_model)

    #         hard_base = torch.randn(num_hard, Config.COMPRESSION_VECTORS[self.level], self.d_model,
    #                                 device=device, dtype=dtype)
    #         hard_base = F.normalize(hard_base.view(num_hard, -1), p=2, dim=-1) * self.upwards_norm
    #         hard_samples = F.normalize(hard_base + self.coherence_generator(hard_base), p=2, dim=-1) * self.upwards_norm
    #         hard_samples = hard_samples.view(num_hard, Config.COMPRESSION_VECTORS[self.level], self.d_model)

    #         # Concatenate all samples (once)
    #         all_samples = torch.cat([real_samples, hard_samples, pure_noise], dim=0)

    #         # Targets for discriminator loss (real = 1.0, fake = 0.0)
    #         target = torch.cat([
    #             torch.ones(num_real, device=device, dtype=dtype),
    #             torch.zeros(num_hard + num_pure, device=device, dtype=dtype)
    #         ], dim=0)

    #         # === Full forward (gradients everywhere, needed for lower vectors + full_loss) ===
    #         pred_full, random_lower_level_vectors = self.calculated_coherence_score(all_samples)
    #         pred_full = pred_full.squeeze(-1)
    #         full_loss = F.binary_cross_entropy(pred_full, target)

    #         pred_hard_disc_only, _ = self.calculated_coherence_score(hard_samples.detach())
    #         pred_hard_disc_only = pred_hard_disc_only.squeeze(-1)

    #         pred_no_g_grad = torch.cat([
    #             pred_full[:num_real],          # real: no generator grad
    #             pred_hard_disc_only,                    # hard: discriminator-only grad
    #             pred_full[num_real + num_hard:]  # pure: no generator grad
    #         ], dim=0)

    #         no_g_grad_loss = F.binary_cross_entropy(pred_no_g_grad, target)

    #         in_level_coherence_loss = (2 * no_g_grad_loss - full_loss).mean()

    #         if self.level == 0:
    #             return in_level_coherence_loss, compressed.new_zeros(())

    #     # === CROSS-LEVEL PENALTY — EXACTLY as you have it now (unchanged) ===
    #     prev_comp = Config.COMPRESSION_VECTORS[self.level - 1]
    #     prev_d_model = Config.D_MODEL[self.level - 1]

    #     lower_bos_probs = self.compute_bos_probability(random_lower_level_vectors)
    #     lower_is_real = self.bos_probs_to_inferred_real_position(lower_bos_probs)

    #     sample_weights = lower_is_real.reshape(-1).clamp_min(Config.EPS)
    #     sample_probs = sample_weights / sample_weights.sum()

    #     num_selected = max(1, num_samples * Config.COMPRESSION_VECTORS[self.level])
    #     sampled_indices = torch.multinomial(sample_probs, num_selected, replacement=True)

    #     flat_lower_vectors = random_lower_level_vectors.reshape(-1, self.d_model)
    #     selected_flat = flat_lower_vectors[sampled_indices]
    #     selected_compressed = selected_flat.view(num_selected, prev_comp, prev_d_model)

    #     lower_scores, _ = self.previous_layer.calculated_coherence_score(selected_compressed)
    #     lower_scores_detached, _ = self.previous_layer.calculated_coherence_score(selected_compressed.detach())
    #     lower_scores = (lower_scores - lower_scores_detached) + lower_scores.detach()

    #     lower_mean_coherence = lower_scores.mean()
    #     cross_level_penalty = 1.0 - lower_mean_coherence

    #     return in_level_coherence_loss, cross_level_penalty    

    def get_level_coherence_loss(self, compressed: torch.Tensor, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Regression version (KL-style centering) with your mixed hard examples.
        25% clean → target=1.0
        25% pure noise → target=0.0
        50% mixed: v1=noise, v2=v1+generator(v1), mixed=w*v1+(1-w)*v2, label=w (w~U(0,1))
        Uses your detach trick for gradient separation.
        """
        batch_size = compressed.shape[0]
        device = compressed.device
        dtype = compressed.dtype
        num_samples = min(N, batch_size)

        if self.level == Config.AGENT_LEVELS - 1:
            # Level 1: only cross-level (exactly as you have it)
            in_level_coherence_loss = compressed.new_zeros(())
            sample_indices = torch.randint(0, batch_size, (num_samples,), device=device)
            _, random_lower_level_vectors = self.calculated_coherence_score(compressed[sample_indices])
        else:
            # 25% clean
            num_clean = int(num_samples * 0.25)
            clean_indices = torch.randint(0, batch_size, (num_clean,), device=device)
            clean_samples = compressed[clean_indices]

            # 25% pure noise
            num_pure = int(num_samples * 0.25)
            pure_noise = torch.randn(num_pure, Config.COMPRESSION_VECTORS[self.level], self.d_model,
                                     device=device, dtype=dtype)
            pure_noise = F.normalize(pure_noise.view(num_pure, -1), p=2, dim=-1) * self.upwards_norm
            pure_noise = pure_noise.view(num_pure, Config.COMPRESSION_VECTORS[self.level], self.d_model)

            # 50% mixed hard examples
            num_mixed = num_samples - num_clean - num_pure
            w = torch.rand(num_mixed, device=device)   # U(0,1)

            v1 = torch.randn(num_mixed, Config.COMPRESSION_VECTORS[self.level], self.d_model,
                             device=device, dtype=dtype)
            v1 = F.normalize(v1.view(num_mixed, -1), p=2, dim=-1) * self.upwards_norm

            v2 = v1 + self.coherence_generator(v1)
            v2 = F.normalize(v2, p=2, dim=-1) * self.upwards_norm

            mixed_samples = w.unsqueeze(-1) * v1 + (1 - w.unsqueeze(-1)) * v2
            mixed_samples = F.normalize(mixed_samples.view(num_mixed, -1), p=2, dim=-1) * self.upwards_norm
            mixed_samples = mixed_samples.view(num_mixed, Config.COMPRESSION_VECTORS[self.level], self.d_model)

            # Concatenate
            all_samples = torch.cat([clean_samples, pure_noise, mixed_samples], dim=0)

            # Targets for regression
            target = torch.cat([
                torch.ones(num_clean, device=device, dtype=dtype),
                torch.zeros(num_pure, device=device, dtype=dtype),
                w
            ], dim=0)

            # === Full forward ===
            pred, random_lower_level_vectors = self.calculated_coherence_score(all_samples)
            pred = pred.squeeze(-1)

            full_loss = F.binary_cross_entropy(pred, target) - F.binary_cross_entropy(target, target)

            # === Detached forward for the "hardener-only" part ===
            # pred_no_g_grad, _ = self.calculated_coherence_score(all_samples.detach())
            mixed_scores, no_grad_random_lower_level_vectors = self.calculated_coherence_score(mixed_samples.detach())
            pred_no_g_grad = torch.cat([
                pred[:num_clean + num_pure],
                mixed_scores.squeeze(-1)
            ], dim=0)
            pred_no_g_grad = pred_no_g_grad
            no_g_grad_loss = F.binary_cross_entropy(pred_no_g_grad, target) - F.binary_cross_entropy(target, target)

            only_g_grad_loss = (full_loss - no_g_grad_loss)
            only_d_grad_loss = (full_loss - only_g_grad_loss)
            in_level_coherence_loss = (only_d_grad_loss - (0.1*only_g_grad_loss + 0.9*only_g_grad_loss.detach())).mean()
            # same mangitude as full_loss, opposite grad sign on g, 0.1 of grad strength on g  

            if self.level > 0:
                random_lower_level_vectors = torch.cat([random_lower_level_vectors[:num_clean + num_pure], no_grad_random_lower_level_vectors[:num_mixed]], dim=0)

        # === CROSS-LEVEL PENALTY — EXACTLY as in your current code (unchanged) ===
        if self.level == 0:
            return in_level_coherence_loss, compressed.new_zeros(())

        prev_comp = Config.COMPRESSION_VECTORS[self.level - 1]
        prev_d_model = Config.D_MODEL[self.level - 1]

        lower_bos_probs = self.compute_bos_probability(random_lower_level_vectors)
        lower_is_real = self.bos_probs_to_inferred_real_position(lower_bos_probs)

        sample_weights = lower_is_real.reshape(-1).clamp_min(Config.EPS)
        sample_probs = sample_weights / sample_weights.sum()

        num_selected = max(1, num_samples * Config.COMPRESSION_VECTORS[self.level])
        sampled_indices = torch.multinomial(sample_probs, num_selected, replacement=True)

        flat_lower_vectors = random_lower_level_vectors.reshape(-1, self.d_model)
        selected_flat = flat_lower_vectors[sampled_indices]
        selected_compressed = selected_flat.view(num_selected, prev_comp, prev_d_model)

        lower_scores, _ = self.previous_layer.calculated_coherence_score(selected_compressed)
        lower_scores_detached, _ = self.previous_layer.calculated_coherence_score(selected_compressed.detach())
        lower_scores = (lower_scores - lower_scores_detached) + lower_scores.detach()

        lower_mean_coherence = lower_scores.mean()
        cross_level_penalty = 1.0 - lower_mean_coherence

        return in_level_coherence_loss, cross_level_penalty
    
    
    def _denoise_and_reconstruct(
        self, 
        compressed: torch.Tensor, 
        input_sequence: Optional[torch.Tensor] = None, 
        all_sentence_bos_probs: Optional[torch.Tensor] = None, 
        noise_level: Optional[torch.Tensor] = None, 
        splitter_existence_share: Optional[torch.Tensor] = None, 
        token_ids: Optional[torch.Tensor] = None, 
        num_sentences_per_doc: Optional[torch.Tensor] = None, 
        previous_denoised: Optional[torch.Tensor] = None,
        original_position: Optional[torch.Tensor] = None,
        clean=False,
        fake_negatives: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Optional[dict], torch.Tensor]:
        # Flow-matching corruption: interpolate the clean compressed vector toward
        # noise along the geodesic by fraction `noise_level` (0 = clean, 1 = noise).
        noisy_compressed = self.add_noise(compressed, noise_level)
        denoised, is_real_inferred = self.compressed_to_denoised(noisy_compressed, noise_level)

        if input_sequence is None:
            return denoised, None, is_real_inferred

        # === Data-prediction (x0) reconstruction + BOS loss, supervised at every flow-time t ===
        with torch.no_grad():
            is_real_label = self.bos_probs_to_inferred_real_position(all_sentence_bos_probs)

        if self.level == 0:
            reconstruction_loss, _ = calculate_token_loss(
                token_ids, denoised, splitter_existence_share, self.previous_layer)
        else:
            reconstruction_loss, _ = calculate_reconstruction_loss(
                denoised=denoised,
                is_real_inferred=is_real_inferred,
                target_sequences=input_sequence,
                splitter_existence_share=splitter_existence_share,
                fake_negatives=fake_negatives)

        dbos_ce_loss = bce(is_real_inferred, is_real_label, reduction='none')[:, 1:].mean() - bce(is_real_label, is_real_label, reduction='none')[:, 1:].mean()

        losses = {
            "reconstruction_loss": reconstruction_loss,
            "bos_loss": dbos_ce_loss * Config.EXISTS_WEIGHT[self.level],
        }

        # Stitcher / document-reassembly losses only make sense near the data
        # manifold, so they are computed only on the clean (t~0) structural pass.
        if clean:
            reconstructed_docs, stitcher_position_loss, stitcher_sequence_loss = self.stitcher(
                denoised, is_real_inferred, num_sentences_per_doc, original_position=original_position,
                original_input_sequences=input_sequence, all_p_exist_share=splitter_existence_share, all_tokens=token_ids)
            losses["stitcher_position_loss"] = stitcher_position_loss
            losses["stitcher_sequence_loss"] = stitcher_sequence_loss * 0.01

        return denoised, losses, is_real_inferred

    
    def calculate_mlm_loss(
        self,
        input_sequence,
        is_real_inferred,
        splitter_existence_share,
        doc_sequences,
        is_real_doc_position_boolean,
        original_position,
        replacement_prob=0.25,
        num_negatives=63,
        fake_negatives=None,
        previous_denoised=None,
        full_reconstruction_ratio=0.85,
        use_compressor=False
        ):
        if self.level == 1: #move to config, keep compatible with reconstruction
            full_reconstruction_ratio = 0.85
        batch_size, seq_len, hidden_dim = input_sequence.shape
        doc_batch, doc_seq_len, _ = doc_sequences.shape
        device = input_sequence.device
        
        replace_mask = torch.rand(batch_size, seq_len, device=device) < replacement_prob
        replace_mask = replace_mask & (splitter_existence_share > 0)
        
        replace_probs = torch.rand(batch_size, seq_len, device=device)
        
        use_mask = replace_mask & (replace_probs < 0.8)
        use_random = replace_mask & (replace_probs >= 0.8) & (replace_probs < 0.9)
                
        corrupted_input = input_sequence
        
        if use_mask.any():
            mask_vector_expanded = self.mask_vector.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, hidden_dim)
            corrupted_input = torch.where(
                use_mask.unsqueeze(-1),
                mask_vector_expanded,
                corrupted_input
            )
        
        if use_random.any():
            random_vectors = torch.randn(batch_size, seq_len, hidden_dim, device=device)
            random_vectors = F.normalize(random_vectors, p=2, dim=-1) * self.dim_norm
            corrupted_input = torch.where(
                use_random.unsqueeze(-1),
                random_vectors,
                corrupted_input
            )
        x = corrupted_input
        x = torch.cat([self.compressor_cls_emd.expand(x.shape[0], -1, -1), x], dim=1)
        is_real_inferred = torch.cat([self.compressor_cls_existence_probs.expand(x.shape[0], -1), is_real_inferred], dim=1)

        if use_compressor:
            x = self.compressor(x, is_real_inferred)
        else:
            x = self.denoiser(x, is_real_inferred)
        x = x[:,Config.COMPRESSION_VECTORS[self.level]:]
        encoder_output = F.normalize(x, p=2, dim=-1) * self.dim_norm

        doc_mask = is_real_doc_position_boolean.reshape(-1)
        doc_sequences_flat = doc_sequences.reshape(-1, hidden_dim)
        
        num_total_positions = doc_batch * doc_seq_len
        num_valid = doc_mask.sum().item()
        
        if num_valid <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        encoder_output_flat = encoder_output.reshape(-1, hidden_dim)
        original_position_flat = original_position.reshape(-1, 2)
        N_queries = encoder_output_flat.shape[0]
        
        encoder_output_norm = F.normalize(encoder_output_flat, p=2, dim=-1)
        targets_norm = F.normalize(doc_sequences_flat, p=2, dim=-1)
        
        doc_idx = original_position_flat[:, 0]
        pos_idx = original_position_flat[:, 1]
        positive_indices = doc_idx * doc_seq_len + pos_idx
        
        sample_probs = doc_mask.float()
        sample_probs = sample_probs / (sample_probs.sum() + Config.EPS)
        
        # Draw all negatives as a flat batch (avoids expanding sample_probs to N_queries x doc_total)
        sampled_indices = torch.multinomial(
            sample_probs, N_queries * num_negatives, replacement=True
        ).reshape(N_queries, num_negatives)
        
        # Fix collisions where negative == positive (loop until none remain)
        positive_indices_exp = positive_indices.unsqueeze(1)
        collision_mask = (sampled_indices == positive_indices_exp)
        while collision_mask.any():
            num_collisions = collision_mask.sum().item()
            sampled_indices[collision_mask] = torch.multinomial(sample_probs, num_collisions, replacement=True)
            collision_mask = (sampled_indices == positive_indices_exp)

        # Compute full similarity matrix: (N_queries, doc_total) — much smaller than
        # gathering (N_queries, num_neg, hidden) and doing elementwise multiply+sum.
        full_sim = torch.matmul(encoder_output_norm, targets_norm.T)
        pos_sim = full_sim[torch.arange(N_queries, device=device), positive_indices]
        neg_sim = torch.gather(full_sim, 1, sampled_indices)
        del full_sim

        # Pre-normalize fake negatives if present (used in both dirty and clean paths)
        fake_neg_norm = None
        if fake_negatives is not None and fake_negatives.shape[0] > 0:
            fake_neg_norm = F.normalize(fake_negatives, p=2, dim=-1)

        if previous_denoised is not None:
            # === Segment-based dirty MLM loss (analogous to dirty reconstruction) ===
            # previous_denoised is sentence-level (batch_size, seq_len, hidden_dim),
            # same shape as input_sequence / encoder_output.
            # Use arc_cosine_similarity_seq for the positive similarity (segment-aware),
            # direct cosine for negatives (they come from doc-level, no matching previous).

            # Segment positive similarity via optimal interpolation
            pos_sim_arc, _, optimal_t = arc_cosine_similarity_seq(
                previous_denoised,
                encoder_output,
                input_sequence,
                splitter_existence_share
            )
            pos_sim_arc = pos_sim_arc.reshape(-1)  # (N_queries,)

            # Segment contrastive loss (segment pos + direct neg)
            all_sim_seg = torch.cat([pos_sim_arc.unsqueeze(1), neg_sim], dim=1)
            if fake_neg_norm is not None:
                fake_sim = torch.matmul(encoder_output_norm, fake_neg_norm.T)
                all_sim_seg = torch.cat([all_sim_seg, fake_sim], dim=1)
            logits_seg = 2 * torch.atanh(torch.clamp(all_sim_seg, min=Config.EPS-1, max=1-Config.EPS))
            labels = torch.zeros(N_queries, dtype=torch.long, device=device)
            ce_seg = F.cross_entropy(logits_seg, labels, reduction='none')

            # Direct contrastive loss (direct pos + direct neg)
            all_sim_direct = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
            if fake_neg_norm is not None:
                all_sim_direct = torch.cat([all_sim_direct, fake_sim], dim=1)
            logits_direct = 2 * torch.atanh(torch.clamp(all_sim_direct, min=Config.EPS-1, max=1-Config.EPS))
            ce_direct = F.cross_entropy(logits_direct, labels, reduction='none')

            # Blend segment and direct losses
            ce_loss = (1.0 - full_reconstruction_ratio) * ce_seg + full_reconstruction_ratio * ce_direct
            ce_loss = ce_loss.reshape(batch_size, seq_len)
            ce_loss = ce_loss * splitter_existence_share
            weighted_loss = ce_loss * replace_mask.float()
            total_loss = weighted_loss.sum() / (replace_mask.float().sum() + Config.EPS)
            return total_loss

        # === Clean MLM path (no previous_denoised) ===
        all_sim = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)

        if fake_neg_norm is not None:
            fake_sim = torch.matmul(encoder_output_norm, fake_neg_norm.T)  # (N_queries, K)
            all_sim = torch.cat([all_sim, fake_sim], dim=1)

        logits = 2 * torch.atanh(torch.clamp(all_sim, min=Config.EPS-1, max=1-Config.EPS))
        
        labels = torch.zeros(N_queries, dtype=torch.long, device=device)
        
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        ce_loss = ce_loss.reshape(batch_size, seq_len)
        ce_loss = ce_loss * splitter_existence_share
        
        weighted_loss = ce_loss * replace_mask.float()
        
        total_loss = weighted_loss.sum() / (replace_mask.float().sum() + Config.EPS)
        
        return total_loss
    
    def training_forward(self, input_sequence: torch.Tensor, is_real_inferred: torch.Tensor,all_sentence_bos_probs: torch.Tensor, splitter_existence_share: torch.Tensor,
                         token_ids: Optional[torch.Tensor] = None,
                         num_sentences_per_doc: Optional[torch.Tensor] = None,
                         original_position: Optional[torch.Tensor] = None,
                         is_real_doc_position_boolean: Optional[torch.Tensor] = None,
                         doc_sequences: Optional[torch.Tensor] = None,
                         mlm_only: Optional[bool] = False,
                         fake_negatives: Optional[torch.Tensor] = None,
                         ) -> torch.Tensor:
        input_sequence = F.normalize(input_sequence, p=2, dim=-1) * self.dim_norm

        
        mlm_loss = self.calculate_mlm_loss(
                input_sequence,
                is_real_inferred,
                splitter_existence_share,
                doc_sequences,
                is_real_doc_position_boolean,
                original_position,
                replacement_prob=0.2,
                num_negatives=100,
                fake_negatives=fake_negatives
            )
        mlm_loss_c = self.calculate_mlm_loss(
                input_sequence,
                is_real_inferred,
                splitter_existence_share,
                doc_sequences,
                is_real_doc_position_boolean,
                original_position,
                replacement_prob=0.2,
                num_negatives=100,
                fake_negatives=fake_negatives,
                use_compressor=True
            )
        mlm_loss = (mlm_loss + mlm_loss_c) / 2

        
        # if token_ids is None:
        #     mlm_loss = self.calculate_mlm_loss(
        #         input_sequence,
        #         is_real_inferred,
        #         splitter_existence_share,
        #         doc_sequences,
        #         is_real_doc_position_boolean,
        #         original_position,
        #         replacement_prob=0.2,
        #         num_negatives=100,
        #         fake_negatives=fake_negatives,
        #     )
        # else:
        #     mlm_loss = torch.zeros(1, device=input_sequence.device).mean()
        
        # if mlm_only:
        #     return None, None, {"clean_mlm_loss": mlm_loss}, None, None
        
        # original_is_real = is_real_inferred
        clean_compressed = self.compress(input_sequence, is_real_inferred)
        doc_ids = original_position[:,0,0]
        batch = clean_compressed.shape[0]

        # === Pass 1: structural anchor at t~0 (clean) ===
        # Trains the decode-at-t=0 regime (the ODE's final step) plus the
        # near-data-only structural losses (stitcher).
        t_clean = torch.zeros(batch, device=clean_compressed.device, dtype=clean_compressed.dtype)
        denoised_clean, clean_losses, is_real_inferred = self.denoise_and_reconstruct(
            clean_compressed, input_sequence, all_sentence_bos_probs, t_clean, splitter_existence_share,
            token_ids=token_ids,
            num_sentences_per_doc=num_sentences_per_doc,
            original_position=original_position,
            clean=True,
            fake_negatives=fake_negatives,
        )

        # === Pass 2: flow-matching denoiser across the whole path, t ~ U(0,1) ===
        # Single pass: the slerp interpolation *is* the intermediate, so the
        # DDMM noisy+dirty passes (which only manufactured realistic intermediates)
        # are no longer needed.
        t_fm = torch.rand(batch, device=clean_compressed.device, dtype=clean_compressed.dtype)
        denoised_fm, fm_losses, is_real_inferred_fm = self.denoise_and_reconstruct(
            clean_compressed, input_sequence, all_sentence_bos_probs, t_fm, splitter_existence_share,
            token_ids=token_ids,
            num_sentences_per_doc=num_sentences_per_doc,
            original_position=original_position,
            clean=False,
            fake_negatives=fake_negatives,
        )

        # MLM robustness on the (near-clean) denoised reconstruction.
        denoised_mlm_loss = self.calculate_mlm_loss(
            denoised_clean,
            is_real_inferred,
            splitter_existence_share,
            doc_sequences,
            is_real_doc_position_boolean,
            original_position,
            replacement_prob=0.2,
            num_negatives=100,
            fake_negatives=fake_negatives,
        )

        # Coverage loss: encourages uniform distribution of compressed vectors on the
        # sphere. Under flow matching this is load-bearing: the ODE integrates from a
        # uniform-on-sphere noise prior, so the data marginal must cover the sphere or
        # the velocity field has to extrapolate into dead zones.
        clean_uniformity = compute_improved_coverage_loss(
            clean_compressed, doc_ids=doc_ids, memory_queue=self._drifting_queue
        )

        # Create fake negatives for the level above: shuffle input vectors across
        # sequences at each position, then compress to get chimeric compressed vectors.
        # E.g. at level 0 this shuffles chars to make fake words for level 1.
        num_fake = Config.NUM_FAKE_NEGATIVES[self.level]
        batch_size_c = input_sequence.shape[0]
        if num_fake > 0 and batch_size_c > 1:
            num_fake = min(num_fake, batch_size_c)
            seq_len_c = input_sequence.shape[1]
            device = input_sequence.device
            pos_range = torch.arange(seq_len_c, device=device)
            perm = torch.stack([torch.randperm(batch_size_c, device=device)[:num_fake] for _ in range(seq_len_c)], dim=1)
            pos_expanded = pos_range.unsqueeze(0).expand(num_fake, -1)
            chimeric_input = input_sequence[perm, pos_expanded]
            with torch.no_grad():
                chimeric_bos_probs = self.compute_bos_probability(chimeric_input)
                chimeric_is_real = self.bos_probs_to_inferred_real_position(chimeric_bos_probs)
                chimeric_compressed = self.compress(chimeric_input, chimeric_is_real)
                fake_negatives_for_upper = chimeric_compressed.view(num_fake, -1).detach()
        else:
            fake_negatives_for_upper = None

        # Combine and rename losses. The "fm_*" terms (the flow-matching denoiser
        # pass) reuse the old DIRTY_* config weights so existing config files keep
        # working unchanged.
        losses = {
            "coverage_loss": clean_uniformity * Config.COVERAGE_WEIGHT[self.level],
            "clean_bos_loss": clean_losses["bos_loss"],
            "clean_mlm_loss": mlm_loss * Config.MLM_WEIGHT[self.level],
            "clean_reconstruction_loss": clean_losses["reconstruction_loss"] * Config.CLEAN_RECONSTRUCTION_WEIGHT[self.level],
            "clean_stitcher_position_loss": clean_losses["stitcher_position_loss"],
            "clean_stitcher_sequence_loss": clean_losses["stitcher_sequence_loss"],
            "fm_bos_loss": fm_losses["bos_loss"],
            "fm_reconstruction_loss": fm_losses["reconstruction_loss"] * Config.DIRTY_RECONSTRUCTION_WEIGHT[self.level],
            "fm_mlm_loss": denoised_mlm_loss * Config.DIRTY_MLM_WEIGHT[self.level],
        }

        return denoised_clean[:input_sequence.shape[0]], clean_compressed, losses, is_real_inferred, fake_negatives_for_upper
    
    def bos_probs_to_inferred_real_position(self, bos_probs: torch.Tensor) -> torch.Tensor:
        # Clamp bos_probs to minimum to prevent float32 precision loss in cumsum
        # When bos_prob is tiny (e.g. 1e-8), log1p(-bos_prob) ≈ -1e-8 which is smaller
        # than float32 precision can represent when added to cumsum values like -3.5
        bos_probs_clamped = torch.clamp(bos_probs, min=Config.EPS)
        cum_not_bos = torch.exp(torch.cumsum(torch.log1p(-bos_probs_clamped), dim=1))
        is_real_inferred = cum_not_bos / cum_not_bos[:, 0].unsqueeze(1)
        return torch.clamp(is_real_inferred, min=Config.EPS, max=1-Config.EPS)
    
    def forward(self, input_sequence: torch.Tensor, is_real_doc_position_boolean: torch.Tensor, 
                token_ids: Optional[torch.Tensor] = None, mlm_only: Optional[bool] = False,
                fake_negatives: Optional[torch.Tensor] = None) -> torch.Tensor:
        (
            all_sentence_vectors,
            all_sentence_bos_probs,
            all_sentence_is_real,
            all_p_exist_share,
            bos_per_position,
            num_sentences_per_doc,
            all_tokens,
            original_position,
            patch_loss,
            short_sentence_loss,
        ) = self.segment_splitter(
            input_sequence,
            is_real_doc_position_boolean,
            input_tokens=token_ids,
        )
        
        is_real_inferred = self.bos_probs_to_inferred_real_position(all_sentence_bos_probs) #with learning on the bos probs

        denoised, compressed, losses, denoised_is_real_inferred, fake_negatives_for_upper = self.training_forward(
            all_sentence_vectors, 
            is_real_inferred, 
            all_sentence_bos_probs, 
            all_p_exist_share,
            token_ids=all_tokens,
            num_sentences_per_doc=num_sentences_per_doc,
            original_position=original_position,
            doc_sequences=input_sequence,
            is_real_doc_position_boolean=is_real_doc_position_boolean,
            mlm_only=mlm_only,
            fake_negatives=fake_negatives)
        if mlm_only:
            return None, None, None, losses, None, None, None


        num_actual_docs = torch.count_nonzero(num_sentences_per_doc) 
        num_sentences_per_doc = num_sentences_per_doc[0:num_actual_docs]
        input_sequence = input_sequence[0:num_actual_docs]

        stitched_docs, total_position_loss, total_sequence_loss = self.stitcher(all_sentence_vectors, is_real_inferred, num_sentences_per_doc, original_position=original_position, original_input_sequences=all_sentence_vectors,all_p_exist_share=all_p_exist_share,all_tokens=all_tokens)
        
        losses["avg_bos_prob"] = bos_per_position.detach()
        wanted_bos_prob = Config.COMPRESSION_VECTORS[self.level]/Config.MAX_SEQ_LENGTHS[self.level]
        bos_per_position = torch.maximum(bos_per_position, torch.tensor(wanted_bos_prob, device=bos_per_position.device))
        losses["average_bos_loss"] = ((bos_per_position+1-wanted_bos_prob)**2 - 1)*Config.COMPRESSION_PENALTY[self.level]
        
        # Reweight quality losses by realized compression rate (unchanged intent).
        compression_reweight = (bos_per_position + 0.1) / (wanted_bos_prob + 0.1)
        losses["clean_reconstruction_loss"] = losses["clean_reconstruction_loss"] * compression_reweight
        losses["clean_stitcher_position_loss"] = losses["clean_stitcher_position_loss"] * compression_reweight
        losses["clean_stitcher_sequence_loss"] = losses["clean_stitcher_sequence_loss"] * compression_reweight
        losses["fm_bos_loss"] = losses["fm_bos_loss"] * compression_reweight
        losses["fm_reconstruction_loss"] = losses["fm_reconstruction_loss"] * compression_reweight
        losses["fm_mlm_loss"] = losses["fm_mlm_loss"] * compression_reweight

        losses["patch_loss"] = patch_loss*10
        losses["short_sentence_loss"] = short_sentence_loss*10
        losses["null_position_loss"] = total_position_loss
        losses["null_sequence_loss"] = total_sequence_loss 

        num_docs = num_sentences_per_doc.shape[0]
        max_sentences = Config.MAX_DOC_LENGTHS[self.level+1]

        feature_dim = compressed.shape[1] * compressed.shape[2]
        compressed_flat = compressed.view(-1, feature_dim)
        compressed_out = compressed.new_zeros(num_docs, max_sentences, feature_dim)
        is_real_out = compressed.new_zeros(num_docs, max_sentences)

        i = 0
        for doc_idx, s in enumerate(num_sentences_per_doc):
            s_original = s.item()
            s_int = min(s_original, max_sentences)

            if s_int > 0:
                end = i + s_int
                compressed_out[doc_idx, :s_int] = compressed_flat[i:end]
                is_real_out[doc_idx, :s_int] = 1.0

            i += s_original

        compressed = compressed_out
        is_real = is_real_out
        
        return denoised, is_real_inferred, compressed, losses, stitched_docs, is_real, fake_negatives_for_upper

    @torch.no_grad()
    def generate(
        self, 
        batch_size: int = 1,
        num_diffusion_steps: int = 2,
        max_length: Optional[int] = None,
        noise_level: float = 1.0,
        fixed_vectors: Optional[torch.Tensor] = None,
        fixed_compressed_vectors: Optional[torch.Tensor] = None,
        is_real_position: Optional[torch.Tensor] = None,
        existance_cutoff: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = Config.DEVICE
        d_model = Config.D_MODEL[self.level]
        compression_vectors = Config.COMPRESSION_VECTORS[self.level]
        
        if isinstance(noise_level, (int, float)):
            noise_level_scalar = noise_level
            noise_level = torch.tensor([noise_level] * batch_size, device=device, dtype=torch.float32)
        elif noise_level.dim() == 0:
            noise_level_scalar = noise_level.item()
            noise_level = noise_level.expand(batch_size)
        else:
            noise_level_scalar = noise_level[0].item() if noise_level.numel() > 0 else 1.0
        
        if max_length is None:
            max_length = Config.MAX_SEQ_LENGTHS[self.level]

        # === Establish the starting point x_t and starting flow-time t_start ===
        # Convention: t = noise fraction (t=0 data, t=1 noise). The ODE integrates
        # t_start -> 0. `noise_level` is the starting noise fraction.
        if fixed_compressed_vectors is not None:
            compressed = fixed_compressed_vectors
            if compressed.dim() == 2:
                batch_size = compressed.shape[0]
                compressed = compressed.view(batch_size, compression_vectors, d_model)
            else:
                batch_size = compressed.shape[0]
            compressed_flat = F.normalize(compressed.reshape(batch_size, -1), p=2, dim=-1) * self.upwards_norm
            compressed = compressed_flat.view(batch_size, compression_vectors, d_model)
            t_start = torch.full((batch_size,), noise_level_scalar, device=device, dtype=compressed.dtype)
            x_t = self.add_noise(compressed, t_start) if noise_level_scalar > 0 else compressed
        elif fixed_vectors is not None:
            fixed_vectors = fixed_vectors[:batch_size, :, :]
            fixed_vectors = F.normalize(fixed_vectors, p=2, dim=-1) * self.dim_norm
            is_real_position = is_real_position[:batch_size, :]
            compressed = self.compress(fixed_vectors, is_real_position)
            t_start = torch.full((batch_size,), noise_level_scalar, device=device, dtype=compressed.dtype)
            x_t = self.add_noise(compressed, t_start) if noise_level_scalar > 0 else compressed
        else:
            # Pure generation: start from uniform-on-sphere noise at t_start = noise_level.
            noise = torch.randn(batch_size, compression_vectors, d_model, device=device)
            noise_flat = F.normalize(noise.view(batch_size, -1), p=2, dim=-1) * self.upwards_norm
            x_t = noise_flat.view(batch_size, compression_vectors, d_model)
            t_start = torch.full((batch_size,), noise_level_scalar, device=device, dtype=x_t.dtype)

        # === Integrate the probability-flow ODE (data-prediction / DDIM-on-sphere) ===
        if num_diffusion_steps <= 0:
            # Single denoiser evaluation at t_start (used to decode a known compressed
            # vector, optionally with a little noise).
            denoised, is_real_inferred = self.compressed_to_denoised(x_t, t_start)
        else:
            ts = torch.linspace(noise_level_scalar, 0.0, num_diffusion_steps + 1, device=device)
            for step in range(num_diffusion_steps):
                t_cur = ts[step]
                t_next = ts[step + 1]
                t_cur_b = t_cur.expand(batch_size)
                denoised, is_real_inferred = self.compressed_to_denoised(x_t, t_cur_b)
                # Data prediction: re-compress the predicted clean sequence to get x1_hat.
                x1_hat = self.compress(denoised, is_real_inferred)
                # Move along the geodesic toward x1_hat so that we land exactly on it at t=0.
                frac = ((t_cur - t_next) / t_cur.clamp(min=Config.EPS)).clamp(0.0, 1.0)
                x_t_unit = F.normalize(x_t.reshape(batch_size, -1), p=2, dim=-1)
                x1_unit = F.normalize(x1_hat.reshape(batch_size, -1), p=2, dim=-1)
                x_t = (self._slerp(x_t_unit, x1_unit, frac) * self.upwards_norm).view(batch_size, compression_vectors, d_model)
            # Final clean decode at t=0 for the sharpest token vectors.
            final_t = torch.zeros(batch_size, device=device)
            denoised, is_real_inferred = self.compressed_to_denoised(x_t, final_t)

        bos_probability_final = self.compute_bos_probability(denoised)
        is_real_inferred_final = self.bos_probs_to_inferred_real_position(bos_probability_final)
        
        existence_mask = (is_real_inferred_final > existance_cutoff).float()

        return denoised, existence_mask, is_real_inferred_final

    @torch.no_grad()
    def ar_generate(self, max_length=None, existance_cutoff=0.1):
        if max_length is None:
            max_length = Config.MAX_SEQ_LENGTHS[self.level]

        device = Config.DEVICE

        # Start with a random normalized vector as seed
        seed = torch.randn(1, 1, self.d_model, device=device)
        seed = F.normalize(seed, p=2, dim=-1) * self.dim_norm

        generated = seed
        for _ in range(max_length - 1):
            output = self.ar_decoder(generated)
            output = F.normalize(output, p=2, dim=-1) * self.dim_norm
            next_vec = output[:, -1:, :]
            generated = torch.cat([generated, next_vec], dim=1)

        bos_probability = self.compute_bos_probability(generated)
        is_real_inferred = self.bos_probs_to_inferred_real_position(bos_probability)
        existence_mask = (is_real_inferred > existance_cutoff).float()

        return generated, existence_mask, is_real_inferred