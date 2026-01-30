import torch
import torch.nn as nn
from configs.default_config import Config
from models.transformer import TransformerDecoder,TransformerEncoder
from typing import Tuple, Optional
import math
import torch.nn.functional as F
from losses.reconstruction import calculate_reconstruction_loss,calculate_token_loss
from utils.helper_functions import calculate_mean_similarity,expected_l2_norm
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


        # Two learned time vectors for interpolation
        self.time_vec_clean = nn.Parameter(torch.randn(d_model))
        self.time_vec_noisy = nn.Parameter(torch.randn(d_model))
        


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
        
        if Config.USE_GRADIENT_CHECKPOINTING:
            self.denoise_and_reconstruct = lambda *args, **kwargs: torch.utils.checkpoint.checkpoint(
                self._denoise_and_reconstruct, *args, **kwargs, use_reentrant=False
            )
        else:
            self.denoise_and_reconstruct = self._denoise_and_reconstruct
    
    def noise_schedule(self, t: torch.Tensor) -> torch.Tensor:
        return t
    
    def add_noise(self, x: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x)
        nl = noise_level.view(-1, 1, 1)
        sqrt_t = torch.sqrt(nl)
        sqrt_1mt = torch.sqrt(1 - nl)
        res = sqrt_t * noise + sqrt_1mt * x

        batch_size = res.shape[0]
        res_flat = res.view(batch_size, -1)
        res_flat = F.normalize(res_flat, p=2, dim=-1) * self.upwards_norm
        return res_flat.view_as(res)
    
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
        time_vec = (1 - noise_level).unsqueeze(-1) * self.time_vec_clean + noise_level.unsqueeze(-1) * self.time_vec_noisy
        
        conditioned_compressed = compressed + time_vec.view(compressed.shape[0], 1, compressed.shape[2])
        
        prompt = torch.cat([
            time_vec.view(compressed.shape[0], 1, compressed.shape[2]),
            conditioned_compressed
        ], dim=1)
        
        decompressed = self.decompressor.generate(prompt, Config.MAX_SEQ_LENGTHS[self.level])
        
        vectors = decompressed[:, (1+Config.COMPRESSION_VECTORS[self.level]):]
        bos_probability = self.compute_bos_probability(vectors)
        is_real_inferred = self.bos_probs_to_inferred_real_position(bos_probability)
        
        cum_not_eos_expanded = torch.cat([
            self.ones.expand(decompressed.shape[0], -1), 
            is_real_inferred
        ], dim=1)
        denoised = self.denoiser(decompressed, cum_not_eos_expanded)[:, (1+Config.COMPRESSION_VECTORS[self.level]):, :]
        denoised = F.normalize(denoised, p=2, dim=-1) * self.dim_norm
        dbos_probability = self.compute_bos_probability(denoised)
        is_real_inferred = self.bos_probs_to_inferred_real_position(dbos_probability)

        return denoised, is_real_inferred
        
    
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
        ) -> Tuple[torch.Tensor, Optional[dict], torch.Tensor]:
        noisy_compressed = self.add_noise(compressed, noise_level)
        denoised, is_real_inferred = self.compressed_to_denoised(noisy_compressed, noise_level)

        if input_sequence is None:
            return denoised, None, is_real_inferred
        
        reconstructed_docs, stitcher_position_loss, stitcher_sequence_loss = self.stitcher(
            denoised, is_real_inferred, num_sentences_per_doc, original_position=original_position, 
            original_input_sequences=input_sequence, all_p_exist_share=splitter_existence_share, all_tokens=token_ids,
            previous_denoised=previous_denoised)
        
        losses = {}
        
        if clean or previous_denoised is not None:
            with torch.no_grad():
                is_real_label =  self.bos_probs_to_inferred_real_position(all_sentence_bos_probs)
                if previous_denoised is not None:
                    previous_bos_probability = self.compute_bos_probability(previous_denoised)
                    previous_is_real_label = self.bos_probs_to_inferred_real_position(previous_bos_probability)
            
            if self.level == 0:
                reconstruction_loss = calculate_token_loss(token_ids, denoised, splitter_existence_share, 
                                                        self.previous_layer)
                optimal_t = None
            else:
                reconstruction_loss, optimal_t = calculate_reconstruction_loss(
                    denoised=denoised, 
                    is_real_inferred=is_real_inferred, 
                    target_sequences=input_sequence, 
                    splitter_existence_share=splitter_existence_share, 
                    previous_denoised=previous_denoised)
            
            # Compute BOS loss using same interpolation parameter as reconstruction loss
            if previous_denoised is not None and optimal_t is not None:
                # Use optimal_t from reconstruction loss to interpolate BOS labels
                # optimal_t: (batch,) - one value per sequence
                # Expand to match sequence dimension
                t_exp = optimal_t.unsqueeze(1)  # (batch, 1)
                
                # Interpolate labels: segment_label = t * is_real_label + (1-t) * previous_is_real_label
                segment_label = t_exp * is_real_label + (1 - t_exp) * previous_is_real_label
                
                # Compute BCE loss with interpolated labels
                dbos_ce_loss = bce(is_real_inferred, segment_label, reduction='none')[:, 1:].mean()
            else:
                dbos_ce_loss = bce(is_real_inferred, is_real_label, reduction='none')[:, 1:].mean()

            #not accounting for splitter_existence_share in this loss, if position K has 100% chance of being a new sequence it means the existance share will be 0, we do not want to multiply the need for a stop signal by 0
            # existence_loss = (is_real_inferred.sum(dim=1) - is_real_label.sum(dim=1)).abs().mean() / Config.MAX_SEQ_LENGTHS[self.level]


            losses = {
            "stitcher_position_loss": stitcher_position_loss,
            "stitcher_sequence_loss": stitcher_sequence_loss * 0.01 * (1-noise_level).mean(),
            "reconstruction_loss": reconstruction_loss,
            "bos_loss": dbos_ce_loss*Config.EXISTS_WEIGHT[self.level]
        }
        
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
        ):
        batch_size, seq_len, hidden_dim = input_sequence.shape
        doc_batch, doc_seq_len, _ = doc_sequences.shape
        device = input_sequence.device
        
        replace_mask = torch.rand(batch_size, seq_len, device=device) < replacement_prob
        replace_mask = replace_mask & (splitter_existence_share > 0)
        
        replace_probs = torch.rand(batch_size, seq_len, device=device)
        
        mask_vector_expanded = self.mask_vector.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, hidden_dim)
        random_vectors = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        random_vectors = F.normalize(random_vectors, p=2, dim=-1) * self.dim_norm
        
        use_mask = replace_mask & (replace_probs < 0.8)
        use_random = replace_mask & (replace_probs >= 0.8) & (replace_probs < 0.9)
                
        corrupted_input = input_sequence.clone()
        
        if use_mask.any():
            corrupted_input = torch.where(
                use_mask.unsqueeze(-1),
                mask_vector_expanded,
                corrupted_input
            )
        
        if use_random.any():
            corrupted_input = torch.where(
                use_random.unsqueeze(-1),
                random_vectors,
                corrupted_input
            )
        x = corrupted_input
        x = torch.cat([self.compressor_cls_emd.expand(x.shape[0], -1, -1), x], dim=1)
        is_real_inferred = torch.cat([self.compressor_cls_existence_probs.expand(x.shape[0], -1), is_real_inferred], dim=1)

        x = self.compressor(x, is_real_inferred)
        x = x[:,Config.COMPRESSION_VECTORS[self.level]:]
        encoder_output = F.normalize(x, p=2, dim=-1) * self.dim_norm

        doc_mask = is_real_doc_position_boolean.reshape(-1)
        doc_sequences_flat = doc_sequences.reshape(-1, hidden_dim)
        
        num_total_positions = doc_batch * doc_seq_len
        num_valid = doc_mask.sum().item()
        
        if num_valid == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        encoder_output_flat = encoder_output.reshape(-1, hidden_dim)
        original_position_flat = original_position.reshape(-1, 2)
        N_queries = encoder_output_flat.shape[0]
        
        encoder_output_norm = F.normalize(encoder_output_flat, p=2, dim=-1)
        targets_norm = F.normalize(doc_sequences_flat, p=2, dim=-1)
        
        doc_idx = original_position_flat[:, 0]
        pos_idx = original_position_flat[:, 1]
        positive_indices = doc_idx * doc_seq_len + pos_idx
        
        positive_targets = targets_norm[positive_indices]
        
        pos_sim = torch.sum(encoder_output_norm * positive_targets, dim=-1)
        
        sample_probs = doc_mask.float()
        sample_probs = sample_probs / (sample_probs.sum() + Config.EPS)
        
        sampled_indices = torch.multinomial(
            sample_probs.unsqueeze(0).expand(N_queries, -1),
            num_negatives,
            replacement=True
        )
        
        neg_targets = targets_norm[sampled_indices]
        
        neg_sim = torch.sum(
            encoder_output_norm.unsqueeze(1) * neg_targets,
            dim=-1
        )
        
        all_sim = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        
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
                         ) -> torch.Tensor:
        input_sequence = F.normalize(input_sequence, p=2, dim=-1) * self.dim_norm

        if token_ids is None:
            mlm_loss = self.calculate_mlm_loss(
                input_sequence,
                is_real_inferred,
                splitter_existence_share,
                doc_sequences,
                is_real_doc_position_boolean,
                original_position,
                replacement_prob=0.2,
                num_negatives=100
            )
        else:
            mlm_loss = torch.zeros(1, device=input_sequence.device).mean()
        
        if mlm_only:
            return None, None, {"clean_mlm_loss": mlm_loss}, None
        
        clean_compressed = self.compress(input_sequence, is_real_inferred)
        doc_ids = original_position[:,0,0]
        clean_compressed_sim = calculate_mean_similarity(clean_compressed, doc_ids)
        
        # Per-sample noise level for the batch
        t0 = self.noise_last_step_size * (self.beta_dist.sample((clean_compressed.shape[0],)) / self.beta_dist_mean)
        # t0 = self.noise_schedule(torch.zeros(input_sequence.shape[0], device=clean_compressed.device))
        
        # First pass: clean compression, we have a redundent op with calculating is_real_inferred twice
        denoised, clean_losses, is_real_inferred = self.denoise_and_reconstruct(
            clean_compressed, input_sequence, all_sentence_bos_probs, t0, splitter_existence_share, 
            token_ids=token_ids,
            num_sentences_per_doc=num_sentences_per_doc,
            original_position=original_position,
            clean=True
        )
        
        
        # Compress with the soft mask from predicted EoS probabilities
        compressed = self.compress(denoised, is_real_inferred)
        noisy_compressed_sim = calculate_mean_similarity(compressed, doc_ids)
        
        # Calculate mean cosine similarity between different batch elements

        t1 = self.noise_schedule(torch.rand_like(t0))
        denoised , noisy_losses, is_real_inferred = self.denoise_and_reconstruct(
            compressed, input_sequence, all_sentence_bos_probs, t1, splitter_existence_share,
            token_ids=token_ids,
            num_sentences_per_doc=num_sentences_per_doc,
            original_position=original_position
        )

        t2 = t1 * (torch.rand_like(t1)*self.noise_step_size+(1-self.noise_step_size))
        
        denoised2, dirty_losses, is_real_inferred = self.denoise_and_reconstruct(
            compressed, input_sequence, all_sentence_bos_probs, t2, splitter_existence_share,
            token_ids=token_ids,
            num_sentences_per_doc=num_sentences_per_doc,
            previous_denoised=denoised,
            original_position=original_position
        )
        
        # Combine and rename losses
        losses = {
            "collapse_loss": (clean_compressed_sim+noisy_compressed_sim).abs(),
            "clean_bos_loss": clean_losses["bos_loss"],
            "clean_mlm_loss": mlm_loss * Config.MLM_WEIGHT[self.level],
            "clean_reconstruction_loss": clean_losses["reconstruction_loss"],
            "clean_stitcher_position_loss": clean_losses["stitcher_position_loss"],
            "clean_stitcher_sequence_loss": clean_losses["stitcher_sequence_loss"],
            "dirty_bos_loss": dirty_losses["bos_loss"],
            "dirty_reconstruction_loss": dirty_losses["reconstruction_loss"] * Config.DIRTY_RECONSTRUCTION_WEIGHT[self.level]
        }

        return denoised2[:input_sequence.shape[0]], clean_compressed, losses, is_real_inferred
    
    def bos_probs_to_inferred_real_position(self, bos_probs: torch.Tensor) -> torch.Tensor:
        cum_not_bos = torch.exp(torch.cumsum(torch.log1p(-bos_probs), dim=1))
        is_real_inferred = cum_not_bos / cum_not_bos[:, 0].unsqueeze(1)
        return torch.clamp(is_real_inferred, min=Config.EPS, max=1-Config.EPS)
    
    def forward(self, input_sequence: torch.Tensor, is_real_doc_position_boolean: torch.Tensor, 
                token_ids: Optional[torch.Tensor] = None, mlm_only: Optional[bool] = False) -> torch.Tensor:
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

        denoised, compressed, losses, denoised_is_real_inferred = self.training_forward(
            all_sentence_vectors, 
            is_real_inferred, 
            all_sentence_bos_probs, 
            all_p_exist_share,
            token_ids=all_tokens,
            num_sentences_per_doc=num_sentences_per_doc,
            original_position=original_position,
            doc_sequences=input_sequence,
            is_real_doc_position_boolean=is_real_doc_position_boolean,
            mlm_only=mlm_only)
        if mlm_only:
            return None, None, None, losses, None, None


        num_actual_docs = torch.count_nonzero(num_sentences_per_doc) 
        num_sentences_per_doc = num_sentences_per_doc[0:num_actual_docs]
        input_sequence = input_sequence[0:num_actual_docs]

        stitched_docs, total_position_loss, total_sequence_loss = self.stitcher(all_sentence_vectors, is_real_inferred, num_sentences_per_doc, original_position=original_position, original_input_sequences=all_sentence_vectors,all_p_exist_share=all_p_exist_share,all_tokens=all_tokens)
        
        losses["avg_bos_prob"] = bos_per_position.detach()
        wanted_bos_prob = 1/Config.MAX_SEQ_LENGTHS[self.level]
        bos_per_position = torch.maximum(bos_per_position, torch.tensor(wanted_bos_prob, device=bos_per_position.device))
        losses["average_bos_loss"] = ((bos_per_position+1-wanted_bos_prob)**2 - 1)*Config.COMPRESSION_PENALTY[self.level]
        
        losses["clean_reconstruction_loss"] = losses["clean_reconstruction_loss"] * (bos_per_position+0.1) / (wanted_bos_prob+0.1) 
        losses["clean_stitcher_position_loss"] = losses["clean_stitcher_position_loss"] * (bos_per_position+0.1) / (wanted_bos_prob+0.1) 
        losses["clean_stitcher_sequence_loss"] = losses["clean_stitcher_sequence_loss"] * (bos_per_position+0.1) / (wanted_bos_prob+0.1) 
        losses["dirty_bos_loss"] = losses["dirty_bos_loss"] * (bos_per_position+0.1) / (wanted_bos_prob+0.1) 
        losses["dirty_reconstruction_loss"] = losses["dirty_reconstruction_loss"] * (bos_per_position+0.1) / (wanted_bos_prob+0.1) 

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
        
        return denoised, is_real_inferred, compressed, losses, stitched_docs, is_real

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
        
        if fixed_vectors is not None:
            fixed_vectors = fixed_vectors[:batch_size,:,:]
            fixed_vectors = F.normalize(fixed_vectors, p=2, dim=-1) * self.dim_norm
            is_real_position = is_real_position[:batch_size,:]
            compressed = self.compress(fixed_vectors, is_real_position)
        else:
            compressed = torch.randn(batch_size, compression_vectors, d_model, device=device)
            compressed_flat = compressed.view(batch_size, -1)
            compressed_flat = F.normalize(compressed_flat, p=2, dim=-1) * self.upwards_norm
            compressed = compressed_flat.view(batch_size, Config.COMPRESSION_VECTORS[self.level], d_model)
            initial_noise_level = torch.full((batch_size,), noise_level_scalar, device=device, dtype=compressed.dtype)
            fixed_vectors, is_real_inferred = self.compressed_to_denoised(compressed, initial_noise_level)

        
        if fixed_compressed_vectors is not None:
            compressed = fixed_compressed_vectors
            if compressed.dim() == 2:
                compressed_flat = compressed
                batch_size = compressed_flat.shape[0]
                compressed = compressed_flat.view(batch_size, Config.COMPRESSION_VECTORS[self.level], d_model)
            else:
                batch_size = compressed.shape[0]
                compressed_flat = compressed.view(batch_size, -1)

            compressed_flat = F.normalize(compressed_flat, p=2, dim=-1) * self.upwards_norm
            compressed = compressed_flat.view(batch_size, Config.COMPRESSION_VECTORS[self.level], d_model)
            initial_noise_level = torch.full((batch_size,), noise_level_scalar, device=device, dtype=compressed.dtype)
            fixed_vectors, is_real_inferred = self.compressed_to_denoised(compressed, initial_noise_level)
        
        t_schedule = torch.linspace(1.0, 0.0, num_diffusion_steps+1, device=device) * noise_level_scalar
        
        noise_levels = self.noise_schedule(t_schedule)
        denoised = fixed_vectors
        is_real_inferred = None
        for step in range(num_diffusion_steps):
            current_noise_level = noise_levels[step].expand(batch_size)
            denoised, is_real_inferred = self.compressed_to_denoised(compressed, current_noise_level)
            compressed = self.compress(denoised, is_real_inferred)

        bos_probability_final = self.compute_bos_probability(denoised)
        is_real_inferred_final = self.bos_probs_to_inferred_real_position(bos_probability_final)
        
        existence_mask = (is_real_inferred_final > existance_cutoff).float()

        return denoised, existence_mask, is_real_inferred_final
        