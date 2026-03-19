import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import sys
import os
import io
import contextlib
from torch.utils.checkpoint import checkpoint
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.default_config import Config
from models.zonkey_layer import ZonkeyLayer
from muon import SingleDeviceMuonWithAuxAdam, MuonWithAuxAdam
import torch.distributed as dist

class PlZonkey(pl.LightningModule):
    def __init__(self, writer=None):
        super().__init__()
        self.automatic_optimization = False
        self.model = Zonkey()
        self.model.compile()
        self.tb_writer = writer

    def calibrate_memory(self):
        """Run a worst-case forward+backward pass to measure peak GPU memory.
        
        Forces all SegmentSplitters to produce the maximum number of segments,
        creates a synthetic batch, and measures the actual peak memory usage.
        Returns True if the worst-case batch fits in GPU memory.
        """
        if not torch.cuda.is_available():
            print("Memory calibration requires CUDA.")
            return True

        device = Config.DEVICE
        self.to(device)
        print("\n--- Memory Calibration (worst-case batch) ---")
        for i, layer in enumerate(self.model.layers):
            print(f"  Level {i}: max_sequences={layer.segment_splitter.max_num_sentences}, "
                  f"seq_len={Config.MAX_SEQ_LENGTHS[i]}, d_model={Config.D_MODEL[i]}")

        # Force all splitters to produce maximum segments
        for layer in self.model.layers:
            layer.segment_splitter.force_max_segments = True

        # Create synthetic batch (random token IDs, full length, all positions real)
        fake_texts = torch.randint(
            1, Config.TOKENIZER_VOCAB_SIZE_CHARS,
            (Config.BATCH_SIZE, Config.MAX_DOC_LENGTHS[0]), device=device)
        batch = {"full_texts": fake_texts}

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        oom = False
        try:
            # Full forward pass
            leveled_compressed, leveled_losses = self.model.forward(batch)

            # Replicate the same loss aggregation as training_step
            total_loss = torch.zeros(1, device=device)
            # for l in range(len(leveled_losses) - 1):
            #     leveled_losses[l]["clean_mlm_loss"] = leveled_losses[l + 1]["clean_mlm_loss"]
            # del leveled_losses[len(leveled_losses) - 1]["clean_mlm_loss"]

            for l in range(len(leveled_losses)):
                vals = []
                for name, value in leveled_losses[l].items():
                    if isinstance(value, torch.Tensor):
                        vals.append(value)
                    elif hasattr(value, 'tensor'):
                        vals.append(value.tensor)
                if vals:
                    total_loss = total_loss + torch.stack(vals).mean()

            # Full backward pass (this is where peak memory usually occurs)
            total_loss.backward()
            torch.cuda.synchronize()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                oom = True
            else:
                raise
        finally:
            # Restore normal splitter behaviour
            for layer in self.model.layers:
                layer.segment_splitter.force_max_segments = False
            # Cleanup
            self.zero_grad(set_to_none=True)
            del batch, fake_texts
            torch.cuda.empty_cache()

        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if oom:
            print(f"\n  RESULT: OOM! Worst-case batch exceeds GPU memory ({total_gb:.1f} GB).")
            print(f"  -> Reduce MAX_SEQUENCES_PER_BATCH or other size parameters.\n")
            return False
        else:
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            pct = 100 * peak_gb / total_gb
            print(f"\n  Peak memory:  {peak_gb:.2f} GB / {total_gb:.2f} GB ({pct:.1f}%)")
            print(f"  Headroom:     {total_gb - peak_gb:.2f} GB")
            if pct > 95:
                print(f"  WARNING: >95% usage. Will likely cause slowdowns or OOM on some batches.")
            elif pct > 85:
                print(f"  CAUTION: >85% usage. May slow down under memory pressure.")
            else:
                print(f"  Looks safe for training.")
            print()
            return True

    def on_train_start(self):
        for opt in self.trainer.optimizers:
            for group in opt.param_groups:
                group["lr"] = Config.LEARNING_RATE

    def on_load_checkpoint(self, checkpoint):
        """
        Override checkpoint loading to handle optimizer switching.
        When switching between Muon and AdamW, skip loading the old optimizer state.
        """
        # Check if optimizer type has changed
        checkpoint_had_muon = False
        if 'optimizer_states' in checkpoint and len(checkpoint['optimizer_states']) > 0:
            # Try to detect if the checkpoint used Muon by checking for 'use_muon' in param_groups
            try:
                first_opt_state = checkpoint['optimizer_states'][0]
                if 'param_groups' in first_opt_state:
                    for group in first_opt_state['param_groups']:
                        if 'use_muon' in group:
                            checkpoint_had_muon = True
                            break
            except (KeyError, IndexError, TypeError):
                pass
        
        current_uses_muon = Config.USE_MUON
        
        # If optimizer type changed, remove optimizer states from checkpoint
        if checkpoint_had_muon != current_uses_muon:
            print(f"\n⚠️  Optimizer type changed: checkpoint used {'Muon' if checkpoint_had_muon else 'AdamW'}, "
                  f"current config uses {'Muon' if current_uses_muon else 'AdamW'}")
            print("   Skipping old optimizer state - will initialize fresh optimizer\n")
            if 'optimizer_states' in checkpoint:
                del checkpoint['optimizer_states']
            if 'lr_schedulers' in checkpoint:
                del checkpoint['lr_schedulers']
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()

        leveled_compressed, leveled_losses = self.model.forward(batch) 

        _print_step = None
        if getattr(self, "trainer", None) is not None:
            _print_step = int(self.trainer.global_step)
        else:
            _print_step = int(self.global_step)
        if _print_step <= 0:
            _print_step = int(batch_idx)

        if (_print_step % int(Config.PRINT_EVERY_N_STEPS) == 0) and (_print_step > 0):
            with torch.no_grad():
                _tb_text_buf = None
                _tb_redirect = None
                if self.tb_writer is not None:
                    _tb_text_buf = io.StringIO()

                    class _Tee:
                        def __init__(self, *streams):
                            self._streams = streams
                        def write(self, data):
                            for s in self._streams:
                                s.write(data)
                        def flush(self):
                            for s in self._streams:
                                if hasattr(s, "flush"):
                                    s.flush()

                    _tb_redirect = contextlib.redirect_stdout(_Tee(sys.stdout, _tb_text_buf))
                else:
                    _tb_redirect = contextlib.nullcontext()

                with _tb_redirect:
                    sample_indices = torch.randperm(Config.TOKENIZER_VOCAB_SIZE_CHARS,device=Config.DEVICE)[:100]
                    sample_embeddings = self.model.token_embedding_layer(sample_indices)
                    normalized = F.normalize(sample_embeddings, p=2, dim=1)
                    cosine_sim_matrix = torch.mm(normalized, normalized.t())
                    mask = torch.triu(torch.ones_like(cosine_sim_matrix), diagonal=1).bool()
                    avg_cosine_sim = cosine_sim_matrix[mask].mean().item()
                    print("avg_cos_sim on embeddings",avg_cosine_sim)
                    texts = batch["full_texts"][0:1,0:100]
                    token_embeddings = self.model.token_embedding_layer(texts)
                    tokens_normalized = F.normalize(token_embeddings[0][0:100], dim=-1)  # (1, seq_len, d_model)
                    embeddings_normalized = F.normalize(self.model.token_embedding_layer.weight, dim=-1)  # (vocab_size, d_model)
                    logits = torch.matmul(tokens_normalized, embeddings_normalized.t())
                    best_token_idx = logits.argmax(dim=-1)
                    tokens_out = best_token_idx.tolist()
                    print("original doc start: ","".join([chr(x) for x in tokens_out]))

                    # test to see if we make a reasonable split for the first word
                    denoised, existence_mask, is_real_inferred_final = self.model.layers[0].generate(num_diffusion_steps=0,fixed_compressed_vectors=leveled_compressed[0][0][0:1],noise_level=torch.tensor([0.0],device=Config.DEVICE))
                    tokens_normalized = F.normalize(denoised[0][0:Config.MAX_SEQ_LENGTHS[0]], dim=-1)  # (1, seq_len, d_model)
                    logits = torch.matmul(tokens_normalized, embeddings_normalized.t())
                    best_token_idx = logits.argmax(dim=-1)
                    tokens_out0 = best_token_idx.tolist()

                    denoised, existence_mask, is_real_inferred_final1 = self.model.layers[0].generate(num_diffusion_steps=0,fixed_compressed_vectors=leveled_compressed[0][0][1:2],noise_level=torch.tensor([0.0],device=Config.DEVICE))
                    tokens_normalized = F.normalize(denoised[0][0:Config.MAX_SEQ_LENGTHS[0]], dim=-1)  # (1, seq_len, d_model)
                    logits = torch.matmul(tokens_normalized, embeddings_normalized.t())
                    best_token_idx = logits.argmax(dim=-1)
                    tokens_out1 = best_token_idx.tolist()

                    denoised, existence_mask, is_real_inferred_final2 = self.model.layers[0].generate(num_diffusion_steps=0,fixed_compressed_vectors=leveled_compressed[0][0][2:3],noise_level=torch.tensor([0.0],device=Config.DEVICE))
                    tokens_normalized = F.normalize(denoised[0][0:Config.MAX_SEQ_LENGTHS[0]], dim=-1)  # (1, seq_len, d_model)
                    logits = torch.matmul(tokens_normalized, embeddings_normalized.t())
                    best_token_idx = logits.argmax(dim=-1)
                    tokens_out2 = best_token_idx.tolist()


                    print("level 0 text 0: ","".join([chr(x) for x in tokens_out0]))
                    print("level 0 text 1: ","".join([chr(x) for x in tokens_out1]))
                    print("level 0 text 2: ","".join([chr(x) for x in tokens_out2]))
                    print("is_real_inferred text 0: ",[int(10000*x)/10000 for x in is_real_inferred_final[0][0:Config.MAX_SEQ_LENGTHS[0]].tolist()])
                    print("is_real_inferred text 1: ",[int(10000*x)/10000 for x in is_real_inferred_final1[0][0:Config.MAX_SEQ_LENGTHS[0]].tolist()])
                    print("is_real_inferred text 2: ",[int(10000*x)/10000 for x in is_real_inferred_final2[0][0:Config.MAX_SEQ_LENGTHS[0]].tolist()])

                    # Generate samples from all levels
                    for level in range(len(leveled_compressed)):
                        if level > 0:
                            print(f"text from level {level} lower_diffusion_steps=0: ")
                            self.model.generate_sequence_from_level_N(level,num_diffusion_steps=0,fixed_compressed_vectors=leveled_compressed[level][0][0:1],noise_level=0.0,existance_cutoff=0.1,lower_diffusion_steps=0)
                        
                        print("-----")    
                        print(f"decompressing from level {level}: ")
                        self.model.generate_sequence_from_level_N(level,fixed_compressed_vectors=leveled_compressed[level][0][0:1])
                        if len(leveled_compressed[level])>1 and level>0:
                            print(f"decompressing from level {level} s2: ")
                            self.model.generate_sequence_from_level_N(level,fixed_compressed_vectors=leveled_compressed[level][0][1:2])

                        print(f"decompressing from level {level} with {Config.NOISE_LAST_STEP_SIZE[level]} noise: ")
                        noise_level = torch.full((leveled_compressed[level].shape[0],), Config.NOISE_LAST_STEP_SIZE[level], device=leveled_compressed[level].device, dtype=leveled_compressed[level].dtype)
                        leveled_compressed_temp = self.model.layers[level].add_noise(leveled_compressed[level],noise_level)
                        self.model.generate_sequence_from_level_N(level,fixed_compressed_vectors=leveled_compressed_temp[0][0:1],noise_level=Config.NOISE_LAST_STEP_SIZE[level])
                        print(f"random seq from level {level}: ")
                        self.model.generate_sequence_from_level_N(level,num_diffusion_steps=250,noise_level=1.0)
                        print(f"ar random seq from level {level}: ")
                        self.model.ar_generate_sequence_from_level_N(level)

                
                    # Clean up generation artifacts to free GPU memory
                    del sample_indices, sample_embeddings, normalized, cosine_sim_matrix, mask
                    del texts, token_embeddings, tokens_normalized, embeddings_normalized, logits, best_token_idx

                    if self.tb_writer is not None and _tb_text_buf is not None:
                        _captured = _tb_text_buf.getvalue()
                        if _captured.strip():
                            self.tb_writer.add_text("samples/generated", _captured, _print_step)
                
        if (self.global_step % 4 == 0):
            torch.cuda.empty_cache()  # Free fragmented memory

        should_log = (self.global_step % 5 == 0)
        # Collect loss values for total_loss computation (fast, no dict creation)
        all_loss_values = [[] for _ in range(len(leveled_losses))]
        total_loss = 0
        
        # Moving the MLM down a level for leveled weight calculations
        # for l in range(len(leveled_losses)-1):
        #     leveled_losses[l]["clean_mlm_loss"] = leveled_losses[l+1]["clean_mlm_loss"]
        # del leveled_losses[len(leveled_losses)-1]["clean_mlm_loss"]

        # Compute total loss from all levels equally
        for l in range(len(leveled_losses)):
            for name, value in leveled_losses[l].items():
                all_loss_values[l].append(value)
                
                # Only log to TensorBoard occasionally (skip Lightning's self.log entirely)
                if should_log and self.tb_writer is not None:
                    if not isinstance(value, torch.Tensor):
                        item = value.tensor.item()
                    else:
                        item = value.item()
                    self.tb_writer.add_scalar(f"level_{l}/{name}", float(item), self.global_step)
            total_loss_level = torch.stack(all_loss_values[l]).mean()
            total_loss += total_loss_level
            if self.tb_writer is not None:
                self.tb_writer.add_scalar(f"loss/_{l}", float(total_loss_level.item()), self.global_step)
                


        # Scale loss for gradient accumulation
        scaled_loss = total_loss / Config.GRAD_ACCUMULATION_STEPS
        self.manual_backward(scaled_loss)
        
        # Determine if we should step the optimizer (every N accumulation steps)
        should_step = (batch_idx + 1) % Config.GRAD_ACCUMULATION_STEPS == 0
        
        if should_step:
            if Config.GRAD_CLIP_VAL > 0:
                self.clip_gradients(optimizer, gradient_clip_val=Config.GRAD_CLIP_VAL, gradient_clip_algorithm="norm")
            optimizer.step()
            optimizer.zero_grad()
        
        if should_log and self.tb_writer is not None:
            self.tb_writer.add_scalar("loss/total", float(total_loss.item()), self.global_step)
            # Log current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            self.tb_writer.add_scalar("training/learning_rate", current_lr, self.global_step)
            # Debug: print LR every 100 steps to verify decay
        
        del all_loss_values, total_loss
        return None

    def configure_optimizers(self):
        if Config.USE_MUON:
            # Separate parameters for Muon (2D hidden layers) vs Adam (embeddings, biases, scalars)
            hidden_matrix_params = []
            other_params = []
            
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    # Use Muon for 2D+ parameters in hidden layers (not embeddings)
                    if p.ndim >= 2 and "embedding" not in name.lower():
                        hidden_matrix_params.append(p)
                    else:
                        other_params.append(p)
            
            # Create parameter groups for MuonWithAuxAdam
            param_groups = []
            if hidden_matrix_params:
                param_groups.append(dict(params=hidden_matrix_params, lr=Config.LEARNING_RATE, 
                                        momentum=Config.MUON_MOMENTUM, use_muon=True))
            if other_params:
                param_groups.append(dict(params=other_params, lr=Config.LEARNING_RATE, 
                                        eps=Config.EPS, use_muon=False))
            
            # Detect if we're in distributed training mode
            is_distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
            
            if is_distributed:
                optimizer = MuonWithAuxAdam(param_groups)
            else:
                optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
        else:
            # Use standard AdamW for all parameters
            params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=Config.LEARNING_RATE, eps=Config.EPS)

        return {"optimizer": optimizer}

class Zonkey(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_layer = nn.Embedding(Config.TOKENIZER_VOCAB_SIZE_CHARS, Config.TOKEN_EMBEDDING_SIZE)
        nn.init.normal_(self.token_embedding_layer.weight.data, mean=0.0, std=1.0)
        
        num_layers = Config.AGENT_LEVELS
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(ZonkeyLayer(i,self.token_embedding_layer))
            else:
                self.layers.append(ZonkeyLayer(i,self.layers[i-1]))



    def _clip_tail_by_existence(self, doc, level, existance_cutoff):
        max_seq_len = Config.MAX_SEQ_LENGTHS[level]
        doc_len = doc.shape[1]
        
        if doc_len <= max_seq_len:
            return doc
        
        tail_start = doc_len - max_seq_len
        tail_vectors = doc[:, tail_start:, :]
        
        bos_probs = self.layers[level].compute_bos_probability(tail_vectors)
        is_real_inferred = self.layers[level].bos_probs_to_inferred_real_position(bos_probs)
        
        existence_mask = (is_real_inferred > existance_cutoff).float()
        num_valid = existence_mask[0].sum().item()
        
        clip_length = tail_start + int(num_valid)
        return doc[:, :clip_length, :]
    
    def print_char_sequence(self,seq):
        generated_normalized = F.normalize(seq, dim=-1)
        embeddings_normalized = F.normalize(self.token_embedding_layer.weight, dim=-1)  # (vocab_size, d_model)
        logits = torch.matmul(generated_normalized, embeddings_normalized.t())
        logits = logits

        tokens_out = logits.argmax(dim=-1)
        print("".join([chr(x) for x in tokens_out]))
        return 
        
    def generate_sequence_from_level_N(self,N,num_diffusion_steps=0,fixed_compressed_vectors=None,noise_level=0.0,existance_cutoff=0.5,lower_diffusion_steps=0):
        #get initial sequence, 
        initial_seq, existence_mask, is_real_inferred_final =  self.layers[N].generate(
            batch_size=1,
            num_diffusion_steps=num_diffusion_steps,
            fixed_compressed_vectors=fixed_compressed_vectors,
            noise_level=torch.tensor(noise_level, dtype=torch.float32, device=Config.DEVICE),
            existance_cutoff=existance_cutoff)
        doc = initial_seq.squeeze(0)[0:existence_mask.bool().sum().item(),:] # <actual_len,d_model>
        while N>0:
            N-=1
            seq, existence_mask, is_real_inferred = self.layers[N].generate(
                fixed_compressed_vectors=doc,
                noise_level=torch.zeros(doc.shape[0], dtype=torch.float32, device=Config.DEVICE), #just use the denoiser, no actual diffusion done
                num_diffusion_steps=lower_diffusion_steps, # zero here for no refinement by lower layers
                existance_cutoff=existance_cutoff #can remove this, no need to hard code it here
                )
            doc,_,_ = self.layers[N].stitcher(seq, is_real_inferred, torch.tensor([seq.shape[0]], dtype=torch.long, device=seq.device))
            
            doc = self._clip_tail_by_existence(doc, N, existance_cutoff)
            doc = doc.squeeze(0)
        self.print_char_sequence(doc[:100])
        return doc

    def ar_generate_sequence_from_level_N(self, N, existance_cutoff=0.5):
        initial_seq, existence_mask, is_real_inferred_final = self.layers[N].ar_generate()
        doc = initial_seq.squeeze(0)[0:existence_mask.bool().sum().item(), :]
        level = N
        while level > 0:
            level -= 1
            seq, existence_mask, is_real_inferred = self.layers[level].generate(
                fixed_compressed_vectors=doc,
                noise_level=torch.zeros(doc.shape[0], dtype=torch.float32, device=Config.DEVICE),
                num_diffusion_steps=0,
                existance_cutoff=existance_cutoff
            )
            doc, _, _ = self.layers[level].stitcher(seq, is_real_inferred, torch.tensor([seq.shape[0]], dtype=torch.long, device=seq.device))
            doc = self._clip_tail_by_existence(doc, level, existance_cutoff)
            doc = doc.squeeze(0)
        self.print_char_sequence(doc[:100])
        return doc

    def forward(self, batch):
        texts = batch["full_texts"]
        token_embeddings = self.token_embedding_layer(texts)
        is_real_position = (texts != 0)
        leveled_compressed = []
        leveled_losses = []
        
        fake_negatives = None
        for i in range(len(self.layers)):
            if Config.USE_GRADIENT_CHECKPOINTING:
                if i == 0:
                    denoised, is_real_inferred, compressed, losses, reconstructed_docs, is_real, fake_negatives = checkpoint(
                        self.layers[i], token_embeddings, is_real_position, texts, False, None, use_reentrant=False)
                else:
                    _, _, compressed, losses, _, is_real, fake_negatives = checkpoint(
                        self.layers[i], compressed, is_real.bool(), None, False, fake_negatives, use_reentrant=False)
            else:
                if i == 0:
                    denoised, is_real_inferred, compressed, losses, reconstructed_docs, is_real, fake_negatives = self.layers[i](
                        token_embeddings, is_real_position, token_ids=texts)
                else:
                    _, _, compressed, losses, _, is_real, fake_negatives = self.layers[i](
                        compressed, is_real.bool(), fake_negatives=fake_negatives)

            leveled_compressed.append(compressed)
            leveled_losses.append(losses)

        return leveled_compressed, leveled_losses
