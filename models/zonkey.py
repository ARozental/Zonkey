import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import sys
import os
import io
import math
import contextlib
from torch.utils.checkpoint import checkpoint
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.default_config import Config
from models.zonkey_layer import ZonkeyLayer
from utils.content_id import prefix_hash_tensor, pow_table
from muon import SingleDeviceMuonWithAuxAdam, MuonWithAuxAdam
import torch.distributed as dist

# Detached / logging-only keys. Including them in the mean rescales real losses
# without training anything (avg_bos_prob is .detach()'d).
METRIC_LOSS_KEYS = frozenset({"avg_bos_prob"})


def _as_loss_tensor(value):
    if isinstance(value, torch.Tensor):
        t = value
    elif hasattr(value, "tensor"):
        t = value.tensor
    else:
        return None
    return t if t.ndim == 0 else t.mean()


def aggregate_leveled_losses(leveled_losses):
    """Mean of non-metric losses per level, then LEVEL_LOSS_WEIGHT[l]. Scale of the
    per-level mean is preserved (not a sum) so the effective LR does not jump."""
    total = None
    level_totals = []
    weights = getattr(Config, "LEVEL_LOSS_WEIGHT", None)
    for l, losses in enumerate(leveled_losses):
        vals = []
        for name, value in losses.items():
            if name in METRIC_LOSS_KEYS or name.startswith("metric_"):
                continue
            t = _as_loss_tensor(value)
            if t is not None:
                vals.append(t)
        if not vals:
            level_totals.append(None)
            continue
        lvl = torch.stack(vals).mean()
        if weights is not None and l < len(weights):
            lvl = lvl * weights[l]
        level_totals.append(lvl)
        total = lvl if total is None else total + lvl
    if total is None:
        raise RuntimeError("aggregate_leveled_losses: no loss tensors")
    return total, level_totals


@torch.no_grad()
def aggregate_progress(leveled_losses):
    """loss/progress: aggregated exactly like aggregate_leveled_losses (same terms, same
    per-level mean and LEVEL_LOSS_WEIGHT), except that each term with a "metric_progress_<term>"
    entry uses that value instead. Those are the terms that grow harder or heavier as the model
    improves (gated and extra-negative reconstructions, p/right_pick-weighted interface), scored
    on a fixed task, so this number going down means the model is still learning."""
    total = None
    level_totals = []
    weights = getattr(Config, "LEVEL_LOSS_WEIGHT", None)
    for l, losses in enumerate(leveled_losses):
        vals = []
        for name, value in losses.items():
            if name in METRIC_LOSS_KEYS or name.startswith("metric_"):
                continue
            t = _as_loss_tensor(losses.get("metric_progress_" + name, value))
            if t is not None:
                vals.append(t.detach())
        if not vals:
            level_totals.append(None)
            continue
        lvl = torch.stack(vals).mean()
        if weights is not None and l < len(weights):
            lvl = lvl * weights[l]
        level_totals.append(lvl)
        total = lvl if total is None else total + lvl
    return total, level_totals

class PlZonkey(pl.LightningModule):
    def __init__(self, writer=None):
        super().__init__()
        self.automatic_optimization = False
        self.model = Zonkey()
        self.model.compile()
        if getattr(Config, "COMPILE_TRANSFORMER_STACKS", False):
            # The denoise passes run outside dynamo (torch.utils.checkpoint disables it for
            # everything it calls, and in "passes" scope the pass is wrapped in
            # torch._dynamo.disable), so the stacks inside them ran uncompiled, about half of
            # all compute. A module compiled in place still runs compiled there; called from
            # an already compiled frame it is simply inlined. Parameter names are unchanged.
            for layer in self.model.layers:
                for stack in (layer.compressor, layer.decompressor, layer.denoiser):
                    stack.compile()
            # Every stack shares nn.Module._call_impl, and each level, grad mode and batch
            # shape is its own compiled variant: the default limit of 8 per frame would be
            # used up and the rest would silently run uncompiled.
            torch._dynamo.config.recompile_limit = max(int(torch._dynamo.config.recompile_limit), 64)
        self.tb_writer = writer
        # EMA of weights (generation samples from the EMA copy). Stored on CPU to
        # keep GPU memory free; updated in place (one persistent buffer, not realloc).
        self.use_ema = bool(getattr(Config, "USE_EMA", False))
        self.ema_decay = float(getattr(Config, "EMA_DECAY", 0.999))
        self.ema_update_every = max(1, int(getattr(Config, "EMA_UPDATE_EVERY", 1)))
        self._ema = None  # lazily initialized list parallel to trainable params
        # Added to global_step for the LR schedule and TensorBoard steps. run_trainer sets it
        # to the checkpoint's step on a weights-only resume (Lightning restarts global_step at 0).
        self._schedule_offset = int(getattr(Config, "LR_SCHEDULE_OFFSET", 0) or 0)
        self._run_start_step = 0

    def _ema_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def load_ema_state(self, ema_state):
        """Restore EMA weights only if they match the current parameter list exactly;
        otherwise (architecture changed) the EMA restarts from the current weights."""
        params = self._ema_params()
        if (ema_state is not None and len(ema_state) == len(params)
                and all(tuple(e.shape) == tuple(p.shape) for e, p in zip(ema_state, params))):
            self._ema = [e.clone() for e in ema_state]
            return True
        if ema_state is not None:
            print("EMA state does not match the current parameters; EMA restarts from the current weights")
        self._ema = None
        return False

    def _schedule_step(self):
        return int(self.global_step) + int(self._schedule_offset)

    @torch.no_grad()
    def _ema_update(self):
        # EMA lives on the parameters' device (~1 GB at 258M params). The old CPU copy moved
        # every parameter GPU->CPU on every step. The decay is compounded over
        # EMA_UPDATE_EVERY so the EMA horizon does not depend on the update interval.
        params = [p.detach() for p in self._ema_params()]
        if self._ema is None:
            self._ema = [p.float().clone() for p in params]
            return
        if self._ema[0].device != params[0].device:
            self._ema = [e.to(device=p.device, dtype=torch.float32) for e, p in zip(self._ema, params)]
        weight = 1.0 - self.ema_decay ** self.ema_update_every
        torch._foreach_lerp_(self._ema, params, weight)

    @contextlib.contextmanager
    def _ema_swapped(self):
        """Temporarily load EMA weights into the model (for generation), then restore."""
        if not self.use_ema or self._ema is None:
            yield
            return
        params = self._ema_params()
        backup = [p.detach().clone() for p in params]
        try:
            for p, e in zip(params, self._ema):
                p.data.copy_(e.to(device=p.device, dtype=p.dtype))
            yield
        finally:
            for p, b in zip(params, backup):
                p.data.copy_(b)

    def on_save_checkpoint(self, checkpoint):
        if self.use_ema and self._ema is not None:
            checkpoint["ema_state"] = [e.detach().cpu() for e in self._ema]
        checkpoint["optimizer_layout_version"] = 2
        # The schedule step at save time, so a weights-only resume can continue the LR curve.
        checkpoint["schedule_step"] = self._schedule_step()

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
            total_loss, _ = aggregate_leveled_losses(leveled_losses)

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
        # Warmup restarts on EVERY run start (fresh or resumed): restored optimizer state
        # can be stale, so the LR is eased back in (RESUME_WARMUP_STEPS after a resume).
        self._run_start_step = int(self.global_step)
        factor = self._lr_factor()
        for opt in self.trainer.optimizers:
            for group in opt.param_groups:
                # Lightning restored optimizer state clobbers base_lr with the OLD
                # checkpoint value; re-assert it from the CURRENT config (peak LRs).
                group["base_lr"] = (Config.MUON_LR if group.get("use_muon")
                                    else Config.LEARNING_RATE)
                group["lr"] = group["base_lr"] * factor
        print(f"LR schedule: {getattr(Config, 'LR_SCHEDULE', 'cosine')}, schedule step {self._schedule_step()}, "
              f"factor {factor:.4f}, horizon {Config.LR_DECAY_STEPS}, floor {Config.MIN_LR_RATIO}")

    def _lr_factor(self):
        """LR multiplier for both optimizer groups (Muon and Adam keep their own base LR).

        warmup: linear over the first steps of THIS run (WARMUP_STEPS for a fresh run,
        RESUME_WARMUP_STEPS after a resume), times the schedule evaluated at the schedule
        step (global_step + offset), so a resumed run continues the same curve instead of
        restarting it. cosine: from 1 down to MIN_LR_RATIO over LR_DECAY_STEPS, then flat.
        """
        since_start = int(self.global_step) - int(getattr(self, "_run_start_step", 0))
        resumed = (int(getattr(self, "_run_start_step", 0)) + int(self._schedule_offset)) > 0
        warmup = int(getattr(Config, "RESUME_WARMUP_STEPS", 0)) if resumed else int(getattr(Config, "WARMUP_STEPS", 0))
        warm = min(1.0, (since_start + 1) / max(1, warmup))
        if str(getattr(Config, "LR_SCHEDULE", "cosine")).lower() == "constant":
            return warm
        progress = min(1.0, self._schedule_step() / max(1, int(Config.LR_DECAY_STEPS)))
        floor = float(Config.MIN_LR_RATIO)
        cosine = floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return warm * cosine

    def _warmup_factor(self):
        # Kept for backward compatibility; the schedule lives in _lr_factor.
        return self._lr_factor()

    def on_load_checkpoint(self, checkpoint):
        """Validate optimizer compatibility and restore current learning rates."""
        # Check if optimizer type has changed
        optimizer_states_present = bool(checkpoint.get("optimizer_states"))
        checkpoint_had_muon = False
        if optimizer_states_present:
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
        optimizer_layout_changed = checkpoint.get("optimizer_layout_version", 1) != 2
        
        if optimizer_states_present and (
            checkpoint_had_muon != current_uses_muon or optimizer_layout_changed
        ):
            raise RuntimeError(
                "Checkpoint optimizer layout is incompatible. Use scripts/run_trainer.py "
                "with --load_weights_only; run_trainer also selects this automatically."
            )

        # Overwrite the LR baked into the restored optimizer state with the CURRENT
        # config value. PyTorch's optimizer.load_state_dict() replaces param_groups
        # wholesale (keeping only `params`), so a stale `lr`/`base_lr` from the
        # checkpoint would otherwise silently override a changed config on resume.
        # Mutating the checkpoint dict here is ordering-independent — it always runs
        # before the restore applies.
        if 'optimizer_states' in checkpoint:
            for opt_state in checkpoint['optimizer_states']:
                for group in opt_state.get('param_groups', []):
                    new_lr = Config.MUON_LR if group.get('use_muon') else Config.LEARNING_RATE
                    group['lr'] = new_lr
                    group['base_lr'] = new_lr

        # Restore EMA weights if present (so EMA survives stop/resume).
        if self.use_ema and checkpoint.get("ema_state") is not None:
            self.load_ema_state(checkpoint["ema_state"])

        # Keep the LR schedule continuous if this checkpoint itself came from a
        # weights-only resume (its global_step restarted at 0 while the schedule did not).
        if "schedule_step" in checkpoint and not int(getattr(Config, "LR_SCHEDULE_OFFSET", 0) or 0):
            self._schedule_offset = max(0, int(checkpoint["schedule_step"]) - int(checkpoint.get("global_step", 0) or 0))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()

        log_step = self._schedule_step()
        diag_every = int(getattr(Config, "DIAGNOSTICS_EVERY_N_STEPS", 0) or 0)
        diag = diag_every > 0 and log_step % diag_every == 0
        for layer in self.model.layers:
            layer.diagnostics_this_step = diag

        leveled_compressed, leveled_losses = self.model.forward(batch)

        _print_step = None
        if getattr(self, "trainer", None) is not None:
            _print_step = int(self.trainer.global_step)
        else:
            _print_step = int(self.global_step)
        if _print_step <= 0:
            _print_step = int(batch_idx)

        if (self.global_step % int(getattr(Config, "EMPTY_CACHE_EVERY_N_STEPS", 4)) == 0):
            torch.cuda.empty_cache()  # Free fragmented memory

        should_log = (self.global_step % 5 == 0)
        total_loss, level_totals = aggregate_leveled_losses(leveled_losses)

        for l, losses in enumerate(leveled_losses):
            if (should_log or diag) and self.tb_writer is not None:
                for name, value in losses.items():
                    t = _as_loss_tensor(value)
                    if t is None:
                        continue
                    self.tb_writer.add_scalar(f"level_{l}/{name}", float(t.item()), log_step)
            if self.tb_writer is not None and level_totals[l] is not None:
                self.tb_writer.add_scalar(f"loss/_{l}", float(level_totals[l].item()), log_step)
        if diag and self.tb_writer is not None:
            # Hidden-matrix RMS per module: should flatten out with MUON_WEIGHT_DECAY
            # (without decay it grew to 10-40x the init scale by 885k).
            with torch.no_grad():
                for l, layer in enumerate(self.model.layers):
                    for name in ("compressor", "denoiser", "decompressor"):
                        rms = [p.detach().float().pow(2).mean().sqrt()
                               for p in getattr(layer, name).parameters() if p.ndim == 2 and min(p.shape) > 1]
                        if rms:
                            self.tb_writer.add_scalar(f"weights/level_{l}_{name}_rms", float(torch.stack(rms).mean()), log_step)



        # Scale loss for gradient accumulation
        scaled_loss = total_loss / Config.GRAD_ACCUMULATION_STEPS
        self.manual_backward(scaled_loss)

        # Push this batch's clean vectors into each layer's coverage queue AFTER backward,
        # so the queue is never mutated inside the gradient-checkpointed forward (which is
        # recomputed during backward and would otherwise see a changed count -> shape error).
        for layer in self.model.layers:
            layer.push_clean_to_queue()

        # Determine if we should step the optimizer (every N accumulation steps)
        should_step = (batch_idx + 1) % Config.GRAD_ACCUMULATION_STEPS == 0

        if should_step:
            factor = self._lr_factor()
            for group in optimizer.param_groups:
                group["lr"] = group.get("base_lr", Config.LEARNING_RATE) * factor
            if Config.GRAD_CLIP_VAL > 0:
                self.clip_gradients(optimizer, gradient_clip_val=Config.GRAD_CLIP_VAL, gradient_clip_algorithm="norm")
            optimizer.step()
            optimizer.zero_grad()
            if self.use_ema and (int(self.global_step) % self.ema_update_every == 0):
                self._ema_update()

        if should_log and self.tb_writer is not None:
            self.tb_writer.add_scalar("loss/total", float(total_loss.item()), log_step)
            # The fixed-task version of loss/total (see aggregate_progress).
            progress_total, progress_levels = aggregate_progress(leveled_losses)
            self.tb_writer.add_scalar("loss/progress", float(progress_total.item()), log_step)
            for l, value in enumerate(progress_levels):
                if value is not None:
                    self.tb_writer.add_scalar(f"loss/progress_{l}", float(value.item()), log_step)
            # Learning rate of each group (Muon group first, as before; Adam group separately).
            for group in optimizer.param_groups:
                tag = "training/learning_rate" if group.get("use_muon", True) else "training/learning_rate_adam"
                self.tb_writer.add_scalar(tag, group["lr"], log_step)

        del total_loss, leveled_compressed, leveled_losses

        # Sample prints run after the optimizer step and the EMA update, so the forward's
        # activations are already freed, and the codes are re-encoded with the same (EMA)
        # weights that decode them; codes from the online forward decoded by EMA decoders
        # never occur in training or in real generation.
        self._print_samples(batch, _print_step)
        return None

    def _print_samples(self, batch, _print_step):
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

                    _real_stdout = sys.__stdout__ if sys.__stdout__ is not None else sys.stdout
                    _tb_redirect = contextlib.redirect_stdout(_Tee(_real_stdout, _tb_text_buf))
                else:
                    _tb_redirect = contextlib.nullcontext()

                with _tb_redirect, self._ema_swapped():
                    for layer in self.model.layers:
                        layer.diagnostics_this_step = False
                    leveled_compressed, _ = self.model.forward(batch)
                    # The real step already pushed its codes; these EMA codes must not be
                    # pushed into the queues at the next step.
                    for layer in self.model.layers:
                        layer._pending_clean_for_queue = None
                        layer._pending_ids_for_queue = None
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
                    print("original doc start: ", self.model.token_ids_to_text(tokens_out))

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


                    print("level 0 text 0: ", self.model.token_ids_to_text(tokens_out0))
                    print("level 0 text 1: ", self.model.token_ids_to_text(tokens_out1))
                    print("level 0 text 2: ", self.model.token_ids_to_text(tokens_out2))
                    print("is_real_inferred text 0: ",[int(10000*x)/10000 for x in is_real_inferred_final[0][0:Config.MAX_SEQ_LENGTHS[0]].tolist()])
                    print("is_real_inferred text 1: ",[int(10000*x)/10000 for x in is_real_inferred_final1[0][0:Config.MAX_SEQ_LENGTHS[0]].tolist()])
                    print("is_real_inferred text 2: ",[int(10000*x)/10000 for x in is_real_inferred_final2[0][0:Config.MAX_SEQ_LENGTHS[0]].tolist()])

                    # Generate samples from all levels
                    for level in range(len(leveled_compressed)):
                        if level > 0:
                            print(f"decompressing CLEAN from level {level} (all levels @t=0): ")
                            self.model.generate_sequence_from_level_N(
                                level, num_diffusion_steps=0,
                                fixed_compressed_vectors=leveled_compressed[level][0][0:1],
                                noise_level=0.0, existance_cutoff=0.1,
                                lower_diffusion_steps=0, lower_t=0.0,
                                print_children=True)
                        else:
                            print(f"decompressing CLEAN from level {level}: ")
                            self.model.generate_sequence_from_level_N(
                                level, fixed_compressed_vectors=leveled_compressed[level][0][0:1],
                                noise_level=0.0, existance_cutoff=0.1)

                        if len(leveled_compressed[level]) > 1 and level > 0:
                            print(f"decompressing CLEAN from level {level} s2: ")
                            self.model.generate_sequence_from_level_N(
                                level, num_diffusion_steps=0,
                                fixed_compressed_vectors=leveled_compressed[level][0][1:2],
                                noise_level=0.0, existance_cutoff=0.1,
                                lower_diffusion_steps=0, lower_t=0.0)

                        print(f"decompressing from level {level} with {Config.NOISE_LAST_STEP_SIZE[level]} noise: ")
                        # Let generate add noise exactly once to the one code being decoded.
                        # The previous diagnostic pre-noised here and generate noised again.
                        _cv = Config.COMPRESSION_VECTORS[level]
                        _one_vec = leveled_compressed[level][0][0:1].view(1, _cv, -1)
                        self.model.generate_sequence_from_level_N(
                            level, fixed_compressed_vectors=_one_vec.view(1, -1),
                            noise_level=Config.NOISE_LAST_STEP_SIZE[level], existance_cutoff=0.1,
                            lower_t=0.0)
                        print(f"random seq from level {level}: ")
                        self.model.generate_sequence_from_level_N(
                            level, num_diffusion_steps=Config.DIFFUSION_STEPS, noise_level=1.0,
                            existance_cutoff=0.1, lower_t=0.0)


                    # Clean up generation artifacts to free GPU memory
                    del sample_indices, sample_embeddings, normalized, cosine_sim_matrix, mask
                    del texts, token_embeddings, tokens_normalized, embeddings_normalized, logits, best_token_idx

                    if self.tb_writer is not None and _tb_text_buf is not None:
                        _captured = _tb_text_buf.getvalue()
                        if _captured.strip():
                            self.tb_writer.add_text("samples/generated", _captured, _print_step)

    def configure_optimizers(self):
        if Config.USE_MUON:
            # Muon is defined for 2-D hidden matrices. Conv1d kernels are 3-D and the
            # old ndim>=2 test accidentally treated them as batches of tiny matrices.
            # Output/routing heads also need AdamW rather than orthogonalized updates.
            hidden_matrix_params = []
            other_params = []
            adam_head_names = (
                "bos_layer.",
                "classification_head.",
                "signal_coherence.",
                "segment_splitter.bos_classifier.proj.",
                "stitcher.score_linear.",
                "stitcher.proj.",
            )
            
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    use_muon = (
                        p.ndim == 2
                        and min(p.shape) > 1
                        and "token_embedding_layer." not in name
                        and not any(fragment in name for fragment in adam_head_names)
                    )
                    if use_muon:
                        hidden_matrix_params.append(p)
                    else:
                        other_params.append(p)
            
            # Create parameter groups for MuonWithAuxAdam.
            # NOTE: Muon's lr is in spectral-norm units (~0.005-0.05), NOT Adam scale.
            # base_lr is stored per group; the warmup schedule multiplies it instead of
            # overwriting every group with the Adam-scale LEARNING_RATE.
            param_groups = []
            if hidden_matrix_params:
                param_groups.append(dict(params=hidden_matrix_params, lr=Config.MUON_LR,
                                        momentum=Config.MUON_MOMENTUM,
                                        weight_decay=float(getattr(Config, "MUON_WEIGHT_DECAY", 0.0)),
                                        use_muon=True))
            if other_params:
                param_groups.append(dict(params=other_params, lr=Config.LEARNING_RATE,
                                        eps=Config.EPS, use_muon=False))
            
            # Detect if we're in distributed training mode
            is_distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
            
            if is_distributed:
                optimizer = MuonWithAuxAdam(param_groups)
            else:
                optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
            # Muon's __init__ asserts exact group keys, so base_lr is attached after
            # construction. The warmup schedule multiplies base_lr per group instead of
            # overwriting every group with the (Adam-scale) LEARNING_RATE.
            for group in optimizer.param_groups:
                group["base_lr"] = Config.MUON_LR if group.get("use_muon") else Config.LEARNING_RATE
        else:
            # Use standard AdamW for all parameters
            params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=Config.LEARNING_RATE, eps=Config.EPS)
            for group in optimizer.param_groups:
                group["base_lr"] = Config.LEARNING_RATE

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



    def _clip_to_last_segment(self, doc, level, existence_mask):
        """Trim a stitched document (1, max_doc_len, d) to its real content.

        The stitcher pastes every segment up to the next one's inferred start, and the last
        segment in full (all MAX_SEQ_LENGTHS positions), after which the buffer is zeros.
        Cut the last segment at its own existence length (existence_mask[-1]), which removes
        the untruncated tail ("        eeee") that ended most generated lines, and drops the
        zero rows so a lower level never decodes them."""
        starts = self.layers[level].stitcher.last_segment_start
        start = int(starts[0]) if len(starts) > 0 else 0
        last_len = max(1, int(existence_mask[-1].sum().item()))
        end = min(doc.shape[1], start + last_len)
        return doc[:, :end, :]

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
        print(self.token_ids_to_text(tokens_out))
        return 

    @staticmethod
    def token_ids_to_text(token_ids):
        """Character decode: token id i is chr(i) (see data.wiki_chars.text_to_ids). Not UTF-8.
        0 (PAD) and 1 (END) are dropped; 2 (UNK, any foreign character) prints as U+FFFD."""
        values = token_ids.detach().reshape(-1).tolist() if torch.is_tensor(token_ids) else list(token_ids)
        return "".join(chr(0xFFFD) if int(x) == 2 else chr(int(x) % 256) for x in values if int(x) not in (0, 1))
        
    def generate_sequence_from_level_N(
        self, N, num_diffusion_steps=0, fixed_compressed_vectors=None,
        noise_level=0.0, existance_cutoff=0.1, lower_diffusion_steps=0,
        lower_t=None, print_children=False,
    ):
        #get initial sequence, 
        initial_seq, existence_mask, is_real_inferred_final =  self.layers[N].generate(
            batch_size=1,
            num_diffusion_steps=num_diffusion_steps,
            fixed_compressed_vectors=fixed_compressed_vectors,
            noise_level=torch.tensor(noise_level, dtype=torch.float32, device=Config.DEVICE),
            existance_cutoff=existance_cutoff)
        doc = initial_seq.squeeze(0)[0:existence_mask.bool().sum().item(),:] # <actual_len,d_model>
        if lower_t is None:
            # A completed parent denoise predicts clean child codes. Its structured
            # model error is not forward-diffusion noise with a known time label.
            lower_t = 0.0
        while N>0:
            N-=1
            seq, existence_mask, is_real_inferred = self.layers[N].generate(
                fixed_compressed_vectors=doc,
                noise_level=torch.full((doc.shape[0],), float(lower_t), dtype=torch.float32, device=Config.DEVICE),
                treat_as_noisy=True,
                num_diffusion_steps=lower_diffusion_steps, # zero here for no refinement by lower layers
                existance_cutoff=existance_cutoff
                )
            if print_children and N == 0:
                for child_idx in range(min(3, seq.shape[0])):
                    child_len = int(existence_mask[child_idx].sum().item())
                    print(f"  child {child_idx}: ", end="")
                    self.print_char_sequence(seq[child_idx, :child_len])
            doc,_,_ = self.layers[N].stitcher(seq, is_real_inferred, torch.tensor([seq.shape[0]], dtype=torch.long, device=seq.device))

            doc = self._clip_to_last_segment(doc, N, existence_mask)
            doc = doc.squeeze(0)
        self.print_char_sequence(doc[:100])
        return doc

    def ar_generate_sequence_from_level_N(self, N, existance_cutoff=0.1):
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
            doc = self._clip_to_last_segment(doc, level, existence_mask)
            doc = doc.squeeze(0)
        self.print_char_sequence(doc[:100])
        return doc

    def forward(self, batch):
        texts = batch["full_texts"]
        token_embeddings = self.token_embedding_layer(texts)
        is_real_position = (texts != 0)
        leveled_compressed = []
        leveled_losses = []

        # Content ids (utils/content_id.py): batches from the dataloader carry the prefix
        # hashes; synthetic batches (calibration, smoke tests) get them computed here.
        prefix_hash_b = batch.get("prefix_hash")
        if prefix_hash_b is None:
            prefix_hash_b = prefix_hash_tensor(texts)
        prefix_hash_b = prefix_hash_b.to(texts.device)
        pow_tab = getattr(self, "_pow_table_cache", None)
        if pow_tab is None or pow_tab.device != texts.device or pow_tab.shape[0] != Config.MAX_DOC_LENGTHS[0] + 1:
            pow_tab = pow_table(Config.MAX_DOC_LENGTHS[0], device=texts.device)
            self._pow_table_cache = pow_tab
        for layer in self.layers:
            layer._batch_prefix_hash = prefix_hash_b
            layer._pow_table = pow_tab

        fake_negatives = None
        child_spans, child_ids = None, None
        for i in range(len(self.layers)):
            if Config.USE_GRADIENT_CHECKPOINTING and getattr(Config, "GRADIENT_CHECKPOINT_SCOPE", "all") in ("all", "levels"):
                if i == 0:
                    denoised, is_real_inferred, compressed, losses, reconstructed_docs, is_real, fake_negatives, child_spans, child_ids = checkpoint(
                        self.layers[i], token_embeddings, is_real_position, texts, False, None, None, None, use_reentrant=False)
                else:
                    _, _, compressed, losses, _, is_real, fake_negatives, child_spans, child_ids = checkpoint(
                        self.layers[i], compressed, is_real.bool(), None, False, fake_negatives, child_spans, child_ids, use_reentrant=False)
            else:
                if i == 0:
                    denoised, is_real_inferred, compressed, losses, reconstructed_docs, is_real, fake_negatives, child_spans, child_ids = self.layers[i](
                        token_embeddings, is_real_position, token_ids=texts)
                else:
                    _, _, compressed, losses, _, is_real, fake_negatives, child_spans, child_ids = self.layers[i](
                        compressed, is_real.bool(), fake_negatives=fake_negatives,
                        child_spans=child_spans, child_ids=child_ids)

            leveled_compressed.append(compressed)
            leveled_losses.append(losses)

        return leveled_compressed, leveled_losses
