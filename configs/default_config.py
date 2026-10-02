import torch
class Config:
    #dimensions 
    TOKENIZER_VOCAB_SIZE_CHARS = 256
    TOKEN_EMBEDDING_SIZE = 256
    COMPRESSION_VECTORS = [4,4]
    AGENT_LEVELS = 2 
    D_MODEL = [TOKEN_EMBEDDING_SIZE,TOKEN_EMBEDDING_SIZE*COMPRESSION_VECTORS[0]]
    NUM_HEADS = [4,4]
    NUM_SPLITTER_LAYERS = [2,2]
    SPLITTER_WINDOW_SIZE = [16,16]
    NUM_UPWARD_LAYERS = [10,10] #compressor
    NUM_DECOMPRESSOR_LAYERS = [2,2]
    NUM_DENOISER_LAYERS = [10,10]
    FF_DIM_RATIO = 4
    GATED_ATTENTION = True
    POS = "sinusoidal" #"rope" or "sinusoidal" (sinusoidal is applied only to Q and K, not embeddings)

    
    MAX_DOC_LENGTHS = [1024,128,32]
    MAX_SEQUENCES_PER_BATCH = [512,64,4] #this is just to avoid oom issues [word-like, sentence-like, paragraph-like] max in batch
    
    
    #learning
    BATCH_SIZE = 3
    LEARNING_RATE = 3e-4
    # LR schedule (manual optimization → applied per optimizer step by PlZonkey._lr_factor).
    # LEARNING_RATE and MUON_LR are PEAK values: do not lower them by hand, the schedule
    # decays both groups. factor = warmup(steps since this run started) * cosine(schedule step),
    # where schedule step = global_step + offset. A full resume restores global_step, so the
    # schedule continues where it stopped; a weights-only resume (Lightning restarts
    # global_step at 0) takes the offset from the checkpoint's saved schedule step
    # automatically (run_trainer.py), so it continues too.
    LR_SCHEDULE = "cosine"       # "cosine" or "constant"
    WARMUP_STEPS = 1000          # linear warmup at the start of a fresh run
    RESUME_WARMUP_STEPS = 500    # short re-warmup after any resume (restored optimizer state can be stale)
    LR_DECAY_STEPS = 300000      # cosine horizon in schedule steps; flat at MIN_LR_RATIO afterwards
    MIN_LR_RATIO = 0.1           # final LR = peak * MIN_LR_RATIO
    LR_SCHEDULE_OFFSET = 0       # set only to override the automatic weights-only resume offset
    DROPOUT = 0.0
    MAX_SEQ_LENGTHS = [16,32]
    COMPRESSION_PENALTY = [3,3] #trades off compression and quality
    COVERAGE_WEIGHT = [0.1,0.2]
    LEVEL_LOSS_WEIGHT = [1,1]
    WORKING_LEVEL_LOSS_WEIGHT = 1.0
    EPS = 1e-7
    EOS_TARGET_BIAS = [-2.0,-2.0,-2.0] #this is a hyperparameter, for slightly better initialization
    USE_MUON = False  # Use Muon optimizer for hidden layers, otherwise use AdamW for all parameters
    # Muon lr is in SPECTRAL-NORM units (muon.py default 0.02, reference setups 0.02-0.05).
    # Passing the Adam-scale LEARNING_RATE here froze every hidden matrix ~100x too slow.
    # Conservative value given tiny batches; only the Adam group uses LEARNING_RATE.
    MUON_LR = 0.005
    MUON_MOMENTUM = 0.95  # Momentum for Muon optimizer
    # Decoupled weight decay on the Muon (hidden-matrix) group only; scaled by the scheduled LR.
    # Without it Muon's fixed-size updates grew hidden matrices to 10-40x their init RMS by 885k.
    MUON_WEIGHT_DECAY = 0.01
    USE_OPTIMIZER_CHECKPOINT = True #use checkpoint when available to restore optimizer state

    # EMA of weights — generation samples from the EMA copy (much more coherent).
    USE_EMA = True
    EMA_DECAY = 0.999
    EMA_UPDATE_EVERY = 1   # update EMA every N optimizer steps (raise to cut CPU<->GPU copies)



    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TB_WRITER = None
    NUM_WORKERS = 12
    USE_GRADIENT_CHECKPOINTING = False
    # What USE_GRADIENT_CHECKPOINTING recomputes: "all" (each level's whole forward AND each
    # denoise pass inside it, i.e. denoise passes run three times), "levels" (whole levels
    # only) or "passes" (each denoise pass only).
    GRADIENT_CHECKPOINT_SCOPE = "all"
    EMPTY_CACHE_EVERY_N_STEPS = 4  # torch.cuda.empty_cache() cadence in training_step
    MAX_STEPS = None
    MAX_EPOCHS = 1
    SAVE_EVERY_N_STEPS = 10000
    SAVE_TOP_K = -1  # -1 keeps every checkpoint; set to e.g. 3 to keep only the latest few
    PRINT_EVERY_N_STEPS = 10000
    GRAD_CLIP_VAL = 0
    GRAD_ACCUMULATION_STEPS = 1  # Number of steps to accumulate gradients (1 = no accumulation)
    PRECISION = "32-true"  # Lightning precision: "32-true", "16-mixed", "bf16-mixed"
    

    #losses
    COMPRESSED_SIM_WEIGHT = [1,1] #anti mode collapse loss weight
    DRIFTING_WEIGHT = [1.0,1.0]
    DRIFTING_TEMPERATURE = [0.05, 0.1, 0.5] #remove later
    BETA = [2.5, 3.2] #for noise schedule
    DRIFTING_NUM_RANDOM = [128,32]
    DRIFTING_QUEUE_SIZE = [8192, 2048]
    NUM_FAKE_NEGATIVES = [0, 0]  # chimeric compressed vectors created for the level above
    NOISE_STEP_SIZE = [0.05,0.05]
    NOISE_LAST_STEP_SIZE = [0.05,0.05]
    DIRTY_RECONSTRUCTION_WEIGHT = [2.5,2.5]
    CLEAN_RECONSTRUCTION_WEIGHT = [0.4,0.4]
    # High-noise regression blend: loss = (1-t^p)*contrastive + (t^p)*(1-cos_to_target).
    # p≈4 makes it negligible at low noise and dominant near t=1; set p>=10 to disable.
    REGRESSION_T_POWER = 4.0
    # FM/dirty noise sampling: schedule position u = U(0,1)^T_FM_EXPONENT, mapped to each
    # level's t by ZonkeyLayer.u_to_t. Exponent<1 shifts mass toward high noise.
    T_FM_EXPONENT = 0.5
    # Noise schedule in effective SNR rho = sqrt(D) * cot(t * pi / 2): schedule position u
    # maps to rho = NOISE_SNR_MID * cot(u * pi / 2) at every level, so wider levels (larger
    # D) get proportionally more noise for the same u. 32 = identity at D = 1024 (level 0).
    NOISE_SNR_MID = 32.0
    # The one step count shared by training and sampling: the self-conditioning training
    # step is 1/DIFFUSION_STEPS in u, exactly one sampler step of generate().
    DIFFUSION_STEPS = 30
    # Self-conditioning: feed the model's previous x1 estimate as an extra prompt token
    # (dirty pass during training, previous ODE step at sampling). Arch token always
    # exists; this flag only controls whether real estimates are fed (vs the null token).
    USE_SELF_COND = True
    MLM_WEIGHT = [0.0,2.0]
    DIRTY_MLM_WEIGHT = [1.0, 1.0]
    DECODER_MLM_WEIGHT = [0.6, 0.4]
    EXISTS_WEIGHT = [0.05,0.05]
    # Per-sample gate on sequence reconstruction for FM/dirty (not BOS, not clean).
    # Multiplies recon by (1-t)^p so high-t is not exact-sequence CE (barycenter).
    SEQUENCE_RECON_T_GATE_POWER = 2.0
    # Spherical CFM on log-displacements (t²-weighted velocity MSE). The sampler
    # still integrates log(x_t, x0)/t; we do not divide by t in the loss.
    FLOW_VELOCITY_WEIGHT = [1.0, 1.0]
    # Always-on (1-cos) added to L>0 contrastive recon. InfoNCE+atanh can still
    # saturate vs easy negatives before the positive is on-manifold.
    DIRECT_COSINE_WEIGHT = 0.5
    # One-hop interface loss: the parent's predicted child codes are decoded by the child
    # level exactly as generation does, and scored with the child level's own
    # reconstruction loss against the decode of the true child codes. Never unrolls more
    # than one level, so it is the same code at every depth. Parent term: child frozen,
    # weighted by the parent's probability of the true child. Child term: parent detached,
    # only where the parent's nearest real child code has the true content id.
    INTERFACE_CONSISTENCY_WEIGHT = [0.0, 1.0]
    INTERFACE_CONSISTENCY_SAMPLES = 64

    # Harder negatives for the clean-pass reconstruction at levels >= 1 (level 0 already
    # scores its whole char vocabulary). MLM alternatives: MLM_ALT_PASSES complementary
    # masked passes, each masking MLM_ALT_FRACTION of the slots, give one context guess per
    # masked slot. Mined negatives: the MINED_NEGATIVES_K real child codes nearest to the
    # prediction whose content id differs from the target's.
    MLM_ALT_PASSES = 2
    MLM_ALT_FRACTION = 0.25
    MINED_NEGATIVES_K = 16

    # Code margin. The clean pass decodes from the code moved toward random noise by
    # t_aug along the geodesic (angle = t_aug * 90 degrees) while the time label stays 0,
    # so "t=0" means "a code that is near a real code" and every decoder learns to read
    # its code from directions that survive small errors (a parent's prediction error, or
    # the sampler's end point). t_aug is log-uniform in the range (scale-free), and
    # CLEAN_NOISE_EXACT_FRACTION of the samples stay exact. Isotropic noise of angle a has
    # only a/sqrt(D) along any single direction, so the same range is never harder at
    # higher levels (larger D); the invariance it teaches depends on the angle only.
    CLEAN_NOISE_T_RANGE = [1e-4, 3e-2]
    CLEAN_NOISE_EXACT_FRACTION = 0.25
    # On-manifold margin (ZonkeyLayer._neighbor_margin): this fraction of the noised clean
    # samples moves toward the nearest real code with a different text instead, by a random
    # fraction (at most CLEAN_NEIGHBOR_MAX_STEP < 0.5) of the angle between them, so the own
    # code stays the nearest one. Isotropic noise alone never trains these directions.
    CLEAN_NEIGHBOR_FRACTION = 0.5
    CLEAN_NEIGHBOR_MAX_STEP = 0.4

    # Generative passes treat codes as data: the FM input/target and the FM/dirty
    # reconstruction targets (child codes, or the char table at level 0) are detached, so
    # the flow loss cannot pull codes toward the denoiser's average prediction.
    DETACH_GENERATIVE_TARGETS = True

    # FM/dirty reconstruction is weighted per sample by the probability that the
    # denoiser's own estimate (compress(denoised)) picks its source among the batch's
    # codes (identical owned content counts as the same source). Unidentifiable inputs
    # can only teach the average output ("eeee"), so they get no reconstruction weight.
    IDENTIFIABILITY_GATE = True

    # Draft = one causal pass of the decompressor over [prompt; learned queries],
    # normalized to dim_norm. The old 27-step self-feeding unroll produced a
    # content-free draft whose norm grew with position (9.5k -> 22k at L0).
    PARALLEL_DRAFT = True

    # Segment-length targets per level. None keeps the old coupling to
    # COMPRESSION_VECTORS (target BOS prob = CV/MAX_SEQ_LENGTH, min length = CV); set
    # them explicitly for upper levels that use COMPRESSION_VECTORS = 1.
    TARGET_BOS_PROB = [None, None, None, None, None, None]
    MIN_SEGMENT_LENGTH = [None, None, None, None, None, None]

    # Cheap no-grad diagnostics logged as level_n/metric_* every N optimizer steps.
    DIAGNOSTICS_EVERY_N_STEPS = 250

