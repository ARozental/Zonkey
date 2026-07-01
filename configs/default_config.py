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
    # LR schedule (manual optimization → applied per optimizer step by PlZonkey).
    WARMUP_STEPS = 1000          # linear warmup over this many optimizer steps
    LR_DECAY_STEPS = 300000      # cosine-decay horizon in optimizer steps (gentle if large)
    MIN_LR_RATIO = 0.1           # final LR = LEARNING_RATE * MIN_LR_RATIO
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
    USE_OPTIMIZER_CHECKPOINT = True #use checkpoint when available to restore optimizer state

    # EMA of weights — generation samples from the EMA copy (much more coherent).
    USE_EMA = True
    EMA_DECAY = 0.999
    EMA_UPDATE_EVERY = 1   # update EMA every N optimizer steps (raise to cut CPU<->GPU copies)



    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TB_WRITER = None
    NUM_WORKERS = 12
    USE_GRADIENT_CHECKPOINTING = False
    MAX_STEPS = None
    MAX_EPOCHS = 1
    SAVE_EVERY_N_STEPS = 10000
    SAVE_TOP_K = -1  # -1 keeps every checkpoint; set to e.g. 3 to keep only the latest few
    PRINT_EVERY_N_STEPS = 10000
    EVAL_DIFFUSION_STEPS = 30  # ODE sampling steps for the debug/print generation block
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
    # FM-pass t sampling: t = U(0,1)^T_FM_EXPONENT. Slerp keeps cos(x_t,x1)=cos(t*pi/2),
    # so uniform t spends half of training above 0.71 cosine (too easy). Exponent<1
    # shifts mass toward high noise (0.5 -> density 2t, median t~0.71).
    T_FM_EXPONENT = 0.5
    # Dirty pass: t_mid ~ Beta(1,3) (informative intermediates), t_dirty ~ U(0,DIRTY_T_MAX)
    # so self-conditioned refinement is trained across the whole noise range.
    DIRTY_T_MAX = 1.0
    # Hierarchical generation: decode upper-level outputs at this truthful flow-time
    # instead of pretending they're clean t=0 vectors (they're slightly off-manifold,
    # which is exactly the regime the dirty pass trains).
    CROSS_LEVEL_DECODE_T = 0.1
    # Self-conditioning: feed the model's previous x1 estimate as an extra prompt token
    # (dirty pass during training, previous ODE step at sampling). Arch token always
    # exists; this flag only controls whether real estimates are fed (vs the null token).
    USE_SELF_COND = True
    MLM_WEIGHT = [0.0,2.0] #0.0 for token level always 
    DIRTY_MLM_WEIGHT = [1.0, 1.0] #mirrors MLM_WEIGHT; 0.0 for token level
    DECODER_MLM_WEIGHT = [0.6, 0.4]
    EXISTS_WEIGHT = [0.05,0.05] 
    
