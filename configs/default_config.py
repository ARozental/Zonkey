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
    DROPOUT = 0.0
    MAX_SEQ_LENGTHS = [16,32]
    COMPRESSION_PENALTY = [3,3] #trades off compression and quality
    COVERAGE_WEIGHT = [0.1,0.2]
    LEVEL_LOSS_WEIGHT = [1,1]
    WORKING_LEVEL_LOSS_WEIGHT = 1.0
    EPS = 1e-7
    EOS_TARGET_BIAS = [-2.0,-2.0,-2.0] #this is a hyperparameter, for slightly better initialization
    USE_MUON = False  # Use Muon optimizer for hidden layers, otherwise use AdamW for all parameters
    MUON_MOMENTUM = 0.95  # Momentum for Muon optimizer
    USE_OPTIMIZER_CHECKPOINT = True #use checkpoint when available to restore optimizer state



    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TB_WRITER = None
    NUM_WORKERS = 12
    USE_GRADIENT_CHECKPOINTING = False
    MAX_STEPS = None
    MAX_EPOCHS = 1
    SAVE_EVERY_N_STEPS = 10000
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
    MLM_WEIGHT = [0.0,2.0] #0.0 for token level always 
    DECODER_MLM_WEIGHT = [0.6, 0.4]
    EXISTS_WEIGHT = [0.05,0.05] 
    
