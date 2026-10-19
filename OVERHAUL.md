# Overhaul (branch `overhaul`)

## What changed

Same method, faster:
- Batched Muon/Adam step (`FAST_OPTIMIZER`, default on): bit-identical updates, same optimizer state.
- `GRADIENT_CHECKPOINT_SCOPE: "stack_layers"`: per-layer checkpointing inside the compiled stacks.
  Pass-level checkpointing with compiled stacks crashed at batch >= 6 (CheckpointError).
- MLM and level >= 1 reconstruction losses no longer build all-pairs similarity matrices (O(batch^2),
  8 GB at batch 24): masked positions only, and `gathered_dot` (losses/reconstruction.py) computes
  only the scored entries. torch.gather saves its whole input for backward, so the row-chunked
  version still kept every chunk until backward (3-level batch 36 worst case 27.0 -> 17.6 GB).
- The stitcher's generation-only buffer (MAX_SEQUENCES_PER_BATCH x MAX_DOC_LENGTHS x d: 3.5 GB of
  zeros at level 0 with 1024-char documents) is allocated per call instead.
- bf16 (`STACK_BF16`) exists but stays off: from scratch it brings back the "eeee" tails.
- `scripts/train.py` allocates in expandable segments; the memory figures below assume it (with the
  default allocator a 24 GB peak reserved ~30 GB and WDDM started paging).

Minimal method (switches; defaults keep the old full method):
- `STITCHER_REFINE/STITCHER_SEQUENCE_LOSS/STITCHER_NULL_LOSSES: false`: only the stitcher's cut and its
  position loss remain (the refinement changed no decode of the 2.09M model).
- The hard negatives stay, as in the 2.09M run: `MLM_ALT_PASSES: 2` (context guesses for half the
  children), `MINED_NEGATIVES_K: 16`, `NUM_FAKE_NEGATIVES: [8, 0, ...]`. Without them the
  reconstruction losses go to ~0 while the model does not learn to generate.
- `FM_PASS: false`, `DIRTY_TRUE_CODE_PROB: 0.5`, `DIRTY_FLOW_LOSS: true`: one generative pass (the
  DDMM pass, half of its samples centered on the true code with the null self-condition, flow loss).
- `MLM_LOSSES: true` stays: without it level 1 learns far worse.

Three levels:
- A level that keeps its child's width reads each child code as its CV vectors (CV tokens per child);
  boundaries only between children; all losses compare whole children, and the hard negatives
  (context guesses, mined neighbours) are whole children too.
- `FREEZE_LEVELS_UNTIL`: a level only encodes (no losses, no updates) until a step. A new level must
  learn on frozen trained levels first: trained jointly from the start it wrecks them.
- `LEVEL_LR_SCALE`: per-level learning-rate multipliers.

## Runs

The run configs train for up to 4 epochs (`MAX_EPOCHS`, as in `single_gpu_config.json`; the default
of 1 would stop a batch-48 run after ~127k steps) and save every 10k steps (the first one every
25k). The memory figures of
the configs below the first were measured with the hard negatives off and before `gathered_dot`;
both together lower them.

Three levels from scratch, all levels training from step 0, half width (128 / 512 / 512), hard
negatives on, documents of 2048 characters (63% of articles fit whole, 42% at 1024; per-document
caps 272 / 44 / 12), batch 38. `MAX_SEQUENCES_PER_BATCH` [8160, 1320, 360] is the memory budget:
worst case 24.9 GB from `--calibrate`, ~28.5 GB on nvidia-smi with optimizer and EMA. An average
batch fills ~78% of it; only a batch ~3 sigma above average overflows, and then its last documents
skip that step. The LR decays over the whole 4 epochs (640k steps):

    python scripts/train.py --param_file configs/overhaul_l3_scratch.json

Start a new architecture from the 2.09M checkpoint (its EMA weights, matched by name):

    python scripts/ema_to_weights.py checkpoints/20261018-074944/epoch=1-step=2090000.ckpt init_2090k_ema.ckpt

Three levels, stage 1 (level 2 on frozen levels 0-1, batch 48, ~17 GB):

    python scripts/train.py --param_file configs/overhaul_l3_stage1.json --resume init_2090k_ema.ckpt --load_weights_only

Stage 2 (optional, all levels, levels 0-1 at 0.1x LR, batch 12, ~25 GB), from a stage-1 checkpoint:

    python scripts/train.py --param_file configs/overhaul_l3_stage2.json --resume <stage-1 ckpt> --load_weights_only

Two levels from scratch, minimal method, windows of 16 positions, batch 24 (~19.3 GB peak):

    python scripts/train.py --param_file configs/overhaul_l2.json

(Windows of 16 instead of 27: at 3k steps, batch 24, child char/word accuracy 0.52/0.41 vs
0.29/0.00, 18% less memory, 13% faster. The 2.09M decoders are sized for 27, so this is for runs
from scratch.)

Two levels from scratch at half width (128 / 512, 61M parameters), batch 36 (~25 GB peak): the best
level-1 numbers of the configs tried at 3k steps, at twice the throughput of full width:

    python scripts/train.py --param_file configs/overhaul_l2_half.json

Level 0 codes need far fewer than their 1024 dimensions (99.4% of words survive projection onto
256 principal directions with the unchanged decoder, 95% onto 128); level 1 codes need ~1024-2048
of 4096. So 4 x 128 per word and 4 x 512 per phrase is enough capacity.

## Evaluation and baselines

    python scripts/evaluate.py <ckpt> --param_file <its config>   # generation word rates, held-out reconstruction
    python scripts/ar_baseline.py --out ar_baseline.json           # autoregressive baseline: same data, batch, optimizer
    python scripts/capacity_test.py <2-level ckpt>                 # how many dimensions the codes need (PCA projection)

Word rate = share of generated words that occur in a vocabulary of real words (`utils/evaluation.py`);
real text scores ~0.93 (all words) / 0.90 (words of 4+ letters).

Tests (CPU): `tests/test_overhaul.py`, `tests/test_fast_optimizer.py` and the existing ones.
