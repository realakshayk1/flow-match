# Running Flow-Match on Google Colab (L4)

This walks through running the full four-phase improvement pipeline (Phases 0–3) on a Colab
L4 GPU. Copy each block into its own Colab cell.

## What lives where

| Asset | In the GitHub repo? | How to get it onto Colab |
|-------|--------------------|--------------------------|
| Code (`src/`, `scripts/`, `tests/`) | ✅ yes | `git clone` |
| `data/splits.json` | ✅ yes | `git clone` |
| `data/processed/*.pt` (4.2 GB, 4639 files) | ❌ gitignored | Google Drive (below) |
| `checkpoints/best_model.pt` (1.8 MB) | ❌ gitignored | Google Drive (below) |

So: **code comes from GitHub, data + checkpoints come from your Google Drive.**

---

## One-time: stage data on Google Drive (run locally, once)

On your own machine, bundle the processed data and checkpoint, then upload the two files to
Google Drive (e.g. a folder `MyDrive/flowmatch/`):

```bash
# from the repo root on your laptop
tar czf processed.tar.gz data/processed          # ~1-2 GB compressed
cp checkpoints/best_model.pt best_model.pt
# upload processed.tar.gz and best_model.pt to Google Drive > flowmatch/
```

---

## Colab cell 1 — GPU check

```python
!nvidia-smi -L          # expect: GPU 0: NVIDIA L4
```
If you don't see an L4/T4/A100, go to **Runtime > Change runtime type > GPU**.

## Colab cell 2 — clone the code

```python
!git clone https://github.com/realakshayk1/flow-match.git
%cd flow-match
!git checkout claude/sad-nash-1e3a2b   # the branch with Phases 0-3; use main once merged
```

## Colab cell 3 — install dependencies

```python
# Colab ships torch + CUDA already; add the project deps.
!pip -q install torch-geometric==2.5.3 rdkit==2024.3.5 MDAnalysis==2.7.0 \
                posebusters>=0.2.7 pandas==2.2.2 tqdm wandb
# If numpy complains, pin it and restart the runtime once:
# !pip -q install "numpy<2"
```
> If a dependency forces a numpy/torch downgrade, use **Runtime > Restart runtime** once, then
> re-run from cell 2 (skip the clone).

## Colab cell 4 — mount Drive and unpack data

```python
from google.colab import drive
drive.mount('/content/drive')

DRIVE = '/content/drive/MyDrive/flowmatch'   # <-- adjust to your folder
!tar xzf {DRIVE}/processed.tar.gz -C /content/flow-match     # -> data/processed/
!mkdir -p /content/flow-match/checkpoints
!cp {DRIVE}/best_model.pt /content/flow-match/checkpoints/best_model.pt
!ls data/processed | wc -l        # expect 4639
```

## Colab cell 5 — sanity checks (fast)

```python
!python scripts/verify_equivariance.py     # 9/9 checks
!python -m pytest tests/ -q                 # 57 passed
```

---

## Phase 0 — corrected, honest headline (no retrain, ~minutes)

Re-score the existing 820K checkpoint with the in-frame, symmetry-corrected RMSD and the joint
`RMSD<2Å & PB-valid` metric.

```python
!python scripts/eval_posebusters.py \
    --checkpoint checkpoints/best_model.pt \
    --split test --n_inference_steps 50 --relax none \
    --output_dir results/phase0_raw --device cuda
```
Read `dock_rmsd_*` and `rmsd_lt2_and_pbvalid_pct` in the printout — these replace the old
86.4 % / 0.835 Å shape numbers.

---

## Phase 1 — physical validity

### 1a. Relaxation only (no retrain)

```python
!python scripts/eval_posebusters.py \
    --checkpoint checkpoints/best_model.pt \
    --split test --n_inference_steps 50 --relax pocket \
    --output_dir results/phase1_relaxed --device cuda
```
Compare PB-valid % in `results/phase1_relaxed` vs `results/phase0_raw`.

### 1b. Retrain with the bonded-geometry auxiliary loss (~hours)

```python
!WANDB_MODE=offline python -m src.training.train \
    --profile l4 --geom_loss_weight 0.1 --n_epochs 100 \
    --checkpoint_dir checkpoints/geom
```
Then evaluate the new checkpoint with the bundle (raw vs pocket-relaxed in one shot):

```python
!python scripts/run_phase1_bundle.py \
    --checkpoint checkpoints/geom/best_model.pt \
    --out_root results/phase1 --split test --n_inference_steps 50 --relax pocket
```

---

## Phase 2 — multi-pose + confidence ranking

Train a confidence head on the trained base model, then run ranked evaluation.

```python
# Train the confidence head (frozen trunk). Use the best base checkpoint you have.
!WANDB_MODE=offline python scripts/train_confidence.py \
    --base_checkpoint checkpoints/geom/best_model.pt \
    --out_checkpoint checkpoints/best_model_conf.pt \
    --split train --n_poses 8 --n_steps 20 --epochs 5 --device cuda

# Ranked eval: top-1 (confidence) vs top-1 (random) vs oracle + selective-prediction curve
!python scripts/eval_ranked.py \
    --checkpoint checkpoints/best_model_conf.pt \
    --split test --n_poses 10 --n_steps 20 \
    --out_dir results/phase2/ranked --device cuda
```
Look for `top1_ranked_success_pct` clearly above `top1_random_success_pct`, and a rising
`selective_curve` (the DiffDock 38→83 effect).

---

## Phase 3 — leakage-controlled split + real baselines

### 3a. Build the clean split and retrain on it

```python
!python scripts/build_clean_split.py --out data/splits_clean.json --cutoff 0.5
# Note the report: median_per_test_max_tanimoto should be ~0.37 vs ~0.69 for the random split.

!WANDB_MODE=offline python -m src.training.train \
    --profile l4 --geom_loss_weight 0.1 --splits data/splits_clean.json \
    --n_epochs 100 --checkpoint_dir checkpoints/clean
```
Expect the numbers to **drop** vs the random split — that is the honest generalization signal.

### 3b. Score external baselines on the same set (optional, needs the tools)

DiffDock / Vina / GNINA must be installed/run separately; then score their SDF outputs with the
shared in-frame evaluator so every method is comparable:

```python
!python scripts/baselines/score_external_poses.py --method diffdock \
    --pred_dir <diffdock_output_sdf_dir> --pred_suffix _rank1 \
    --ref_dir <crystal_ligand_sdf_dir> \
    --out_dir results/phase3/baselines --run_posebusters
```

---

## Saving results back to Drive

Colab disks are ephemeral. Copy artifacts (and any new checkpoints) back to Drive before the
session ends:

```python
!cp -r results {DRIVE}/results_$(date +%Y%m%d)
!cp checkpoints/geom/best_model.pt {DRIVE}/best_model_geom.pt
!cp checkpoints/best_model_conf.pt {DRIVE}/best_model_conf.pt
```

## Tips

- **Keep the session alive**: long retrains can hit Colab's idle timeout — keep the tab active.
- **wandb**: `WANDB_MODE=offline` avoids login prompts; drop it (and pass `--wandb_project
  flow-match`) if you want live logging.
- **Faster smoke first**: add `--max_complexes 20` to any eval script to validate the pipeline
  before committing to the full test set.
- **Checkpoint architecture**: every eval script reads `hidden_dim`/`n_layers` from the
  checkpoint's `run_config`, so the L4 (128/6, 819,975-param) model and the CPU smoke model are
  both handled automatically.
```
