# Flow-Match: SE(3)-Equivariant Flow Matching for Pocket-Conditioned Ligand Pose Generation

A lightweight (**819,975-parameter**) SE(3)-equivariant EGNN trained with conditional flow
matching to generate small-molecule binding poses given a protein pocket. The emphasis of this
repo is **evaluation honesty**: every headline number is an in-frame, symmetry-corrected docking
RMSD reported jointly with PoseBusters physical validity, on a leakage-controlled split.

> **Task framing (read first).** This is **pocket-conditioned re-docking**: the binding-site
> point cloud is provided as input (extracted around the crystal ligand). This is a
> substantially easier task than **blind docking** (DiffDock, FlowDock, etc., which take the
> whole protein). **Do not compare the numbers below to blind-docking benchmarks** — the tasks
> are different.

## Results

Model: 819,975 params (hidden=128, 6 EGNN layers), 20–50-step Euler inference.
Split: **ligand-declustered** PDBBind (Butina clustering on Morgan/Tanimoto, whole clusters
assigned to one split). Test = 464 complexes. Median train↔test max-Tanimoto **0.368**
(vs **0.693** for a random split, and **1.00** — i.e. identical ligands — for the original
seed-42 split). Poses are polished with a pocket-restrained MMFF relaxation (`--relax pocket`).

| Metric | Value |
|---|---:|
| **RMSD < 2 Å AND PB-valid** (headline joint metric) | **68.3%** |
| PoseBusters valid (`mol` checks) | 80.8% |
| Dock RMSD < 2 Å (in-frame, symmetry-corrected) | 74.1% |
| Dock RMSD, median | 1.36 Å |
| Shape RMSD, median (Kabsch-aligned; **not** docking) | 1.09 Å |
| ETKDG baseline, shape RMSD < 2 Å | 65.9% |

**Two metrics, on purpose.** *Dock RMSD* is measured in the crystal frame with no
superposition and with molecular-symmetry correction — it reflects whether the ligand is
*placed* correctly. *Shape RMSD* is Kabsch-aligned (rotation+translation removed) — it only
reflects conformer shape and is **not** a docking metric. We report both so the gap is visible.

**Physical validity requires the relaxation step.** Raw model output passes almost no
PoseBusters bond-geometry checks (PB-valid ≈ 0%); the pocket-restrained MMFF relaxation lifts
PB-valid to ~81% while barely changing dock RMSD (≈ +0.07 Å median). This mirrors the
"+energy-minimization" recipe reported for FlowDock.

**Confidence ranking / selective prediction** (multi-pose + a shared-trunk confidence head).
On a strong base model most sampled poses are already correct (~79%), so confidence-ranked
top-1 (75.8%) ≈ random top-1 (75.4%). But the head is well-calibrated for *triage*: restricting
to the most-confident subset gives 95.8% success at 66% coverage, 98.3% at 50%, and 100% at 10%.

### Honest caveats
- **Pocket-conditioned, not blind.** The binding site is given. Not comparable to blind-docking
  numbers (DiffDock ~38% RMSD<2Å, FlowDock ~51% PB-valid on the PoseBusters set).
- **The split de-duplicates ligands, not proteins.** Test proteins may still appear in training,
  so this is "novel-ish ligands on known pockets," not full generalization. A true
  generalization number requires evaluating on the **PoseBusters benchmark set** (post-2021
  complexes) or adding protein/sequence clustering to the split — **pending**.
- **No external baselines run yet on this set.** Vina/GNINA/DiffDock have not been run on these
  exact complexes, so no head-to-head "competitive with X" claim is made. The shared evaluator
  (`scripts/baselines/score_external_poses.py`) exists to do this — **pending**.

## Method

1. **Featurize** ligand graph (topology only, no coordinates) + pocket point cloud (atoms within
   10 Å of the ligand centroid). Precomputed to `HeteroData` `.pt` files.
2. **EGNN velocity field** (SE(3)-equivariant, verified numerically to ~1e-8). Pocket coordinates
   are held fixed; cross-edges flow pocket→ligand.
3. **Conditional flow matching**: linear interpolant from N(0,I) noise to crystal coords, constant
   target velocity, optional bonded-geometry auxiliary loss (`--geom_loss_weight`).
4. **Inference**: 20–50-step Euler, then pocket-restrained MMFF relaxation.
5. **Eval**: in-frame symmetry-corrected dock RMSD + PoseBusters `mol` checks, reported jointly.

## Reproduce

Data (PDBBind processed `.pt` files) and checkpoints are not committed (gitignored). Training was
run on Colab (L4). See **`docs/colab_guide.md`** for the full notebook, and
`docs/four_phase_runbook.md` / `docs/eval_protocol.md` for per-phase commands and policy.

```bash
# Leakage-controlled split
python scripts/build_clean_split.py --out data/splits_clean.json --cutoff 0.5

# Train (L4/A100/H100 profiles use torch.compile + AMP; add --geom_loss_weight 0.1)
python -m src.training.train --profile l4 --geom_loss_weight 0.1 \
    --splits data/splits_clean.json --n_epochs 100 --checkpoint_dir checkpoints/clean

# Honest eval: in-frame dock RMSD + PB-valid, with pocket relaxation
python scripts/eval_posebusters.py --checkpoint checkpoints/clean/best_model.pt \
    --splits data/splits_clean.json --split test --n_inference_steps 50 --relax pocket \
    --output_dir results/phase3/clean_eval

# Multi-pose + confidence ranking
python scripts/train_confidence.py --base_checkpoint checkpoints/clean/best_model.pt \
    --out_checkpoint checkpoints/clean_conf.pt --split train --n_poses 8 --epochs 5
python scripts/eval_ranked.py --checkpoint checkpoints/clean_conf.pt --split test --n_poses 10
```

## Status

| Piece | State |
|---|---|
| SE(3)-equivariant EGNN + flow matching | done, equivariance-tested |
| In-frame symmetry-corrected dock RMSD + joint PB-valid metric | done |
| Pocket-restrained relaxation (physical validity) | done |
| Bonded-geometry auxiliary loss | done |
| Multi-pose + confidence head + selective-prediction curve | done |
| Leakage-controlled (ligand-declustered) split + retrain | done |
| Eval on the true PoseBusters benchmark set (post-2021) | **pending** |
| Real Vina/GNINA/DiffDock baselines on the same complexes | **pending** |

## References

- Satorras, Hoogeboom, Welling (2021) — E(n)-Equivariant GNNs
- Lipman et al. (2022) — Flow Matching for Generative Modeling
- Corso et al. (2023) — DiffDock
- Buttenschoen et al. (2024) — PoseBusters
- Morehead et al. (2024) — FlowDock
- PDBbind CleanSplit / Leak-Proof PDBBind — split-leakage methodology
