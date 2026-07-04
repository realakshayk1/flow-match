# Single Source of Truth

## Current Best Numbers

**Headline (pocket-conditioned re-docking, leakage-controlled split, Colab L4).**
819,975-param model (hidden=128, 6 layers), 50-step Euler + pocket-restrained MMFF relaxation.
Split: ligand-declustered (Butina Tanimoto 0.5); test = 464 complexes; median train↔test
max-Tanimoto `0.368` (vs `0.693` random, `1.00` original seed-42 split).

- **RMSD < 2 Å AND PB-valid (joint headline):** `68.3%`
- PoseBusters valid (`mol`): `80.8%`
- Dock RMSD < 2 Å (in-frame, symmetry-corrected): `74.1%`
- Dock RMSD median: `1.36 Å`
- Shape RMSD median (Kabsch-aligned; **not** docking): `1.09 Å`
- ETKDG baseline shape RMSD < 2 Å: `65.9%` (model shape < 2 Å: `85.6%`)
- Source: `results/phase3/clean_eval/results_summary.json`

**Physical validity depends on relaxation.** Raw model output: PB-valid ≈ `0%`; with
`--relax pocket`: `~81%` (dock RMSD median moves ≈ +0.07 Å). Measured on the geom checkpoint
(random split): `0.2%` → `81.6%`.

**Confidence / selective prediction** (multi-pose + shared-trunk confidence head, random-split
checkpoint): top-1 ranked `75.8%` vs top-1 random `75.4%`; oracle over 10 poses `84.4%`.
Selective curve (success @ coverage): `95.8%` @66%, `98.3%` @50%, `100%` @10%.
- Source: `results/phase2/ranked/ranked_summary.json`

**Metric definitions.** *Dock RMSD* = crystal-frame, no superposition, symmetry-corrected
(the docking metric). *Shape RMSD* = Kabsch-aligned (conformer shape only; not docking).

> Historical smoke-scale numbers (Phase 2 throughput, Phase 3 1-pair cross-dock, Phase 4 mock
> EF1%) are superseded and should not be used. Throughput/EF1% remain to be run at real scale.

## Automation (what to run)

| Phase | One-shot orchestrator | Notes |
|-------|----------------------|--------|
| 1 | `python scripts/run_phase1_bundle.py --checkpoint <ckpt> --out_root results/phase1` | Raw + UFF + `compare_posebusters_runs` + `baseline_bundle/` |
| 2 | `python scripts/benchmark_throughput.py prepare-manifest-pdbbind ...` then `prepare_pdbqt_for_vina.py` on manifest, then `benchmark --engine vina` / `--engine flowmatch` | `timing_raw.csv` is a copy of per-row timings |
| 3 | `python scripts/run_crossdock_bundle.py --manifest_csv ... --out_root results/phase3/run` | Optional PDBQT prep via `prepare_pdbqt_for_vina.py`; `--dry_run` for smoke |
| 4 | `python scripts/run_phase4_screening_bundle.py --input_csv ... --known_actives_csv ... --out_root results/phase4/<target>` | Library → screen → EF1% → top-hit MD |

## Pending Runs (Priority Order)

1. Evaluate the clean checkpoint on the true **PoseBusters benchmark set** (post-2021) — turns
   "re-docking on known pockets" into a real generalization number.
2. Run **Vina/GNINA (and DiffDock-Pocket)** on the same complexes via
   `scripts/baselines/score_external_poses.py` for a head-to-head table.
3. Add protein/sequence clustering to the split (currently ligand-only declustering).
4. Phase 2 throughput vs real docking engines; Phase 4 scaled screen with non-mock scores.

## Known Caveats

- The headline numbers are **pocket-conditioned re-docking** on a **ligand-declustered** split
  (proteins may overlap train/test) — not blind docking and not full generalization. Do not
  compare to blind-docking benchmarks (DiffDock/FlowDock).
- PoseBusters runs use ligand-only `mol` mode and do not replace protein-context cross-docking evidence.
- Throughput comparisons against docking engines remain incomplete until real Vina/GNINA rows are measured.
- Phase 4 runtime currently uses `mock_target_model` scoring unless precomputed model scores are supplied.
- Vina/GNINA require binaries on PATH; `obabel` is required for PDBQT preparation and conversions.
