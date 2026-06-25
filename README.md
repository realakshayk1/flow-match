# Flow-Match Four-Phase Evaluation Stack

This repository is organized around an integrated four-phase improvement plan:

1. Phase 1: PoseBusters baseline + UFF comparison
2. Phase 2: Throughput/profiling against external baselines
3. Phase 3: CASF-style cross-docking with model-vs-Vina comparison
4. Phase 4: Real-target screening, EF1%, and top-hit interpretation

## Canonical Docs

- `docs/four_phase_runbook.md` - step-by-step commands per phase
- `docs/eval_protocol.md` - benchmark policy, assumptions, caveats
- `results/single_source_of_truth.md` - current best numbers and pending runs
- `docs/integration_milestone_status.md` - done/partial/blocked status
- `PRD.me` - project rationale and roadmap context

## Quick Start

### Phase 1 Bundle

```bash
python scripts/run_phase1_bundle.py --checkpoint checkpoints/best_model.pt --out_root results/phase1 --split test --n_inference_steps 20 --device cpu
```

### Phase 3 Bundle

```bash
python scripts/run_crossdock_bundle.py --manifest_csv eval/crossdock/manifest.csv --out_root results/phase3/run --checkpoint checkpoints/best_model.pt
```

### Phase 4 Bundle

```bash
python scripts/run_phase4_screening_bundle.py --target cdk2 --input_csv data/screening/library_templates/toy_library.csv --known_actives_csv data/screening/targets/cdk2/known_actives.csv --out_root results/phase4/cdk2_bundle
```

## Current Integration Status

- Phase 1: runnable and smoke-validated; full reruns may still be pending.
- Phase 2: harness is in place; real Vina/GNINA benchmarking depends on local binaries and prepared manifest fields.
- Phase 3: prep/runners/evaluator exist; full runs depend on checkpoint and docking toolchain.
- Phase 4: library/screen/enrichment/top-hit scripts are present; full-scale runs depend on larger input libraries.

## Scientific Caveats

- PoseBusters `mol` mode checks ligand geometry, not full protein-context docking validity.
- Re-docking and cross-docking are different settings and should be reported separately.
- Throughput speedup claims are only final after matched-input external-engine runs.
- EF1% claims must include active-set provenance and explicit random baseline.
# Flow-Match Four-Phase Evaluation Stack

This repository contains a lightweight SE(3)-equivariant flow-matching model and an integrated four-phase evaluation pipeline:

- Phase 1: PoseBusters baseline and UFF post-processing
- Phase 2: Throughput/profiling against external baselines
- Phase 3: CASF-style cross-docking + Vina comparison
- Phase 4: Real-target screening + EF1% + top-hit interpretation

For the operational workflow, use:

- `docs/four_phase_runbook.md` (single runbook)
- `docs/eval_protocol.md` (benchmark and caveat policy)
- `results/single_source_of_truth.md` (latest integrated status)

## Quick Start (Integrated)

### Phase 1 bundle

```bash
python scripts/run_phase1_bundle.py --checkpoint checkpoints/best_model.pt --out_root results/phase1 --split test --n_inference_steps 20 --device cpu
```

### Phase 2 setup and smoke

```bash
python scripts/benchmark_throughput.py inspect --out_dir eval/throughput
python scripts/benchmark_throughput.py prepare-manifest --poses_dir eval/posebusters_uff/poses --out_manifest eval/benchmark_100/manifest.csv --limit 100
python scripts/benchmark_throughput.py benchmark --engine noop --manifest eval/benchmark_100/manifest.csv --out_dir eval/throughput/noop --warmup 5 --repeats 100
```

### Phase 3 bundle

```bash
python scripts/run_crossdock_bundle.py --manifest_csv eval/crossdock/manifest.csv --out_root results/phase3/smoke --dry_run --limit_pairs 5
```

### Phase 4 bundle

```bash
python scripts/run_phase4_screening_bundle.py --target cdk2 --input_csv data/screening/library_templates/toy_library.csv --known_actives_csv data/screening/targets/cdk2/known_actives.csv --out_root results/phase4/cdk2_bundle_smoke
```

## Current Integration State

- Phase 1/2 tooling is integrated and runnable.
- Phase 3/4 scaffolds and bundle scripts are present and smoke-runnable.
- External benchmark quality depends on local environment:
  - `vina`, `gnina`, and `obabel` availability
  - prepared receptor/ligand inputs for docking baselines

Always treat `results/single_source_of_truth.md` as the canonical status for interview-facing claims.

## Scientific Caveats

- PoseBusters in `mol` mode is ligand geometry validity, not full protein-context docking validity.
- Re-docking, cross-docking, and screening enrichment are separate evidence tiers and should not be conflated.
- LIT-PCBA is excluded from official claims per protocol due leakage/redundancy concerns.

## References

- Satorras et al. 2021, E(n)-Equivariant GNNs
- Lipman et al. 2022, Flow Matching
- Corso et al. 2023, DiffDock
- Buttenschoen et al. 2024, PoseBusters
