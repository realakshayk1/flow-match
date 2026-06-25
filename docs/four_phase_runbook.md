# Four-Phase Runbook

This runbook is the top-level operational guide for Phases 1-4.

## CLI and Artifact Conventions

- Use `--output_dir` for phase run outputs (folders with multiple artifacts).
- Use `--out_dir` for utility/reporting outputs (benchmark/report folders).
- Keep canonical artifacts under `eval/` until `results/phase*` is fully wired.
- Use consistent split/config flags where available:
  - `--split test`
  - `--n_inference_steps <int>`
  - `--device cpu|cuda`

## Phase 1: Baseline + UFF

- **Objective:** Produce raw vs UFF PoseBusters comparison from the same checkpoint/split.
- **One-shot bundle (recommended):**
  - `python scripts/run_phase1_bundle.py --checkpoint checkpoints/best_model.pt --out_root results/phase1 --split test --n_inference_steps 20 --device cpu`
  - Writes `results/phase1/comparison/`, `results/phase1/baseline_bundle/` (comparison JSON/MD, per_complex raw/uff CSVs, summary copies).
- **Primary command (raw):**
  - `python scripts/eval_posebusters.py --checkpoint checkpoints/best_model.pt --processed_dir data/processed --splits data/splits.json --output_dir eval/posebusters_raw --split test --n_inference_steps 20 --device cpu`
- **Optional standardized bundle (P1.2-style paths):** append `--export_phase1_dir results/phase1/baseline` to copy `results_summary.json` → `summary.json` and `results_raw.csv` → `per_complex.csv`.
- **Primary command (+UFF):**
  - `python scripts/eval_posebusters.py --checkpoint checkpoints/best_model.pt --processed_dir data/processed --splits data/splits.json --output_dir eval/posebusters_uff --split test --n_inference_steps 20 --device cpu --uff_postprocess`
- **Comparison report:**
  - `python scripts/compare_posebusters_runs.py --raw eval/posebusters_raw/results_raw.csv --uff eval/posebusters_uff/results_raw.csv --out_dir eval/posebusters_report`
- **Expected artifacts:**
  - `eval/posebusters_raw/results_summary.json`
  - `eval/posebusters_uff/results_summary.json`
  - `eval/posebusters_report/phase1_summary.json`

## Phase 2: Throughput and Profiling

- **Objective:** Produce latency and failure summaries on a matched compound set.
- **Inspect environment:**
  - `python scripts/benchmark_throughput.py inspect --out_dir eval/throughput`
- **Prepare manifest from poses (no protein paths):**
  - `python scripts/benchmark_throughput.py prepare-manifest --poses_dir eval/posebusters_uff/poses --out_manifest eval/benchmark_100/manifest.csv --limit 100`
- **Prepare manifest from PDBBind raw tree + splits** (adds `protein_path` / `ligand_path` for GNINA and PDBQT prep):
  - `python scripts/benchmark_throughput.py prepare-manifest-pdbbind --raw_dir data/raw --splits data/splits.json --split_name test --out_manifest eval/benchmark_pdbbind/manifest.csv --limit 100`
- **Add Vina PDBQT columns** (requires `obabel`):
  - `python scripts/prepare_pdbqt_for_vina.py --input_csv eval/benchmark_pdbbind/manifest.csv --out_csv eval/benchmark_pdbbind/manifest_vina.csv --cache_dir eval/pdbqt_cache --skip_existing`
- **Run harness smoke:**
  - `python scripts/benchmark_throughput.py benchmark --engine noop --manifest eval/benchmark_100/manifest.csv --out_dir eval/throughput/noop --warmup 5 --repeats 100`
- **Flow-Match inference timing** (requires processed tensors per `complex_id`):
  - `python scripts/benchmark_throughput.py benchmark --engine flowmatch --manifest eval/benchmark_pdbbind/manifest.csv --out_dir eval/throughput/flowmatch --checkpoint checkpoints/best_model.pt --processed_dir data/processed --warmup 2 --repeats 50 --device cpu`
- **Run Vina timing** (manifest must include `receptor_pdbqt` / `ligand_pdbqt`):
  - `python scripts/benchmark_throughput.py benchmark --engine vina --manifest eval/benchmark_pdbbind/manifest_vina.csv --out_dir eval/throughput/vina --warmup 0 --repeats 20`
- **Expected artifacts:**
  - `eval/throughput/env_inspect.json`
  - `eval/throughput/*/summary.json`, `per_ligand_times.csv`, **`timing_raw.csv`** (duplicate of per-ligand rows for Phase 2 SSOT naming)
  - Optional: `scripts/bench/run_benchmark.py` + `configs/bench/` for external command backends

## Phase 3: Cross-Docking + Vina Comparison

- **Objective:** Build CASF-style manifest and run model/Vina runners plus apples-to-apples comparison.
- **Prep (manifest):**
  - `python scripts/crossdock/prepare_casf_crossdock.py --index_csv <casf_index.csv> --root_dir . --out_manifest eval/crossdock/manifest.csv --limit_pairs 20`
- **Model runner (use `--dry_run` without checkpoint for plumbing smoke):**
  - `python scripts/crossdock/run_model_crossdock.py --manifest_csv eval/crossdock/manifest.csv --checkpoint checkpoints/best_model.pt --out_dir results/phase3/model_run --n_steps 20 --device cpu`
  - `python scripts/crossdock/run_model_crossdock.py --manifest_csv eval/crossdock/manifest.csv --out_dir results/phase3/model_smoke --dry_run --limit_pairs 5`
- **Vina runner (manifest rows need `receptor_pdbqt` and `ligand_pdbqt`; requires `vina` + `obabel` on PATH):**
  - `python scripts/crossdock/run_vina_crossdock.py --manifest_csv eval/crossdock/manifest.csv --out_dir results/phase3/vina_run --dry_run --limit_pairs 5`
- **Comparison report:**
  - `python scripts/crossdock/evaluate_crossdock.py --model_results results/phase3/model_run/model_results.jsonl --vina_results results/phase3/vina_run/vina_results.jsonl --out_dir results/phase3/compare`
- **Bundle (prep PDBQT + model + Vina + compare):**
  - `python scripts/run_crossdock_bundle.py --manifest_csv eval/crossdock/manifest.csv --out_root results/phase3/full_run --checkpoint checkpoints/best_model.pt`
  - Smoke: `--dry_run` (skips PDBQT/vina binaries; exercises runners + evaluator paths — use a manifest with existing pair rows).
- **PDBQT columns only:**
  - `python scripts/prepare_pdbqt_for_vina.py --input_csv eval/crossdock/manifest.csv --out_csv eval/crossdock/manifest_vina.csv --cache_dir eval/pdbqt_cache`
- **Expected artifacts:**
  - `results/phase3/model_run/model_summary.json`, `model_results.jsonl`, `poses/*.sdf`
  - `results/phase3/vina_run/vina_summary.json`, `vina_results.jsonl`
  - `results/phase3/compare/crossdock_comparison.json`, `crossdock_report.md`

## Phase 4: Target Screening + EF1% + Hit Interpretation

- **Objective:** Build target-centric screening pipeline and enrichment reporting.
- **One-shot bundle:**
  - `python scripts/run_phase4_screening_bundle.py --target cdk2 --input_csv data/screening/library_templates/toy_library.csv --known_actives_csv data/screening/targets/cdk2/known_actives.csv --out_root results/phase4/cdk2_bundle`
- **Current available commands:**
  - `python scripts/screening/build_library.py --input_csv data/screening/library_templates/toy_library.csv --output_csv eval/screening/clean_library.csv --filter_log_csv eval/screening/filter_log.csv --stats_json eval/screening/library_stats.json`
  - `python scripts/screening/run_screen.py --library_csv eval/screening/clean_library.csv --target cdk2 --out_ranked_csv eval/screening/ranked_hits.csv --out_runtime_json eval/screening/runtime.json`
  - `python scripts/screening/evaluate_enrichment.py --ranked_csv eval/screening/ranked_hits.csv --known_actives_csv data/screening/targets/cdk2/known_actives.csv --target cdk2 --out_json results/phase4/cdk2/enrichment_summary.json --out_annotated_csv results/phase4/cdk2/ranked_annotated.csv`
  - `python scripts/screening/analyze_top_hits.py --ranked_csv eval/screening/ranked_hits.csv --known_actives_csv data/screening/targets/cdk2/known_actives.csv --target cdk2 --top_n 20 --out_csv results/phase4/cdk2/top_hits.csv --out_md results/phase4/cdk2/top_hits_report.md`
- **Expected artifacts:**
  - `results/phase4/<target>/enrichment_summary.json`, `ranked_annotated.csv`
  - `results/phase4/<target>/top_hits.csv`, `top_hits_report.md`

## Scientific Caveats (Must Stay Explicit)

- PoseBusters `mol` mode validates molecular geometry but is not full protein-context docking validity.
- Re-docking metrics and cross-docking metrics are not interchangeable.
- Throughput comparisons are invalid until Vina/GNINA are installed and manifests include required receptor/ligand paths.
- Target-screening EF1% is sensitive to active/inactive mapping quality and chemical-series redundancy.
