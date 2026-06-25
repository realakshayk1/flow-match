# Phase 4 Screening Scripts

## `build_library.py`

- Purpose: standardize, deduplicate, and filter a compound library.
- Input:
  - `--input_csv`: CSV with `compound_id`, `smiles` (optional `source`)
- Output:
  - `--output_csv`: cleaned library with `std_smiles` + descriptors
  - `--filter_log_csv`: accepted/rejected decisions
  - `--stats_json`: aggregate build statistics

## `run_screen.py`

- Purpose: score and rank compounds for a target with runtime accounting.
- Input:
  - `--library_csv`: output from `build_library.py`
  - `--target`: `cdk2` or `egfr`
  - optional `--precomputed_scores_csv` with `compound_id,predicted_score_kcal_mol`
- Output:
  - `--out_ranked_csv`: sorted hit table with ranks
  - `--out_runtime_json`: throughput and latency metrics

## `evaluate_enrichment.py`

- Purpose: calculate EF1% against known active mapping.
- Input:
  - `--ranked_csv`: output from `run_screen.py`
  - `--known_actives_csv`: known active mapping (`target,compound_id,std_smiles,is_active`)
- Output:
  - `--out_json`: enrichment summary (`ef_1pct`, active prevalence)
  - `--out_annotated_csv`: ranked table with `is_active` flag

## `analyze_top_hits.py`

- Purpose: produce top-hit interpretation scaffold with novelty and descriptors.
- Input:
  - `--ranked_csv`: output from `run_screen.py`
  - `--known_actives_csv`: same mapping used for enrichment
- Output:
  - `--out_csv`: top-hit novelty/descriptor table
  - `--out_md`: markdown report scaffold
