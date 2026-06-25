# Phase 4 Screening Data Contracts

## Target choice

CDK2 is the default Phase 4 target in this scaffold because it has compact kinase-focused active chemotypes and a low-friction setup for enrichment sanity checks before scaling to larger libraries.

## Input templates

- `library_templates/toy_library.csv`
  - Required columns: `compound_id`, `smiles`
  - Optional columns: `source`
- `targets/cdk2/known_actives.csv`
  - Required columns: `target`, `compound_id`, `std_smiles`, `is_active`
  - Optional columns: `source_note`

## Expected generated outputs (recommended)

- `results/phase4/library/clean_library.csv`
- `results/phase4/library/filter_log.csv`
- `results/phase4/library/library_stats.json`
- `results/phase4/screen/ranked_hits.csv`
- `results/phase4/screen/runtime_summary.json`
- `results/phase4/enrichment/enrichment_summary.json`
- `results/phase4/enrichment/ranked_with_activity.csv`
- `results/phase4/top_hits/top_hits_analysis.csv`
- `results/phase4/top_hits/top_hits_report.md`
