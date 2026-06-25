# Final Completion Checklist (Live)

## Phase 1 - Evaluation Credibility
- [x] Run baseline PoseBusters on checkpoint (10-complex run): `results/phase1/terminal_run/posebusters_raw`
- [x] Run +UFF PoseBusters (10-complex run): `results/phase1/terminal_run/posebusters_uff`
- [x] Generate raw vs UFF comparison: `results/phase1/terminal_run/comparison/phase1_summary.json`
- [x] Export canonical bundle artifacts: `results/phase1/terminal_run/baseline_bundle`
- [ ] Full test split run (all complexes) for final interview-grade numbers

## Phase 2 - Throughput
- [x] Environment inspected: `eval/throughput/smoke_inspect/env_inspect.json`
- [x] PDBBind manifest generated: `eval/smoke_bench_manifest.csv`
- [x] FlowMatch timing run completed: `eval/throughput/flowmatch_terminal/summary.json`
- [x] Noop timing run completed: `eval/throughput/smoke_noop/summary.json`
- [ ] Vina timing run (blocked: Vina + obabel missing)
- [ ] GNINA timing run (blocked: gnina missing)

## Phase 3 - Cross-Docking
- [x] Crossdock manifest smoke prepared: `eval/crossdock/smoke_manifest.csv`
- [x] Model crossdock real (non-dry) run: `results/phase3/model_terminal_real/model_summary.json`
- [x] Vina runner dry smoke run: `results/phase3/vina_terminal_dry/vina_summary.json`
- [x] Comparison report emitted: `results/phase3/compare_terminal_mixed/crossdock_report.md`
- [ ] Real Vina crossdock baseline (blocked: Vina + obabel missing)
- [ ] CASF-scale manifest and report (requires CASF index + full run)

## Phase 4 - Screening Story
- [x] End-to-end bundle run (toy): `results/phase4/cdk2_bundle_smoke`
- [x] EF1% computed: `results/phase4/cdk2_bundle_smoke/enrichment_summary.json`
- [x] Top-hit report produced: `results/phase4/cdk2_bundle_smoke/top_hits_report.md`
- [ ] 50K library build + full runtime story
- [ ] Final medicinal chemistry interpretation write-up (fill TODO template text)

## Global Blockers To Unblock Next
- [ ] Install Open Babel (`obabel`) on PATH
- [ ] Install AutoDock Vina (`vina`) on PATH
- [ ] Install GNINA (`gnina`) on PATH
- [ ] Provide/prepare CASF production index + larger run budgets
