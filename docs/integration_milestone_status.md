# Integration Milestone Status

## Conflict Resolutions Applied

- Fixed Phase 3 CLI import issue in `scripts/crossdock/prepare_casf_crossdock.py` so it runs from repo root.
- Reconciled artifact naming by documenting a single integration convention in `docs/four_phase_runbook.md`.
- Reconciled phase readiness mismatch (historical): Phase 3 runners/eval and Phase 4 enrichment/hit scripts are now in-repo; docs/runbook updated to match.

## Phase Status

- **Phase 1:** Partial (smoke complete, full reruns pending)
- **Phase 2:** Partial (harness complete, external engines blocked)
- **Phase 3:** Runnable (prep + model/Vina runners + comparison eval; needs CASF index + optional Vina/obabel)
- **Phase 4:** Runnable (library build, screen, enrichment EF1%, top-hit analysis; needs ranked CSV + known actives)

## Canonical Paths

- Runbook: `docs/four_phase_runbook.md`
- Protocol: `docs/eval_protocol.md`
- SSOT metrics/caveats: `results/single_source_of_truth.md`
- Bundle entrypoints: `scripts/run_phase1_bundle.py`, `scripts/run_crossdock_bundle.py`, `scripts/run_phase4_screening_bundle.py`, `scripts/prepare_pdbqt_for_vina.py`
