import argparse
import csv
import hashlib
import json
import shlex
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class StageProfiler:
    totals: dict[str, float]
    counts: dict[str, int]

    def __init__(self) -> None:
        self.totals = {}
        self.counts = {}

    def add(self, stage: str, elapsed_ms: float) -> None:
        self.totals[stage] = self.totals.get(stage, 0.0) + elapsed_ms
        self.counts[stage] = self.counts.get(stage, 0) + 1

    def summary(self) -> dict[str, dict[str, float | int]]:
        out: dict[str, dict[str, float | int]] = {}
        for stage in sorted(self.totals.keys()):
            total = self.totals[stage]
            count = self.counts.get(stage, 0)
            out[stage] = {
                "total_ms": round(total, 3),
                "calls": count,
                "avg_ms": round(total / max(1, count), 3),
            }
        return out


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return float(ordered[idx])


def _safe_float(v: float | None) -> float | None:
    if v is None:
        return None
    return round(float(v), 3)


def _metric_summary(values_ms: list[float]) -> dict[str, float | int | None]:
    if not values_ms:
        return {
            "n_measured": 0,
            "mean_ms_per_ligand": None,
            "std_ms_per_ligand": None,
            "median_ms_per_ligand": None,
            "p95_ms_per_ligand": None,
            "throughput_ligands_per_s": None,
        }
    mean_ms = float(statistics.mean(values_ms))
    return {
        "n_measured": len(values_ms),
        "mean_ms_per_ligand": _safe_float(mean_ms),
        "std_ms_per_ligand": _safe_float(float(statistics.pstdev(values_ms))),
        "median_ms_per_ligand": _safe_float(float(statistics.median(values_ms))),
        "p95_ms_per_ligand": _safe_float(_percentile(values_ms, 0.95)),
        "throughput_ligands_per_s": _safe_float(1000.0 / mean_ms if mean_ms > 0 else None),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_manifest_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    required = {"ligand_id", "ligand_path"}
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _build_manifest_from_glob(glob_pattern: str, max_ligands: int) -> list[dict[str, str]]:
    ligands = sorted(Path(".").glob(glob_pattern))
    rows: list[dict[str, str]] = []
    for path in ligands[:max_ligands]:
        cid = path.name.replace("_pred.sdf", "")
        rows.append(
            {
                "ligand_id": cid,
                "ligand_path": path.as_posix(),
                "protein_path": "",
                "receptor_pdbqt": "",
                "ligand_pdbqt": "",
            }
        )
    if not rows:
        raise ValueError(f"No ligands matched glob: {glob_pattern}")
    return rows


def _manifest_hash(rows: list[dict[str, str]]) -> str:
    payload = "\n".join(f"{r['ligand_id']}|{r['ligand_path']}" for r in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _render_command(template: list[str] | str, row: dict[str, str]) -> list[str]:
    data = dict(row)
    data.setdefault("ligand", row.get("ligand_path", ""))
    if isinstance(template, str):
        return [part.format_map(data) for part in shlex.split(template)]
    return [part.format_map(data) for part in template]


def _binary_available(cmd: list[str]) -> bool:
    if not cmd:
        return False
    executable = cmd[0]
    if "/" in executable or "\\" in executable:
        return Path(executable).exists()
    return shutil.which(executable) is not None


def _run_once(cmd: list[str]) -> tuple[float, int, str]:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    stderr_tail = (proc.stderr or "")[-400:]
    return dt_ms, proc.returncode, stderr_tail


def run(config_path: Path, out_dir: Path) -> dict[str, Any]:
    profiler = StageProfiler()

    t0 = time.perf_counter()
    cfg = _load_json(config_path)
    profiler.add("load_config", (time.perf_counter() - t0) * 1000.0)

    warmup = int(cfg.get("warmup", 2))
    repeats = int(cfg.get("repeats", 8))
    backends_cfg: dict[str, dict[str, Any]] = cfg["backends"]
    backend_order = cfg.get("backend_order", list(backends_cfg.keys()))
    speedup_reference = cfg.get("speedup_reference", backend_order[0])

    t0 = time.perf_counter()
    manifest_rows: list[dict[str, str]]
    if "manifest_csv" in cfg:
        manifest_rows = _read_manifest_csv(Path(cfg["manifest_csv"]))
    else:
        manifest_rows = _build_manifest_from_glob(str(cfg["ligand_glob"]), int(cfg.get("max_ligands", 100)))
    profiler.add("prepare_manifest", (time.perf_counter() - t0) * 1000.0)

    selected = manifest_rows[: warmup + repeats]
    if len(selected) < warmup:
        raise ValueError("Not enough ligands for warmup")

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "ligand_manifest_used.csv", selected)

    raw_rows: list[dict[str, Any]] = []
    backend_summaries: dict[str, dict[str, Any]] = {}

    for backend in backend_order:
        bc = backends_cfg[backend]
        command_template = bc["command"]
        skip_if_missing = bool(bc.get("skip_if_missing", False))
        measured: list[float] = []
        failures = 0
        skipped = 0

        for i, row in enumerate(selected):
            phase = "warmup" if i < warmup else "measured"
            ligand_id = row["ligand_id"]

            t_stage = time.perf_counter()
            cmd = _render_command(command_template, row)
            profiler.add("build_command", (time.perf_counter() - t_stage) * 1000.0)

            if not _binary_available(cmd):
                status = "skipped" if skip_if_missing else "failed"
                skipped += 1 if skip_if_missing else 0
                failures += 0 if skip_if_missing else 1
                raw_rows.append(
                    {
                        "backend": backend,
                        "ligand_id": ligand_id,
                        "phase": phase,
                        "duration_ms": None,
                        "status": status,
                        "return_code": -127,
                        "stderr_tail": f"missing executable: {cmd[0]}",
                        "command": " ".join(cmd),
                    }
                )
                continue

            t_stage = time.perf_counter()
            dt_ms, rc, err = _run_once(cmd)
            profiler.add(f"run_{backend}", (time.perf_counter() - t_stage) * 1000.0)

            ok = rc == 0
            if phase == "measured" and ok:
                measured.append(dt_ms)
            if not ok:
                failures += 1

            raw_rows.append(
                {
                    "backend": backend,
                    "ligand_id": ligand_id,
                    "phase": phase,
                    "duration_ms": _safe_float(dt_ms),
                    "status": "ok" if ok else "failed",
                    "return_code": rc,
                    "stderr_tail": err,
                    "command": " ".join(cmd),
                }
            )

        metrics = _metric_summary(measured)
        metrics.update(
            {
                "failures": failures,
                "skipped": skipped,
                "warmup_ligands": warmup,
                "measured_target_ligands": repeats,
                "failure_rate_pct": _safe_float((failures / max(1, len(selected))) * 100.0),
            }
        )
        backend_summaries[backend] = metrics

    ref_mean = backend_summaries.get(speedup_reference, {}).get("mean_ms_per_ligand")
    for backend, summary in backend_summaries.items():
        mean = summary.get("mean_ms_per_ligand")
        if isinstance(ref_mean, (int, float)) and isinstance(mean, (int, float)) and mean > 0:
            summary["speedup_vs_reference"] = _safe_float(ref_mean / mean)
        else:
            summary["speedup_vs_reference"] = None

    report = {
        "config_path": config_path.as_posix(),
        "manifest_hash": _manifest_hash(selected),
        "n_ligands_total": len(selected),
        "warmup": warmup,
        "repeats": repeats,
        "speedup_reference": speedup_reference,
        "backends": backend_summaries,
        "stage_profile": profiler.summary(),
    }

    t0 = time.perf_counter()
    _write_csv(out_dir / "raw_timings.csv", raw_rows)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    profiler.add("write_artifacts", (time.perf_counter() - t0) * 1000.0)
    report["stage_profile"] = profiler.summary()
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# Phase 2 Throughput Summary",
        "",
        f"- Config: `{config_path.as_posix()}`",
        f"- Ligands (warmup + measured): {len(selected)} ({warmup} + {repeats})",
        f"- Speedup reference backend: `{speedup_reference}`",
        f"- Manifest hash: `{report['manifest_hash']}`",
        "",
        "## Backend Metrics",
        "",
        "| backend | mean ms | std ms | median ms | p95 ms | throughput lig/s | speedup vs ref | failures | skipped |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for backend in backend_order:
        s = backend_summaries[backend]
        md_lines.append(
            "| {backend} | {mean} | {std} | {median} | {p95} | {tput} | {speed} | {fail} | {skip} |".format(
                backend=backend,
                mean=s.get("mean_ms_per_ligand"),
                std=s.get("std_ms_per_ligand"),
                median=s.get("median_ms_per_ligand"),
                p95=s.get("p95_ms_per_ligand"),
                tput=s.get("throughput_ligands_per_s"),
                speed=s.get("speedup_vs_reference"),
                fail=s.get("failures"),
                skip=s.get("skipped"),
            )
        )
    (out_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 2 throughput benchmark entrypoint")
    p.add_argument("--config", required=True, help="JSON config with backend commands and timing settings")
    p.add_argument("--out-dir", required=True, help="Output directory for raw and summary artifacts")
    return p


def main() -> None:
    args = build_parser().parse_args()
    report = run(config_path=Path(args.config), out_dir=Path(args.out_dir))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
