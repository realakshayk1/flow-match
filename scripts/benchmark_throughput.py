import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _summary(times_ms: list[float]) -> dict[str, Any]:
    if not times_ms:
        return {
            "n": 0,
            "mean_ms_per_ligand": None,
            "std_ms_per_ligand": None,
            "p50_ms": None,
            "p90_ms": None,
            "p95_ms": None,
        }
    vals = sorted(times_ms)
    n = len(vals)
    idx90 = min(n - 1, int(round(0.90 * (n - 1))))
    idx95 = min(n - 1, int(round(0.95 * (n - 1))))
    return {
        "n": n,
        "mean_ms_per_ligand": round(float(statistics.mean(vals)), 3),
        "std_ms_per_ligand": round(float(statistics.pstdev(vals)), 3),
        "p50_ms": round(float(statistics.median(vals)), 3),
        "p90_ms": round(float(vals[idx90]), 3),
        "p95_ms": round(float(vals[idx95]), 3),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _check_binary(name: str) -> bool:
    return shutil.which(name) is not None


def command_inspect(args: argparse.Namespace) -> int:
    status = {
        "timestamp": _now_iso(),
        "vina_available": _check_binary("vina"),
        "gnina_available": _check_binary("gnina"),
        "obabel_available": _check_binary("obabel"),
        "python": shutil.which("python"),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "env_inspect.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0


def command_prepare_manifest(args: argparse.Namespace) -> int:
    poses_dir = Path(args.poses_dir)
    if not poses_dir.exists():
        raise FileNotFoundError(f"poses_dir does not exist: {poses_dir}")
    sdf_paths = sorted(poses_dir.glob("*_pred.sdf"))
    selected = sdf_paths[: args.limit]
    rows: list[dict[str, Any]] = []
    for p in selected:
        cid = p.name.replace("_pred.sdf", "")
        rows.append(
            {
                "complex_id": cid,
                "pred_sdf": str(p.as_posix()),
                "protein_path": "",
                "ligand_path": "",
                "receptor_pdbqt": "",
                "ligand_pdbqt": "",
            }
        )
    out = Path(args.out_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(out, rows)
    print(f"Wrote {len(rows)} rows to {out}")
    return 0


def command_prepare_manifest_pdbbind(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from src.data.dataset import load_splits

    splits = load_splits(args.splits)
    if args.split_name not in splits:
        raise ValueError(f"split {args.split_name!r} not in splits (have {list(splits.keys())})")
    ids: list[str] = splits[args.split_name]
    if args.limit:
        ids = ids[: args.limit]
    raw = Path(args.raw_dir)
    rows: list[dict[str, Any]] = []
    poses_root = Path(args.poses_dir).resolve() if args.poses_dir else None
    for cid in ids:
        prot = raw / cid / f"{cid}_protein.pdb"
        lig = raw / cid / f"{cid}_ligand.sdf"
        if not prot.is_file() or not lig.is_file():
            continue
        pred = ""
        if poses_root is not None:
            pp = poses_root / f"{cid}_pred.sdf"
            if pp.is_file():
                pred = str(pp)
        rows.append(
            {
                "complex_id": cid,
                "pred_sdf": pred,
                "protein_path": str(prot.resolve()),
                "ligand_path": str(lig.resolve()),
                "receptor_pdbqt": "",
                "ligand_pdbqt": "",
            }
        )
    out = Path(args.out_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(out, rows)
    print(f"Wrote {len(rows)} rows to {out}")
    return 0


def _run_external_timed(cmd: list[str]) -> tuple[float, int, str]:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    stderr_tail = (proc.stderr or "")[-500:]
    return dt_ms, proc.returncode, stderr_tail


def _load_flow_matcher(args: argparse.Namespace, flow_cache: dict[str, Any]):
    import torch

    from src.models.egnn import build_default_model
    from src.models.flow_model import FlowMatcher

    ckpt_path = Path(args.checkpoint).resolve()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    run_config = ckpt.get("run_config", {})
    state = ckpt["model_state"]
    hidden_dim = args.hidden_dim
    if hidden_dim is None:
        hidden_dim = run_config.get("hidden_dim", state["lig_emb.weight"].shape[0])
    n_layers = args.n_layers
    if n_layers is None:
        layer_indices = [int(k.split(".")[1]) for k in state.keys() if k.startswith("layers.")]
        n_layers = run_config.get("n_layers", max(layer_indices) + 1 if layer_indices else 6)

    device = torch.device(args.device)
    model = build_default_model(hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    model.load_state_dict(state)
    model.eval()
    matcher = FlowMatcher(model, n_steps=args.n_inference_steps).to(device)
    matcher.eval()
    flow_cache["matcher"] = matcher
    flow_cache["device"] = device


def command_benchmark(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8")))
    if not rows:
        raise ValueError("manifest has no rows")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_ligand_rows: list[dict[str, Any]] = []
    measured_times: list[float] = []
    failures = 0
    flow_cache: dict[str, Any] = {}

    for i, row in enumerate(rows):
        if i < args.warmup:
            continue
        if args.repeats and len(measured_times) >= args.repeats:
            break

        cid = row.get("complex_id", f"row_{i}")
        if args.engine == "noop":
            # A dry-run baseline for pipeline overhead when dependencies are unavailable.
            dt_ms = 0.05
            rc = 0
            err = ""
        elif args.engine == "vina":
            receptor = row.get("receptor_pdbqt", "")
            ligand = row.get("ligand_pdbqt", "")
            if not receptor or not ligand:
                failures += 1
                per_ligand_rows.append(
                    {"complex_id": cid, "engine": "vina", "time_ms": None, "return_code": -1, "error": "missing receptor_pdbqt/ligand_pdbqt in manifest"}
                )
                continue
            cmd = [
                "vina",
                "--receptor",
                receptor,
                "--ligand",
                ligand,
                "--center_x",
                "0",
                "--center_y",
                "0",
                "--center_z",
                "0",
                "--size_x",
                "20",
                "--size_y",
                "20",
                "--size_z",
                "20",
                "--exhaustiveness",
                str(args.exhaustiveness),
                "--out",
                str((out_dir / f"{cid}_vina_out.pdbqt").as_posix()),
            ]
            dt_ms, rc, err = _run_external_timed(cmd)
        elif args.engine == "gnina":
            receptor = row.get("protein_path", "")
            ligand = row.get("ligand_path", "")
            if not receptor or not ligand:
                failures += 1
                per_ligand_rows.append(
                    {"complex_id": cid, "engine": "gnina", "time_ms": None, "return_code": -1, "error": "missing protein_path/ligand_path in manifest"}
                )
                continue
            cmd = [
                "gnina",
                "-r",
                receptor,
                "-l",
                ligand,
                "--center_x",
                "0",
                "--center_y",
                "0",
                "--center_z",
                "0",
                "--size_x",
                "20",
                "--size_y",
                "20",
                "--size_z",
                "20",
                "--exhaustiveness",
                str(args.exhaustiveness),
                "-o",
                str((out_dir / f"{cid}_gnina_out.sdf").as_posix()),
            ]
            dt_ms, rc, err = _run_external_timed(cmd)
        elif args.engine == "flowmatch":
            repo = Path(__file__).resolve().parents[1]
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            import torch
            from torch_geometric.loader import DataLoader

            from src.data.dataset import PDBBindDataset

            if not args.checkpoint or not args.processed_dir:
                raise SystemExit("benchmark --engine flowmatch requires --checkpoint and --processed_dir")
            cid_fm = (row.get("complex_id") or "").strip()
            if not cid_fm:
                failures += 1
                per_ligand_rows.append(
                    {
                        "complex_id": cid,
                        "engine": "flowmatch",
                        "time_ms": None,
                        "return_code": -1,
                        "error": "missing complex_id in manifest row",
                    }
                )
                continue
            pt_path = Path(args.processed_dir) / f"{cid_fm}.pt"
            if not pt_path.is_file():
                failures += 1
                per_ligand_rows.append(
                    {
                        "complex_id": cid,
                        "engine": "flowmatch",
                        "time_ms": None,
                        "return_code": -1,
                        "error": f"no processed tensor for {cid_fm}",
                    }
                )
                continue
            try:
                if "matcher" not in flow_cache:
                    _load_flow_matcher(args, flow_cache)
                matcher = flow_cache["matcher"]
                device = flow_cache["device"]
                ds = PDBBindDataset(args.processed_dir, [cid_fm], cache=False)
                if len(ds) == 0:
                    raise RuntimeError("dataset filtered out missing pt")
                loader = DataLoader(ds, batch_size=1, shuffle=False)
                batch = next(iter(loader)).to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                matcher.generate(batch, n_steps=args.n_inference_steps)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt_ms = (time.perf_counter() - t0) * 1000.0
                rc = 0
                err = ""
            except Exception as exc:
                failures += 1
                per_ligand_rows.append(
                    {
                        "complex_id": cid,
                        "engine": "flowmatch",
                        "time_ms": None,
                        "return_code": -1,
                        "error": str(exc)[:500],
                    }
                )
                continue
        else:
            raise ValueError(f"Unsupported engine: {args.engine}")

        if rc != 0:
            failures += 1
            per_ligand_rows.append(
                {"complex_id": cid, "engine": args.engine, "time_ms": round(dt_ms, 3), "return_code": rc, "error": err}
            )
            continue

        measured_times.append(dt_ms)
        per_ligand_rows.append(
            {"complex_id": cid, "engine": args.engine, "time_ms": round(dt_ms, 3), "return_code": 0, "error": ""}
        )

    summary = _summary(measured_times)
    summary.update(
        {
            "timestamp": _now_iso(),
            "engine": args.engine,
            "manifest": str(manifest_path.as_posix()),
            "n_manifest_rows": len(rows),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "failures": failures,
            "failure_rate_pct": round((failures / max(1, len(per_ligand_rows))) * 100.0, 1),
        }
    )

    timing_csv = out_dir / "per_ligand_times.csv"
    _write_csv(timing_csv, per_ligand_rows)
    shutil.copy2(timing_csv, out_dir / "timing_raw.csv")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Throughput benchmarking utility")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_inspect = sub.add_parser("inspect", help="Inspect external benchmarking dependencies")
    p_inspect.add_argument("--out_dir", required=True)
    p_inspect.set_defaults(func=command_inspect)

    p_manifest = sub.add_parser("prepare-manifest", help="Build benchmark manifest from predicted SDF poses")
    p_manifest.add_argument("--poses_dir", required=True)
    p_manifest.add_argument("--out_manifest", required=True)
    p_manifest.add_argument("--limit", type=int, default=100)
    p_manifest.set_defaults(func=command_prepare_manifest)

    p_manifest_pb = sub.add_parser(
        "prepare-manifest-pdbbind",
        help="Build benchmark manifest rows from raw PDBBind layout + splits.json",
    )
    p_manifest_pb.add_argument("--raw_dir", required=True, help="e.g. data/raw with <id>/<id>_protein.pdb")
    p_manifest_pb.add_argument("--splits", required=True, help="splits.json path")
    p_manifest_pb.add_argument("--split_name", default="test", choices=["train", "val", "test"])
    p_manifest_pb.add_argument("--out_manifest", required=True)
    p_manifest_pb.add_argument("--limit", type=int, default=0, help="Max complexes (0 = all in split)")
    p_manifest_pb.add_argument(
        "--poses_dir",
        default="",
        help="Optional dir of <complex_id>_pred.sdf to attach pred_sdf column",
    )
    p_manifest_pb.set_defaults(func=command_prepare_manifest_pdbbind)

    p_bench = sub.add_parser("benchmark", help="Run timing benchmark")
    p_bench.add_argument("--engine", required=True, choices=["noop", "vina", "gnina", "flowmatch"])
    p_bench.add_argument("--manifest", required=True)
    p_bench.add_argument("--out_dir", required=True)
    p_bench.add_argument("--warmup", type=int, default=5)
    p_bench.add_argument("--repeats", type=int, default=100)
    p_bench.add_argument("--exhaustiveness", type=int, default=8)
    p_bench.add_argument("--checkpoint", default="", help="Flow-Match checkpoint (required for flowmatch)")
    p_bench.add_argument("--processed_dir", default="", help="Processed .pt directory (required for flowmatch)")
    p_bench.add_argument("--n_inference_steps", type=int, default=20)
    p_bench.add_argument("--hidden_dim", type=int, default=None)
    p_bench.add_argument("--n_layers", type=int, default=None)
    p_bench.add_argument("--device", default="cpu")
    p_bench.set_defaults(func=command_benchmark)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
