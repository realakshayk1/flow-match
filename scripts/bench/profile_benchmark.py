import argparse
import cProfile
import json
import pstats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.bench.run_benchmark import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile benchmark execution with cProfile")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-n", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profiler = cProfile.Profile()
    profiler.enable()
    summary = run(config_path=Path(args.config), out_dir=out_dir)
    profiler.disable()

    stats_path = out_dir / "cprofile_stats.txt"
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.dump_stats(str(out_dir / "cprofile_stats.prof"))
    stats.stream = stats_path.open("w", encoding="utf-8")
    stats.print_stats(args.top_n)
    stats.stream.close()

    profile_json = {
        "top_n": args.top_n,
        "stage_profile": summary.get("stage_profile", {}),
        "recommendation": "Most cumulative time is expected in subprocess execution stages (run_*). Prioritize batching and minimizing per-ligand process startup by introducing a persistent worker for each backend.",
    }
    (out_dir / "profile_summary.json").write_text(json.dumps(profile_json, indent=2), encoding="utf-8")
    print(json.dumps(profile_json, indent=2))


if __name__ == "__main__":
    main()
