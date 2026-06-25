import argparse
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic stub backend for throughput smoke tests")
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--ligand", required=True)
    parser.add_argument("--work-ms", type=float, default=10.0)
    args = parser.parse_args()

    # Simulate backend-specific latency differences while keeping deterministic behavior.
    bias = 2.0 if args.device == "cuda" else 0.0
    sleep_s = max(0.0, (args.work_ms + bias) / 1000.0)
    time.sleep(sleep_s)
    print(f"ok device={args.device} ligand={args.ligand}")


if __name__ == "__main__":
    main()
import argparse
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic stub backend for throughput smoke tests")
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--ligand", required=True)
    parser.add_argument("--work-ms", type=float, default=10.0)
    args = parser.parse_args()

    # Simulate backend-specific latency differences while keeping deterministic behavior.
    bias = 2.0 if args.device == "cuda" else 0.0
    sleep_s = max(0.0, (args.work_ms + bias) / 1000.0)
    time.sleep(sleep_s)
    print(f"ok device={args.device} ligand={args.ligand}")


if __name__ == "__main__":
    main()
