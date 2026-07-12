#!/usr/bin/env python3
"""Compare Jigsaw training logs by epoch loss and LR.

Example:
  python scripts/compare_training_progress.py \
    --run control=runs/logs/coverage_v2_allpos_fresh_direct_20260606_011834.out.log \
    --run sched=runs/logs/coverage_v2_khop_sched_e60_direct_YYYYMMDD_HHMMSS.out.log \
    --from-epoch 60
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import mean


SUMMARY_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s+Summary:\s+Avg Loss\s+=\s+(?P<loss>[0-9.]+),\s+LR\s+=\s+(?P<lr>[0-9.eE+-]+)"
)


def parse_log(path: Path) -> dict[int, tuple[float, float]]:
    rows: dict[int, tuple[float, float]] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = SUMMARY_RE.search(line)
        if not match:
            continue
        epoch = int(match.group("epoch"))
        loss = float(match.group("loss"))
        lr = float(match.group("lr"))
        rows[epoch] = (loss, lr)
    return rows


def tail_mean(values: list[float], n: int = 5) -> float:
    if not values:
        return float("nan")
    return mean(values[-n:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="label=path training log. Pass more than once.",
    )
    parser.add_argument("--from-epoch", type=int, default=0)
    parser.add_argument("--tail", type=int, default=5)
    args = parser.parse_args()

    parsed: dict[str, dict[int, tuple[float, float]]] = {}
    for item in args.run:
        if "=" not in item:
            raise SystemExit(f"--run must be label=path, got: {item}")
        label, raw_path = item.split("=", 1)
        path = Path(raw_path)
        if not path.exists():
            raise SystemExit(f"Missing log for {label}: {path}")
        rows = {
            epoch: vals
            for epoch, vals in parse_log(path).items()
            if epoch >= args.from_epoch
        }
        parsed[label] = rows

    print("run,epochs,last_epoch,start_loss,last_loss,best_loss,tail_avg_loss,last_lr")
    for label, rows in parsed.items():
        epochs = sorted(rows)
        if not epochs:
            print(f"{label},0,,,,,,")
            continue
        losses = [rows[e][0] for e in epochs]
        last_lr = rows[epochs[-1]][1]
        print(
            f"{label},{len(epochs)},{epochs[-1]},"
            f"{losses[0]:.6f},{losses[-1]:.6f},{min(losses):.6f},"
            f"{tail_mean(losses, args.tail):.6f},{last_lr:.3e}"
        )

    if len(parsed) == 2:
        labels = list(parsed)
        a_label, b_label = labels[0], labels[1]
        common = sorted(set(parsed[a_label]) & set(parsed[b_label]))
        if common:
            deltas = [
                parsed[b_label][epoch][0] - parsed[a_label][epoch][0]
                for epoch in common
            ]
            print()
            print(
                f"overlap,{a_label},{b_label},epochs={common[0]}-{common[-1]},"
                f"mean_loss_delta_b_minus_a={mean(deltas):.6f},"
                f"last_loss_delta_b_minus_a={deltas[-1]:.6f}"
            )


if __name__ == "__main__":
    main()
