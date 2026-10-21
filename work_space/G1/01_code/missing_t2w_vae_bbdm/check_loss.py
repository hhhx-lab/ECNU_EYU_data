"""Quick loss curve check without TensorBoard.

Prints first/last/mean/min loss and smoothness from tfevents logs.

Usage:
    python check_loss.py                          # BBDM loss
    python check_loss.py --logdir training/endec/logs  # EncDec loss
"""
import os
import argparse
import configs
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default=os.path.join(configs.PATH_TRAINING, "bbdm", "logs"))
    args = parser.parse_args()

    # Find latest log subdirectory
    subdirs = sorted([d for d in os.listdir(args.logdir) if d.startswith("logs_")])
    if not subdirs:
        print(f"No log directories found under {args.logdir}")
        return
    latest = os.path.join(args.logdir, subdirs[-1])
    print(f"Reading: {latest}\n")

    ea = event_accumulator.EventAccumulator(latest)
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    if not tags:
        print("No scalar data found.")
        return

    for tag in sorted(tags):
        events = ea.Scalars(tag)
        if not events:
            continue
        values = [e.value for e in events]
        steps = [e.step for e in events]

        # Smoothness: std of differences between consecutive values
        diffs = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
        jitter = sum(diffs) / len(diffs) if diffs else 0

        print(f"--- {tag} ---")
        print(f"  steps:     {steps[0]} -> {steps[-1]}  ({len(steps)} points)")
        print(f"  first:     {values[0]:.6f}")
        print(f"  last:      {values[-1]:.6f}")
        print(f"  min:       {min(values):.6f}")
        print(f"  max:       {max(values):.6f}")
        print(f"  mean:      {sum(values)/len(values):.6f}")
        print(f"  jitter:    {jitter:.6f}  (lower = smoother)")

        # Trend check
        if len(values) >= 10:
            first_half = sum(values[:len(values)//2]) / (len(values)//2)
            second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
            trend = "↓ decreasing" if second_half < first_half else "↑ increasing"
            print(f"  trend:     {trend}  (first_half={first_half:.6f}, second_half={second_half:.6f})")
        print()


if __name__ == "__main__":
    main()
