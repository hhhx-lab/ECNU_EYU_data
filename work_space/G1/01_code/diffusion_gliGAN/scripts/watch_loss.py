"""
Monitor training loss in real time from a second terminal.

Usage (static, one-shot):
    python scripts/watch_loss.py ../../Checkpoint/quick_test t1c

Usage (live, auto-refresh every 10s):
    python scripts/watch_loss.py ../../Checkpoint/quick_test t1c --live

The script reads the human-readable .log file written by save_losses(),
so it works over SSH without needing a display. For a graphical plot,
add --plot (requires matplotlib).

Requires: matplotlib (only with --plot)
"""
import argparse
import os
import sys
import time


def read_log(checkpoint_dir, modality):
    logpath = os.path.join(checkpoint_dir, modality, "loss_lists", "loss_diffusion.log")
    if not os.path.isfile(logpath):
        print(f"[WARN] {logpath} not found yet, waiting...")
        return [], []
    epochs, values = [], []
    with open(logpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    epochs.append(int(parts[0]))
                    values.append(float(parts[1]))
                except ValueError:
                    pass
    return epochs, values


def terminal_plot(values, width=60, height=8):
    """ASCII sparkline-style plot in terminal."""
    if len(values) < 2:
        print("  (need at least 2 data points for a plot)")
        return
    vmin, vmax = min(values), max(values)
    if vmin == vmax:
        vmin, vmax = vmin - 1, vmin + 1
    eps = (vmax - vmin) / (height - 1)
    rows = [[" "] * len(values) for _ in range(height)]

    for i, v in enumerate(values):
        row_idx = min(height - 1, int((vmax - v) / eps))
        rows[row_idx][i] = "█"

    # Skip empty rows at top/bottom
    non_empty = [i for i, r in enumerate(rows) if any(c != " " for c in r)]
    if non_empty:
        rmin, rmax = min(non_empty), max(non_empty)
    else:
        rmin, rmax = 0, height - 1

    # Stride: downsample to fit width
    stride = max(1, len(values) // width)
    for i in range(rmin, rmax + 1):
        row_display = "".join(rows[i][::stride])[:width]
        val = vmax - (i + 0.5) * eps
        if i == rmin:
            print(f"{val:>8.4f} │{row_display}")
        elif i == rmax:
            print(f"{val:>8.4f} │{row_display}")
        elif i == (rmin + rmax) // 2:
            print(f"         │{row_display}")
        else:
            print(f"         │{row_display}")
    print(f"         └{'─' * min(width, len(values) // stride)}")
    print(f"          epoch {min(epochs, default=0)} → {max(epochs, default=0)}")


def main():
    parser = argparse.ArgumentParser(description="Watch training loss")
    parser.add_argument("checkpoint_dir", type=str, help="Path to Checkpoint/exp_name")
    parser.add_argument("modality", type=str, default="t1c", nargs="?",
                        help="Modality (default: t1c)")
    parser.add_argument("--live", action="store_true",
                        help="Auto-refresh (default: one-shot)")
    parser.add_argument("--interval", type=int, default=10,
                        help="Refresh interval in seconds (default: 10)")
    parser.add_argument("--plot", action="store_true",
                        help="Use matplotlib GUI instead of terminal plot")
    args = parser.parse_args()

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[ERROR] matplotlib not installed. pip install matplotlib")
            sys.exit(1)

    prev_len = 0
    while True:
        epochs, values = read_log(args.checkpoint_dir, args.modality)

        if not values:
            print(f"[{time.strftime('%H:%M:%S')}] waiting for data...")
        else:
            os.system("cls" if os.name == "nt" else "clear")
            print(f"=== Loss Curve: {args.modality}  [{time.strftime('%H:%M:%S')}] ===\n")

            if args.plot:
                import matplotlib.pyplot as plt
                plt.clf()
                plt.plot(epochs, values, "b-", marker=".")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"Diffusion Loss — {args.modality}")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.pause(0.1)
            else:
                print(f"  latest epoch {epochs[-1]}: loss = {values[-1]:.6f}  "
                      f"(trend: {'↓' if len(values) >= 2 and values[-1] < values[-2] else '↑'})\n")
                terminal_plot(values)

        if not args.live:
            break

        prev_len = len(values)
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[stop]")
            break


if __name__ == "__main__":
    main()
