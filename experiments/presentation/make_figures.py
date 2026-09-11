#!/usr/bin/env python3
"""Generate exact, slide-ready figures from the committed experiment CSVs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paged-kv-traversal-study" / "results"
OUT = Path(__file__).resolve().parent / "figures"

NAVY = "#172A46"
BLUE = "#2F6BFF"
TEAL = "#00A7A5"
ORANGE = "#F28E2B"
PURPLE = "#7A52C7"
GRID = "#D9E2EC"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": NAVY,
            "axes.labelcolor": NAVY,
            "axes.titlecolor": NAVY,
            "xtick.color": NAVY,
            "ytick.color": NAVY,
            "text.color": NAVY,
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titleweight": "bold",
        }
    )


def annotate_bars(ax: plt.Axes, bars) -> None:
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{bar.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=NAVY,
        )


def phase1_figure() -> None:
    traversals = ["BNHD", "BHND", "HBND"]
    seq_rows = read_csv(RESULTS / "job-7616" / "phase1-sequential.csv")
    shuf_rows = read_csv(RESULTS / "job-7616" / "phase1-shuffled.csv")
    seq = {row["traversal"]: float(row["median_ms"]) for row in seq_rows}
    shuf = {row["traversal"]: float(row["median_ms"]) for row in shuf_rows}

    x = np.arange(len(traversals))
    width = 0.34
    fig, ax = plt.subplots(figsize=(12, 6.75))
    bars1 = ax.bar(x - width / 2, [seq[t] for t in traversals], width,
                   color=BLUE, label="Sequential block table")
    bars2 = ax.bar(x + width / 2, [shuf[t] for t in traversals], width,
                   color=ORANGE, label="Fully shuffled block table")
    annotate_bars(ax, bars1)
    annotate_bars(ax, bars2)
    ax.set_xticks(x, traversals)
    ax.set_ylabel("Median one-token decode-kernel time (ms)")
    ax.set_title("Phase 1 — fixed BNHD memory: matching traversal wins")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.text(
        0.985,
        0.025,
        "AMD Slurm job 7616 · B=512, H=32, N=16, D=128 · float32",
        ha="right",
        va="bottom",
        fontsize=10,
        color="#52667A",
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(OUT / "phase1_fixed_layout.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def phase2_figure() -> None:
    layouts = ["BNHD", "BHND", "HBND"]
    paths = [
        RESULTS / "job-7616" / "phase2-sequential.csv",
        RESULTS / "job-7616" / "phase2-shuffled.csv",
    ]
    titles = ["Sequential physical blocks", "Fully shuffled physical blocks"]
    matrices = []
    for path in paths:
        values = {(row["memory_layout"], row["traversal"]):
                  float(row["median_ms"]) for row in read_csv(path)}
        matrices.append(np.array([[values[(m, t)] for t in layouts]
                                  for m in layouts]))

    vmin = min(matrix.min() for matrix in matrices)
    vmax = max(matrix.max() for matrix in matrices)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.6), constrained_layout=True)
    for ax, matrix, title in zip(axes, matrices, titles):
        image = ax.imshow(matrix, cmap="YlGnBu_r", vmin=vmin, vmax=vmax)
        for row in range(3):
            for col in range(3):
                is_best = matrix[row, col] == matrix.min()
                ax.text(
                    col,
                    row,
                    f"{matrix[row, col]:.1f} ms",
                    ha="center",
                    va="center",
                    color="white" if matrix[row, col] > 54 else NAVY,
                    fontsize=12,
                    fontweight="bold" if is_best else "normal",
                )
        ax.set_xticks(range(3), layouts)
        ax.set_yticks(range(3), layouts)
        ax.set_xlabel("Traversal order")
        ax.set_ylabel("Physical memory layout")
        ax.set_title(title)
        for index in range(3):
            ax.add_patch(
                plt.Rectangle(
                    (index - 0.48, index - 0.48),
                    0.96,
                    0.96,
                    fill=False,
                    edgecolor=TEAL,
                    linewidth=3,
                )
            )
    colorbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.03)
    colorbar.set_label("Median decode-kernel time (ms)")
    fig.suptitle(
        "Phase 2 — matched traversal wins 3/3 layouts; fragmentation changes the global winner",
        fontsize=16,
        fontweight="bold",
        color=NAVY,
    )
    fig.text(
        0.5,
        0.93,
        "AMD Slurm job 7616 · teal outline = memory-matched traversal",
        ha="center",
        va="top",
        fontsize=10,
        color="#52667A",
    )
    fig.savefig(OUT / "phase2_layout_traversal_heatmaps.png", dpi=220,
                bbox_inches="tight")
    plt.close(fig)


def crossover_figure() -> None:
    rows = read_csv(RESULTS / "crossover-7617" / "crossover-summary.csv")
    runs = np.array([int(row["run_length"]) for row in rows])
    fig, ax = plt.subplots(figsize=(12, 6.75))
    for key, label, color, marker in [
        ("bnhd_median_ms", "BNHD matched", BLUE, "o"),
        ("bhnd_median_ms", "BHND matched", ORANGE, "s"),
        ("hbnd_median_ms", "HBND matched", TEAL, "^"),
    ]:
        ax.plot(
            np.arange(len(runs)),
            [float(row[key]) for row in rows],
            marker=marker,
            linewidth=2.7,
            markersize=7,
            color=color,
            label=label,
        )
    ax.axvspan(7.35, 8.65, color=PURPLE, alpha=0.10)
    ax.text(8.0, 32.62, "crossover region", ha="center", color=PURPLE,
            fontsize=11, fontweight="bold")
    ax.set_xticks(np.arange(len(runs)), [str(run) for run in runs])
    ax.set_xlabel("Contiguous physical-block run length  →  more fragmented")
    ax.set_ylabel("Median one-token decode-kernel time (ms)")
    ax.set_title("AMD timing sweep — fragmentation flips HBND to BHND")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.text(
        0.99,
        0.04,
        "Job 7617 · five seeds · B=512, H=32, N=16, D=128 · 256 MiB K+V",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="#52667A",
    )
    fig.tight_layout()
    fig.savefig(OUT / "crossover_amd_job7617.png", dpi=220,
                bbox_inches="tight")
    plt.close(fig)


def uprof_figure() -> None:
    rows = read_csv(RESULTS / "uprof-7775" / "uprof-counters.csv")
    by_case = {(int(row["run_length"]), row["layout"]): row for row in rows}
    runs = [8, 4, 3, 2, 1]

    def median(field: str, run: int, layout: str) -> float:
        values = [
            float(row[field])
            for row in rows
            if int(row["run_length"]) == run and row["layout"] == layout
        ]
        return float(np.median(values))

    x = np.arange(len(runs))
    time_ratio = [median("median_ms", r, "HBND") /
                  median("median_ms", r, "BHND") for r in runs]
    bhnd_ipc = [median("ipc", r, "BHND") for r in runs]
    hbnd_ipc = [median("ipc", r, "HBND") for r in runs]
    bhnd_miss = [median("l3_miss_per_thousand_instructions", r, "BHND")
                 for r in runs]
    hbnd_miss = [median("l3_miss_per_thousand_instructions", r, "HBND")
                 for r in runs]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.6), constrained_layout=True)
    axes[0].plot(x, time_ratio, marker="o", color=PURPLE, linewidth=2.7)
    axes[0].axhline(1.0, color=NAVY, linestyle="--", linewidth=1.2)
    axes[0].fill_between(x, 0.98, 1.02, color=GRID, alpha=0.65)
    axes[0].set_title("Timing: HBND / BHND")
    axes[0].set_ylabel("Ratio (>1 means BHND wins)")

    axes[1].plot(x, bhnd_ipc, marker="s", color=ORANGE, linewidth=2.7,
                 label="BHND")
    axes[1].plot(x, hbnd_ipc, marker="^", color=TEAL, linewidth=2.7,
                 label="HBND")
    axes[1].set_title("IPC")
    axes[1].set_ylabel("Instructions per cycle")
    axes[1].legend(frameon=False)

    axes[2].plot(x, bhnd_miss, marker="s", color=ORANGE, linewidth=2.7,
                 label="BHND")
    axes[2].plot(x, hbnd_miss, marker="^", color=TEAL, linewidth=2.7,
                 label="HBND")
    axes[2].set_title("Normalized L3 misses")
    axes[2].set_ylabel("L3 misses / 1,000 retired instructions")
    axes[2].legend(frameon=False)

    for ax in axes:
        ax.set_xticks(x, [str(run) for run in runs])
        ax.set_xlabel("Contiguous run length  →  fragmented")
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    fig.suptitle(
        "AMD uProf repeat — crossover tracks falling HBND IPC, not rising miss percentage",
        fontsize=16,
        fontweight="bold",
        color=NAVY,
    )
    fig.text(
        0.5,
        -0.02,
        "mn01 job 7775 · uProf 5.3.521 MSR mode · core 0 / CCX 0 · median of 3 seeds",
        ha="center",
        fontsize=10,
        color="#52667A",
    )
    fig.savefig(OUT / "uprof_amd_job7775.png", dpi=220,
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    set_style()
    phase1_figure()
    phase2_figure()
    crossover_figure()
    uprof_figure()
    for path in sorted(OUT.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
