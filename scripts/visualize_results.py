#!/usr/bin/env python3
"""
Visualization suite for parallel HGS-CVRP benchmark results.
Run from repo root:  python scripts/visualize_results.py
Outputs PNG files to results/plots/.
"""

import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")
OUT_DIR = Path("results/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Ordering & aesthetics ──────────────────────────────────────────────────────
PRESET_ORDER = [
    "baseline",
    "gpu",
    "offspring16t8",
    "gpu_offspring16t8",
    "island8_star",
    "island8_star_gpu_off16t8",
]
PRESET_LABELS = {
    "baseline":                 "Baseline",
    "gpu":                      "GPU SWAP*",
    "offspring16t8":            "Offspring×16\n(8 threads)",
    "gpu_offspring16t8":        "GPU + Offspring×16",
    "island8_star":             "8 Islands (star)",
    "island8_star_gpu_off16t8": "8 Islands +\nGPU + Offspring",
}
PRESET_LABELS_SHORT = {
    "baseline":                 "Baseline",
    "gpu":                      "GPU",
    "offspring16t8":            "Offspring×16",
    "gpu_offspring16t8":        "GPU+Off",
    "island8_star":             "Island×8",
    "island8_star_gpu_off16t8": "Island+GPU+Off",
}
PRESET_COLORS = {
    "baseline":                 "#6B7280",
    "gpu":                      "#2563EB",
    "offspring16t8":            "#D97706",
    "gpu_offspring16t8":        "#7C3AED",
    "island8_star":             "#059669",
    "island8_star_gpu_off16t8": "#DC2626",
}

INSTANCE_ORDER = [
    "XL-n2494-k194",
    "XL-n2354-k631",
    "XL-n5174-k55",
    "XL-n5406-k783",
    "XL-n5288-k1246",
    "XL-n10001-k1570",
    "XL-n2634-k17",
]
INSTANCE_LABELS = {
    "XL-n2494-k194":   "n2494\n(194 routes)",
    "XL-n2354-k631":   "n2354\n(631 routes)",
    "XL-n5174-k55":    "n5174\n(55 routes)",
    "XL-n5406-k783":   "n5406\n(783 routes)",
    "XL-n5288-k1246":  "n5288\n(1246 routes)",
    "XL-n10001-k1570": "n10001\n(1570 routes)",
    "XL-n2634-k17":    "n2634\n(17 routes)",
}
INSTANCE_LABELS_SHORT = {k: k.replace("XL-", "") for k in INSTANCE_ORDER}

N_CLIENTS = {
    "XL-n2494-k194":   2493,
    "XL-n2354-k631":   2353,
    "XL-n5174-k55":    5173,
    "XL-n5406-k783":   5405,
    "XL-n5288-k1246":  5287,
    "XL-n10001-k1570": 10000,
    "XL-n2634-k17":    2633,
}

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         12,
    "axes.titlesize":    14,
    "axes.labelsize":    13,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   11,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

# ── Parser ─────────────────────────────────────────────────────────────────────

def parse_stdout(path: Path) -> dict:
    text = path.read_text(errors="replace")
    r = dict(
        instance=None, mode=None, preset=None,
        best=None, iter=None, time_s=None,
        ls_calls=0, ls_time_ms=0, ri_moves=0, ri_time_ms=0, ss_time_ms=0,
        pairs_eval=0, moves_applied=0, kernel_calls=0,
        offspring_rounds=0, offspring_generated=0,
        migrations_sent=None, migrations_received=None,
        cpu_util=None, gpu_core_util=None, gpu_mem_util=None,
        trace=[],
    )

    def add(key, val):
        r[key] = (r[key] or 0) + val

    seen_mode = seen_best = seen_iter = seen_time = False

    for raw_line in text.splitlines():
        s = raw_line.strip()

        m = re.match(r"instance\s*:\s*(.+)", s)
        if m and r["instance"] is None:
            r["instance"] = Path(m.group(1).strip()).stem; continue

        m = re.match(r"mode\s*:\s*(.+)", s)
        if m and not seen_mode:
            r["mode"] = m.group(1).strip(); seen_mode = True; continue

        m = re.match(r"best\s*:\s*([\d.]+)", s)
        if m and not seen_best:
            r["best"] = float(m.group(1)); seen_best = True; continue

        m = re.match(r"iter\s*:\s*(\d+)", s)
        if m and not seen_iter:
            r["iter"] = int(m.group(1)); seen_iter = True; continue

        m = re.match(r"time\(s\)\s*:\s*([\d.]+)", s)
        if m and not seen_time:
            r["time_s"] = float(m.group(1)); seen_time = True; continue

        m = re.match(r"LS calls\s*:\s*(\d+)", s)
        if m: add("ls_calls", int(m.group(1))); continue

        m = re.match(r"LS time\(ms\)\s*:\s*(\d+)", s)
        if m: add("ls_time_ms", int(m.group(1))); continue

        m = re.match(r"RI moves\s*:\s*(\d+)", s)
        if m: add("ri_moves", int(m.group(1))); continue

        m = re.match(r"RI time\(ms\)\s*:\s*(\d+)", s)
        if m: add("ri_time_ms", int(m.group(1))); continue

        m = re.match(r"SS time\(ms\)\s*:\s*(\d+)", s)
        if m: add("ss_time_ms", int(m.group(1))); continue

        m = re.match(r"pairs eval\s*:\s*([\d,]+)", s)
        if m: add("pairs_eval", int(m.group(1).replace(",", ""))); continue

        m = re.match(r"moves applied\s*:\s*(\d+)", s)
        if m: add("moves_applied", int(m.group(1))); continue

        m = re.match(r"kernel calls\s*:\s*(\d+)", s)
        if m: add("kernel_calls", int(m.group(1))); continue

        m = re.match(r"rounds\s*:\s*(\d+)", s)
        if m: add("offspring_rounds", int(m.group(1))); continue

        m = re.match(r"generated\s*:\s*(\d+)", s)
        if m: add("offspring_generated", int(m.group(1))); continue

        m = re.match(r"sent\s*:\s*(\d+)", s)
        if m: r["migrations_sent"] = int(m.group(1)); continue

        m = re.match(r"received\s*:\s*(\d+)", s)
        if m: r["migrations_received"] = int(m.group(1)); continue

        m = re.match(r"CPU avg util\s*:\s*([\d.]+)", s)
        if m: r["cpu_util"] = float(m.group(1)); continue

        m = re.match(r"GPU core avg\s*:\s*([\d.]+)", s)
        if m: r["gpu_core_util"] = float(m.group(1)); continue

        m = re.match(r"GPU mem\s+avg\s*:\s*([\d.]+)", s)
        if m: r["gpu_mem_util"] = float(m.group(1)); continue

    # Convergence trace: "It <i> <n> | T(s) <t> | Feas <c> <best> ..."
    trace_raw = []
    for raw_line in text.splitlines():
        m = re.match(
            r"It\s+\d+\s+\d+\s+\|\s+T\(s\)\s+([\d.]+)\s+\|\s+Feas\s+\d+\s+([\d.]+)",
            raw_line.strip(),
        )
        if m:
            trace_raw.append((float(m.group(1)), float(m.group(2))))

    if trace_raw:
        trace_raw.sort(key=lambda x: x[0])
        run_min = float("inf")
        trace = []
        for t, val in trace_raw:
            run_min = min(run_min, val)
            trace.append((t, run_min))
        r["trace"] = trace

    return r


def scan_results(results_dir: Path) -> dict:
    """Return data[instance][preset] = metrics.  Later run-dirs overwrite earlier."""
    data: dict[str, dict[str, dict]] = defaultdict(dict)

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        for sub in sorted(run_dir.iterdir()):
            if not sub.is_dir():
                continue
            m = re.match(r"\d+__(.+)__(.+)", sub.name)
            if not m:
                continue
            instance, preset = m.group(1), m.group(2)
            stdout = sub / "stdout.log"
            if not stdout.exists():
                continue
            r = parse_stdout(stdout)
            if r["best"] is None:
                continue
            r["preset"] = preset
            data[instance][preset] = r

    return data


# ── Helpers ───────────────────────────────────────────────────────────────────

def quality_delta_pct(data, instance, preset):
    """% improvement vs baseline (positive = better)."""
    base = data[instance].get("baseline")
    cur  = data[instance].get(preset)
    if base is None or cur is None:
        return None
    return (base["best"] - cur["best"]) / base["best"] * 100


def savefig(fig, name):
    p = OUT_DIR / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {p}")


# ── Figure 1 — Solution quality improvement vs baseline ───────────────────────

def plot_quality_improvement(data):
    instances = [i for i in INSTANCE_ORDER if i in data]
    non_base  = [p for p in PRESET_ORDER if p != "baseline"]

    n_inst    = len(instances)
    n_preset  = len(non_base)
    bar_w     = 0.13
    x         = np.arange(n_inst)

    fig, ax = plt.subplots(figsize=(14, 6))

    for j, preset in enumerate(non_base):
        vals = []
        for inst in instances:
            d = quality_delta_pct(data, inst, preset)
            vals.append(d if d is not None else np.nan)
        vals = np.array(vals, dtype=float)

        offset = (j - n_preset / 2 + 0.5) * bar_w
        bars = ax.bar(
            x + offset, vals,
            width=bar_w,
            color=PRESET_COLORS[preset],
            label=PRESET_LABELS_SHORT[preset],
            edgecolor="white", linewidth=0.5,
        )
        # mark NaN / failed
        for xi, v in zip(x + offset, vals):
            if np.isnan(v):
                ax.text(xi, 0.02, "N/A", ha="center", va="bottom",
                        fontsize=7, color="gray")

    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([INSTANCE_LABELS[i] for i in instances])
    ax.set_ylabel("Solution quality improvement vs Baseline (%)")
    ax.set_title("Solution Quality Improvement Over Baseline  (positive = better, 600 s time limit)")
    ax.legend(loc="upper right", ncol=3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))

    savefig(fig, "01_quality_improvement.png")


# ── Figure 2 — Convergence curves ─────────────────────────────────────────────

def plot_convergence(data):
    instances = [i for i in INSTANCE_ORDER if i in data]
    # Pick a few representative / interesting instances for main figure
    focus = ["XL-n2494-k194", "XL-n5406-k783", "XL-n2354-k631",
             "XL-n5174-k55",  "XL-n5288-k1246", "XL-n10001-k1570"]
    focus = [i for i in focus if i in data]

    ncols = 3
    nrows = (len(focus) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4.5))
    axes = np.array(axes).flatten()

    for ax_idx, inst in enumerate(focus):
        ax = axes[ax_idx]
        base_best = data[inst].get("baseline", {}).get("best", None)

        for preset in PRESET_ORDER:
            r = data[inst].get(preset)
            if r is None or not r["trace"]:
                continue
            times = [p[0] for p in r["trace"]]
            vals  = [p[1] for p in r["trace"]]

            # Normalize to baseline's final best (% above)
            if base_best is not None:
                norm = [(v / base_best - 1) * 100 for v in vals]
            else:
                norm = vals

            label = PRESET_LABELS_SHORT[preset]
            lw = 2.0 if preset != "baseline" else 1.5
            ax.plot(times, norm, color=PRESET_COLORS[preset],
                    label=label, linewidth=lw,
                    alpha=0.9 if preset != "baseline" else 0.7)

        ax.axhline(0, color="#6B7280", linewidth=1, linestyle="--", alpha=0.5)
        ax.set_xlim(0, 620)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Gap to baseline final (%)")
        ax.set_title(INSTANCE_LABELS[inst].replace("\n", "  "))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))

    # shared legend
    handles = [mpatches.Patch(color=PRESET_COLORS[p],
                              label=PRESET_LABELS_SHORT[p]) for p in PRESET_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04), frameon=True)

    # hide unused axes
    for ax in axes[len(focus):]:
        ax.set_visible(False)

    fig.suptitle("Convergence: Best Feasible Solution Over Time\n(normalized — 0 % = baseline final best)",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    savefig(fig, "02_convergence_curves.png")


# ── Figure 3 — Hardware utilization ───────────────────────────────────────────

def plot_hw_utilization(data):
    # Average CPU/GPU across all instances per preset
    presets = PRESET_ORDER
    instances = [i for i in INSTANCE_ORDER if i in data]

    cpu_mat  = np.full((len(presets), len(instances)), np.nan)
    gpu_mat  = np.full((len(presets), len(instances)), np.nan)

    for i, preset in enumerate(presets):
        for j, inst in enumerate(instances):
            r = data[inst].get(preset)
            if r is None:
                continue
            if r["cpu_util"] is not None:
                cpu_mat[i, j] = r["cpu_util"]
            if r["gpu_core_util"] is not None:
                gpu_mat[i, j] = r["gpu_core_util"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    inst_labels = [INSTANCE_LABELS[i].replace("\n", " ") for i in instances]
    preset_labels = [PRESET_LABELS_SHORT[p] for p in presets]

    for ax, mat, title, cmap, vmax in zip(
        axes,
        [cpu_mat, gpu_mat],
        ["CPU Utilization (%)", "GPU Core Utilization (%)"],
        ["YlOrBr", "Blues"],
        [100, 100],
    ):
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(range(len(instances)))
        ax.set_xticklabels(inst_labels, rotation=30, ha="right", fontsize=10)
        ax.set_yticks(range(len(presets)))
        ax.set_yticklabels(preset_labels, fontsize=10)
        ax.set_title(title, fontsize=13)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        # annotate cells
        for i in range(len(presets)):
            for j in range(len(instances)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.0f}%", ha="center", va="center",
                            fontsize=9, color="black" if mat[i,j] < 60 else "white")
                else:
                    ax.text(j, i, "N/A", ha="center", va="center",
                            fontsize=8, color="#aaa")

    fig.suptitle("Hardware Utilization per Preset and Instance", fontsize=14)
    fig.tight_layout()
    savefig(fig, "03_hw_utilization_heatmap.png")


# ── Figure 4 — SWAP* pair evaluations ─────────────────────────────────────────

def plot_pairs_evaluated(data):
    instances = [i for i in INSTANCE_ORDER if i in data]
    non_base  = [p for p in PRESET_ORDER if p != "baseline"]

    n_inst   = len(instances)
    n_preset = len(non_base)
    bar_w    = 0.13
    x        = np.arange(n_inst)

    fig, ax = plt.subplots(figsize=(14, 6))

    # Normalize to baseline pairs
    for j, preset in enumerate(non_base):
        ratios = []
        for inst in instances:
            base = data[inst].get("baseline")
            cur  = data[inst].get(preset)
            if base is None or cur is None or base["pairs_eval"] == 0:
                ratios.append(np.nan)
            else:
                ratios.append(cur["pairs_eval"] / base["pairs_eval"])
        ratios = np.array(ratios, dtype=float)

        offset = (j - n_preset / 2 + 0.5) * bar_w
        ax.bar(x + offset, ratios, width=bar_w,
               color=PRESET_COLORS[preset], label=PRESET_LABELS_SHORT[preset],
               edgecolor="white", linewidth=0.5)

    ax.axhline(1, color="black", linewidth=1.2, linestyle="--", label="Baseline (×1)")
    ax.set_xticks(x)
    ax.set_xticklabels([INSTANCE_LABELS[i] for i in instances])
    ax.set_ylabel("SWAP* pairs evaluated  (×  vs Baseline)")
    ax.set_title("SWAP* Pair Evaluation Volume  (relative to Baseline, in 600 s)")
    ax.legend(loc="upper right", ncol=3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f×"))

    savefig(fig, "04_swap_pairs_evaluated.png")


# ── Figure 5 — LS calls throughput ────────────────────────────────────────────

def plot_ls_throughput(data):
    instances = [i for i in INSTANCE_ORDER if i in data]
    n_inst    = len(instances)
    n_preset  = len(PRESET_ORDER)
    bar_w     = 0.12
    x         = np.arange(n_inst)

    fig, ax = plt.subplots(figsize=(15, 6))

    for j, preset in enumerate(PRESET_ORDER):
        vals = []
        for inst in instances:
            r = data[inst].get(preset)
            vals.append(r["ls_calls"] if r else np.nan)
        vals = np.array(vals, dtype=float)

        offset = (j - n_preset / 2 + 0.5) * bar_w
        ax.bar(x + offset, vals / 1000, width=bar_w,
               color=PRESET_COLORS[preset], label=PRESET_LABELS_SHORT[preset],
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([INSTANCE_LABELS[i] for i in instances])
    ax.set_ylabel("Local Search calls  (thousands)")
    ax.set_title("Local Search Calls Completed in 600 s")
    ax.legend(loc="upper right", ncol=3)

    savefig(fig, "05_ls_calls_throughput.png")


# ── Figure 6 — Time decomposition ─────────────────────────────────────────────

def plot_time_decomposition(data):
    # One representative instance per column
    focus     = ["XL-n2494-k194", "XL-n5406-k783", "XL-n2354-k631"]
    focus     = [i for i in focus if i in data]
    presets   = PRESET_ORDER
    n_p       = len(presets)
    n_inst    = len(focus)

    fig, axes = plt.subplots(1, n_inst, figsize=(n_inst * 6, 6), sharey=False)
    if n_inst == 1:
        axes = [axes]

    for ax, inst in zip(axes, focus):
        ri_times  = []
        ss_times  = []
        labels    = []
        colors_ri = []
        colors_ss = []

        for preset in presets:
            r = data[inst].get(preset)
            if r is None:
                ri_times.append(0); ss_times.append(0)
            else:
                ri_times.append(r["ri_time_ms"] / 1000)
                ss_times.append(r["ss_time_ms"] / 1000)
            labels.append(PRESET_LABELS_SHORT[preset])
            colors_ri.append(PRESET_COLORS[preset])
            colors_ss.append(PRESET_COLORS[preset])

        y = np.arange(n_p)
        bars1 = ax.barh(y, ri_times, color=[PRESET_COLORS[p] for p in presets],
                        alpha=0.9, label="Route Improvement (RI)")
        bars2 = ax.barh(y, ss_times, left=ri_times,
                        color=[PRESET_COLORS[p] for p in presets],
                        alpha=0.4, hatch="///", label="SWAP*")

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("CPU time (s)")
        ax.set_title(INSTANCE_LABELS[inst].replace("\n", "  "))

    # Legend
    ri_patch = mpatches.Patch(facecolor="#888", alpha=0.9, label="Route Improvement")
    ss_patch = mpatches.Patch(facecolor="#888", alpha=0.4, hatch="///", label="SWAP*")
    fig.legend(handles=[ri_patch, ss_patch], loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.06), frameon=True)

    fig.suptitle("Time Decomposition: Route Improvement vs SWAP*  (total CPU-time of rank-0, 600 s wall)", fontsize=13)
    fig.tight_layout()
    savefig(fig, "06_time_decomposition.png")


# ── Figure 7 — Solution quality heatmap ───────────────────────────────────────

def plot_quality_heatmap(data):
    instances = [i for i in INSTANCE_ORDER if i in data]
    presets   = [p for p in PRESET_ORDER if p != "baseline"]

    mat = np.full((len(presets), len(instances)), np.nan)
    for j, inst in enumerate(instances):
        for i, preset in enumerate(presets):
            d = quality_delta_pct(data, inst, preset)
            if d is not None:
                mat[i, j] = d

    fig, ax = plt.subplots(figsize=(13, 5))

    vmax = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 1
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", norm=norm)

    ax.set_xticks(range(len(instances)))
    ax.set_xticklabels(
        [INSTANCE_LABELS[i].replace("\n", " ") for i in instances],
        rotation=30, ha="right",
    )
    ax.set_yticks(range(len(presets)))
    ax.set_yticklabels([PRESET_LABELS_SHORT[p] for p in presets])

    for i in range(len(presets)):
        for j in range(len(instances)):
            if not np.isnan(mat[i, j]):
                txt = f"{mat[i,j]:+.2f}%"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="black" if abs(mat[i,j]) < vmax * 0.6 else "white")
            else:
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=9, color="#aaa")

    plt.colorbar(im, ax=ax, label="% improvement vs Baseline", fraction=0.03, pad=0.02)
    ax.set_title("Solution Quality Improvement vs Baseline  (green = better,  red = worse)", fontsize=13)
    fig.tight_layout()
    savefig(fig, "07_quality_heatmap.png")


# ── Figure 8 — Iterations completed ───────────────────────────────────────────

def plot_iterations(data):
    instances = [i for i in INSTANCE_ORDER if i in data]
    n_inst    = len(instances)
    n_preset  = len(PRESET_ORDER)
    bar_w     = 0.12
    x         = np.arange(n_inst)

    fig, ax = plt.subplots(figsize=(15, 6))

    for j, preset in enumerate(PRESET_ORDER):
        vals = []
        for inst in instances:
            r = data[inst].get(preset)
            vals.append(r["iter"] if r and r["iter"] is not None else np.nan)
        vals = np.array(vals, dtype=float)
        # For island model multiply by 8 (total iterations across all nodes)
        if "island" in preset:
            vals = vals * 8

        offset = (j - n_preset / 2 + 0.5) * bar_w
        ax.bar(x + offset, vals / 1000, width=bar_w,
               color=PRESET_COLORS[preset], label=PRESET_LABELS_SHORT[preset],
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([INSTANCE_LABELS[i] for i in instances])
    ax.set_ylabel("GA Iterations (thousands)")
    ax.set_title("GA Iterations Completed in 600 s  (island: total across 8 nodes)")
    ax.legend(loc="upper right", ncol=3)

    savefig(fig, "08_iterations_completed.png")


# ── Figure 9 — GPU SWAP* speedup (pairs/s) ────────────────────────────────────

def plot_swapstar_speedup(data):
    instances = [i for i in INSTANCE_ORDER if i in data]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: pairs/second CPU vs GPU (log scale)
    ax = axes[0]
    x = np.arange(len(instances))
    w = 0.35

    cpu_rates, gpu_rates = [], []
    for inst in instances:
        base = data[inst].get("baseline")
        gpu  = data[inst].get("gpu")
        off  = data[inst].get("offspring16t8")
        gpu_off = data[inst].get("gpu_offspring16t8")

        # CPU rate: from the preset that uses CPU SWAP*
        cpu_rate = (base["pairs_eval"] / (base["ss_time_ms"] / 1000)
                    if base and base["ss_time_ms"] > 0 else np.nan)
        gpu_rate = (gpu["pairs_eval"] / (gpu["ss_time_ms"] / 1000)
                    if gpu and gpu["ss_time_ms"] > 0 else np.nan)
        cpu_rates.append(cpu_rate)
        gpu_rates.append(gpu_rate)

    cpu_rates = np.array(cpu_rates, dtype=float)
    gpu_rates = np.array(gpu_rates, dtype=float)

    bars_c = ax.bar(x - w/2, cpu_rates / 1e6, w, label="CPU SWAP*",
                    color="#6B7280", edgecolor="white")
    bars_g = ax.bar(x + w/2, gpu_rates / 1e6, w, label="GPU SWAP*",
                    color="#2563EB", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([INSTANCE_LABELS_SHORT[i] for i in instances], rotation=20, ha="right")
    ax.set_ylabel("Pairs evaluated per second (millions)")
    ax.set_title("SWAP* Throughput: CPU vs GPU")
    ax.legend()

    # Right: speedup ratio
    ax2 = axes[1]
    speedup = gpu_rates / cpu_rates
    bars = ax2.bar(x, speedup, color="#2563EB", edgecolor="white")
    for bar, val in zip(bars, speedup):
        if not np.isnan(val):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f"{val:.1f}×", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.axhline(1, color="gray", linewidth=1, linestyle="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels([INSTANCE_LABELS_SHORT[i] for i in instances], rotation=20, ha="right")
    ax2.set_ylabel("GPU / CPU pairs-per-second  (×speedup)")
    ax2.set_title("GPU SWAP* Speedup over CPU SWAP*")

    fig.suptitle("SWAP* Evaluation: GPU vs CPU Throughput", fontsize=14)
    fig.tight_layout()
    savefig(fig, "09_swapstar_gpu_speedup.png")


# ── Figure 10 — Best quality: combined bar per instance ───────────────────────

def plot_absolute_best(data):
    instances = [i for i in INSTANCE_ORDER if i in data]
    n_inst    = len(instances)

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n_inst)
    bar_w = 0.12
    n_preset = len(PRESET_ORDER)

    for j, preset in enumerate(PRESET_ORDER):
        vals = []
        for inst in instances:
            r = data[inst].get(preset)
            # normalize by baseline
            base = data[inst].get("baseline")
            if r is None or base is None:
                vals.append(np.nan)
            else:
                vals.append(r["best"] / base["best"] * 100)

        vals = np.array(vals, dtype=float)
        offset = (j - n_preset / 2 + 0.5) * bar_w
        ax.bar(x + offset, vals, width=bar_w,
               color=PRESET_COLORS[preset],
               label=PRESET_LABELS_SHORT[preset],
               edgecolor="white", linewidth=0.5)

    ax.axhline(100, color="black", linewidth=1.2, linestyle="--", label="Baseline (100%)")
    ax.set_xticks(x)
    ax.set_xticklabels([INSTANCE_LABELS[i] for i in instances])
    ax.set_ylabel("Best solution  (% of Baseline)")
    ax.set_title("Best Solution Quality  (100% = Baseline best;  lower is better)")
    ax.legend(loc="upper right", ncol=3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    # Zoom to interesting range
    all_vals = []
    for preset in PRESET_ORDER:
        for inst in instances:
            r = data[inst].get(preset)
            base = data[inst].get("baseline")
            if r and base:
                all_vals.append(r["best"] / base["best"] * 100)
    if all_vals:
        lo = min(all_vals)
        hi = max(all_vals)
        margin = (hi - lo) * 0.3
        ax.set_ylim(max(lo - margin, lo - 1), hi + margin)

    savefig(fig, "10_absolute_best_quality.png")


# ── Figure 11 — Island model migration stats ───────────────────────────────────

def plot_island_migrations(data):
    instances = [i for i in INSTANCE_ORDER if i in data]
    island_presets = ["island8_star", "island8_star_gpu_off16t8"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, preset in zip(axes, island_presets):
        sent_vals = []
        recv_vals = []
        inst_labels = []
        for inst in instances:
            r = data[inst].get(preset)
            if r is None:
                continue
            sent = r["migrations_sent"]
            recv = r["migrations_received"]
            if sent is None and recv is None:
                continue
            sent_vals.append(sent or 0)
            recv_vals.append(recv or 0)
            inst_labels.append(INSTANCE_LABELS_SHORT[inst])

        y = np.arange(len(inst_labels))
        w = 0.35
        ax.barh(y - w/2, sent_vals, w, label="Sent",     color="#2563EB")
        ax.barh(y + w/2, recv_vals, w, label="Received", color="#059669")
        ax.set_yticks(y)
        ax.set_yticklabels(inst_labels)
        ax.set_xlabel("Migrations (rank-0)")
        ax.set_title(PRESET_LABELS_SHORT[preset])
        ax.legend()

    fig.suptitle("Island Model: Migrations per Instance (rank-0 node, 600 s)", fontsize=13)
    fig.tight_layout()
    savefig(fig, "11_island_migrations.png")


# ── Figure 12 — Offspring offspring rounds ────────────────────────────────────

def plot_offspring_throughput(data):
    instances = [i for i in INSTANCE_ORDER if i in data]
    off_presets = ["offspring16t8", "gpu_offspring16t8", "island8_star_gpu_off16t8"]

    n_inst   = len(instances)
    n_preset = len(off_presets)
    bar_w    = 0.25
    x        = np.arange(n_inst)

    fig, ax = plt.subplots(figsize=(14, 5.5))

    for j, preset in enumerate(off_presets):
        rounds = []
        for inst in instances:
            r = data[inst].get(preset)
            rounds.append(r["offspring_rounds"] if r else np.nan)
        rounds = np.array(rounds, dtype=float)

        offset = (j - n_preset / 2 + 0.5) * bar_w
        ax.bar(x + offset, rounds, width=bar_w,
               color=PRESET_COLORS[preset],
               label=PRESET_LABELS_SHORT[preset],
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([INSTANCE_LABELS[i] for i in instances])
    ax.set_ylabel("Offspring rounds completed")
    ax.set_title("Parallel Offspring Rounds in 600 s  (each round = 16 offspring using 8 threads)")
    ax.legend()

    savefig(fig, "12_offspring_rounds.png")


# ── Figure 13 — Radar / spider chart: multi-metric per preset ─────────────────

def plot_radar(data):
    """Radar chart comparing presets on several normalized metrics (avg over instances)."""
    instances = [i for i in INSTANCE_ORDER if i in data]

    metrics = {
        "Quality\nimprovement": lambda inst, p: quality_delta_pct(data, inst, p) or 0,
        "LS calls\n(relative)": lambda inst, p: (
            (data[inst][p]["ls_calls"] / data[inst]["baseline"]["ls_calls"])
            if p in data[inst] and "baseline" in data[inst] and data[inst]["baseline"]["ls_calls"] > 0
            else 1.0
        ),
        "Pairs evaluated\n(relative)": lambda inst, p: (
            (data[inst][p]["pairs_eval"] / data[inst]["baseline"]["pairs_eval"])
            if p in data[inst] and "baseline" in data[inst] and data[inst]["baseline"]["pairs_eval"] > 0
            else 1.0
        ),
        "CPU utilization\n(%)": lambda inst, p: (
            data[inst][p]["cpu_util"] / 100 if p in data[inst] and data[inst][p]["cpu_util"] is not None else 0
        ),
        "GPU utilization\n(%)": lambda inst, p: (
            data[inst][p]["gpu_core_util"] / 100 if p in data[inst] and data[inst][p]["gpu_core_util"] is not None else 0
        ),
    }
    metric_names = list(metrics.keys())
    N = len(metric_names)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9),
                             subplot_kw=dict(polar=True))
    axes = axes.flatten()

    for ax_i, preset in enumerate(PRESET_ORDER):
        ax = axes[ax_i]
        avg_vals = []
        for metric_fn in metrics.values():
            vals = []
            for inst in instances:
                try:
                    v = metric_fn(inst, preset)
                    if v is not None and not np.isnan(v):
                        vals.append(float(v))
                except Exception:
                    pass
            avg_vals.append(np.mean(vals) if vals else 0)

        # Normalize: quality improvement 0-2%, ratios already relative, util 0-1
        # Simple min-max per metric across all presets
        avg_vals += avg_vals[:1]

        ax.plot(angles, avg_vals, "o-", linewidth=2,
                color=PRESET_COLORS[preset])
        ax.fill(angles, avg_vals, alpha=0.25, color=PRESET_COLORS[preset])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, fontsize=9)
        ax.set_title(PRESET_LABELS_SHORT[preset], pad=15,
                     color=PRESET_COLORS[preset], fontweight="bold")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Multi-metric Profile per Parallelization Strategy\n(averaged over all instances)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    savefig(fig, "13_radar_profiles.png")


# ── Figure 14 — Convergence: detail for 2 best instances ─────────────────────

def plot_convergence_detail(data):
    focus = ["XL-n2494-k194", "XL-n5406-k783"]
    focus = [i for i in focus if i in data]

    fig, axes = plt.subplots(1, len(focus), figsize=(len(focus) * 7, 5.5))
    if len(focus) == 1:
        axes = [axes]

    for ax, inst in zip(axes, focus):
        base_best = data[inst].get("baseline", {}).get("best")

        for preset in PRESET_ORDER:
            r = data[inst].get(preset)
            if r is None or not r["trace"]:
                continue
            times = np.array([p[0] for p in r["trace"]])
            vals  = np.array([p[1] for p in r["trace"]])

            if base_best:
                vals = (vals / base_best - 1) * 100

            lw   = 2.5 if preset != "baseline" else 1.5
            dash = (4, 2) if preset == "baseline" else ()
            ax.plot(times, vals,
                    color=PRESET_COLORS[preset],
                    label=PRESET_LABELS_SHORT[preset],
                    linewidth=lw, dashes=dash, alpha=0.9)

        ax.axhline(0, color="#6B7280", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlim(0, 620)
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Gap to baseline final best (%)")
        ax.set_title(f"{INSTANCE_LABELS[inst].replace(chr(10), '  ')}  —  detail", fontsize=13)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
        ax.legend(loc="upper right", fontsize=10)

    fig.suptitle("Convergence Detail: Best Feasible Solution vs Time\n"
                 "(0% = baseline final best;  negative = better than baseline)",
                 fontsize=13)
    fig.tight_layout()
    savefig(fig, "14_convergence_detail.png")


# ── Figure 15 — RI moves throughput ──────────────────────────────────────────

def plot_ri_moves(data):
    instances = [i for i in INSTANCE_ORDER if i in data]
    n_inst    = len(instances)
    n_preset  = len(PRESET_ORDER)
    bar_w     = 0.12
    x         = np.arange(n_inst)

    fig, ax = plt.subplots(figsize=(15, 6))

    for j, preset in enumerate(PRESET_ORDER):
        vals = []
        for inst in instances:
            r = data[inst].get(preset)
            vals.append(r["ri_moves"] / 1e6 if r else np.nan)
        vals = np.array(vals, dtype=float)

        offset = (j - n_preset / 2 + 0.5) * bar_w
        ax.bar(x + offset, vals, width=bar_w,
               color=PRESET_COLORS[preset],
               label=PRESET_LABELS_SHORT[preset],
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([INSTANCE_LABELS[i] for i in instances])
    ax.set_ylabel("Route Improvement moves executed (millions)")
    ax.set_title("Route Improvement (2-opt/Or-opt) Moves Executed in 600 s")
    ax.legend(loc="upper right", ncol=3)

    savefig(fig, "15_ri_moves.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Scanning results…")
    data = scan_results(RESULTS_DIR)
    print(f"  Found {len(data)} instances:")
    for inst, presets in sorted(data.items()):
        print(f"    {inst}: {list(presets.keys())}")

    print("\nGenerating figures…")
    plot_quality_improvement(data)
    plot_convergence(data)
    plot_hw_utilization(data)
    plot_pairs_evaluated(data)
    plot_ls_throughput(data)
    plot_time_decomposition(data)
    plot_quality_heatmap(data)
    plot_iterations(data)
    plot_swapstar_speedup(data)
    plot_absolute_best(data)
    plot_island_migrations(data)
    plot_offspring_throughput(data)
    plot_radar(data)
    plot_convergence_detail(data)
    plot_ri_moves(data)
    print("\nDone. All plots saved to", OUT_DIR)


if __name__ == "__main__":
    main()
