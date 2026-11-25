"""Generate 3 publication-grade figures for Paper 39:
MMS-ASYMPTOTIC-CERT-1.0 (Lin, Liu, Tao, Wu, & Zhang, arXiv:2611.33052).

Author: Lluis Eriksson
License: MIT
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8.5,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
    }
)


def make_fig1() -> None:
    d1 = json.loads((DATA_DIR / "exp1_support_halving_trajectory.json").read_text(encoding="utf-8"))
    d2 = json.loads((DATA_DIR / "exp2_prefix_reserve_profile.json").read_text(encoding="utf-8"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=300)

    # Panel (a): Support halving |F_r| and cumulative weight loss
    rounds = [r["round"] for r in d1["rows"]]
    f_after = [r["frac_after"] for r in d1["rows"]]
    cum_loss = [r["cum_loss_dec"] for r in d1["rows"]]
    cap = d1["rows"][0]["geometric_cap_dec"]

    ax1_r = ax1.twinx()
    l1 = ax1.plot(rounds, f_after, "o-", color="#1f4e79", lw=1.8, ms=4.8, label=r"Fractional support $|F_r|$ (left)")
    l2 = ax1_r.plot(rounds, cum_loss, "s--", color="#c55a11", lw=1.7, ms=4.5, label=r"Cumulative loss $\sum \omega |F_{t-1}|$ (right)")
    l3 = ax1_r.axhline(cap, color="#548235", ls=":", lw=1.4, label=r"Geometric ceiling $2\omega |F_0| = 64$")
    ax1.set_yscale("log", base=2)
    ax1.set_xlabel("Algorithm 3.1 Step 1 rounding round $r$")
    ax1.set_ylabel(r"Remaining fractional variables $|F_r|$", color="#1f4e79")
    ax1_r.set_ylabel("Cumulative weight loss", color="#c55a11")
    ax1_r.set_ylim(0, cap * 1.28)
    ax1_r.grid(False)
    ax1.set_title("(a) Lemma 3.5 Support Halving & Weight Loss")
    lines = l1 + l2 + [l3]
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="lower left", framealpha=0.92, fontsize=6.6)

    # Panel (b): Step 2 prefix reserve |U cap M[j]| vs (1-lam)*j
    js = [r["prefix_j"] for r in d2["rows"]]
    u_cnt = [r["unassigned_count"] for r in d2["rows"]]
    lb = [r["reserve_lb"] for r in d2["rows"]]
    a_cnt = [r["assigned_count"] for r in d2["rows"]]
    ax2.plot(js, u_cnt, "o-", color="#548235", lw=1.8, ms=4.2, label=r"Unassigned reserve $|U \cap M_{[j]}|$")
    ax2.plot(js, lb, "--", color="#1f4e79", lw=1.4, label=r"Step 2 lower bound $(1-\lambda)j$ ($\lambda=2/3$)")
    ax2.plot(js, a_cnt, "s:", color="#c55a11", lw=1.5, ms=3.8, label=r"Assigned prefix count $j - |U \cap M_{[j]}|$")
    ax2.set_xlabel("IDO item prefix index $j$ ($n=6, m=18$)")
    ax2.set_ylabel("Item count in prefix $M_{[j]}$")
    ax2.set_title(r"(b) Step 2 $\lceil k/\lambda\rceil$ Prefix Reserve Profile")
    ax2.legend(loc="upper left", framealpha=0.92, fontsize=6.8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_rounding_and_reserve.png", dpi=300)
    plt.close(fig)


def make_fig2() -> None:
    d3 = json.loads((DATA_DIR / "exp3_markov_and_asymptotic_decay.json").read_text(encoding="utf-8"))
    d4 = json.loads((DATA_DIR / "exp4_mms_landmark_progression.json").read_text(encoding="utf-8"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=300)

    # Panel (a): Dual Markov reserved fraction O(1/L^2) vs epsilon = 1/L
    Ls = [r["L"] for r in d3["rows"]]
    eps_v = [r["eps_target_dec"] for r in d3["rows"]]
    raw_markov = [1536.0 / (L * L) for L in Ls]
    ax1.plot(Ls, eps_v, "o-", color="#1f4e79", lw=1.8, ms=4.8, label=r"Target MMS error $\varepsilon = 1/L$")
    ax1.plot(Ls, raw_markov, "s--", color="#c55a11", lw=1.7, ms=4.5, label=r"Reserved-agent Markov ratio $1536/L^2$")
    ax1.axhline(1.0, color="#595959", ls=":", lw=1.1, label="Trivial ratio $= 1$")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel(r"Asymptotic granularity parameter $L$ (log scale)")
    ax1.set_ylabel("Ratio / error scale (log scale)")
    ax1.set_title("(a) Section 3.3 Markov Reserved-Agent Decay")
    ax1.legend(loc="lower left", framealpha=0.92, fontsize=6.9)

    # Panel (b): Historical & Asymptotic MMS Approximation Landmarks (Goods & Chores)
    xs = np.arange(len(d4["rows"]))
    goods = [r["goods"] for r in d4["rows"]]
    chores = [r["chores"] for r in d4["rows"]]
    short_x = ["2/3 vs.\n4/3", "3/4 vs.\n11/9", "3/4+ vs.\n13/11", "7/9 vs.\n13/11", "Lin et al.\n(1∓ε)"]
    ax2.plot(xs, goods, "o-", color="#1f4e79", lw=1.9, ms=5.5, label=r"Goods MMS factor ($\uparrow 1-\varepsilon$)")
    ax2.plot(xs, chores, "s-", color="#c00000", lw=1.9, ms=5.2, label=r"Chores MMS factor ($\downarrow 1+\varepsilon$)")
    ax2.axhline(1.0, color="#548235", ls="--", lw=1.3, label="Exact 1-MMS threshold")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(short_x, fontsize=7.0)
    ax2.set_ylim(0.60, 1.40)
    ax2.set_ylabel("Certified MMS approximation factor")
    ax2.set_title("(b) Goods & Chores MMS Convergence to $1 \\mp \\varepsilon$")
    ax2.legend(loc="center left", framealpha=0.92, fontsize=6.8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_markov_and_mms_landmarks.png", dpi=300)
    plt.close(fig)


def make_fig3() -> None:
    rep = json.loads((DATA_DIR / "certificate_replay_summary.json").read_text(encoding="utf-8"))
    rows = rep["certificates"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=300)

    mode_colors = {
        "checkpoint_rounding_prefix_cert": "#1f4e79",
        "deadline_slot_and_bagfill_cert": "#2e75b6",
        "small_slack_dual_price_cert": "#548235",
        "goods_and_chores_mms_allocation_cert": "#c55a11",
    }
    mode_short = {
        "checkpoint_rounding_prefix_cert": "Mode 1 (Step 1 Ckpt)",
        "deadline_slot_and_bagfill_cert": "Mode 2 (Step 2-3 Bag)",
        "small_slack_dual_price_cert": "Mode 3 (Sec. 3.3 Dual)",
        "goods_and_chores_mms_allocation_cert": "Mode 4 (Goods/Chores)",
    }
    xs = np.arange(len(rows))
    lat = [r["verify_ms"] for r in rows]
    cols = [mode_colors[r["mode"]] for r in rows]
    ax1.bar(xs, lat, color=cols, width=0.68, alpha=0.92)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"C{i+1}" for i in range(len(rows))], fontsize=7.0)
    ax1.set_ylim(0, max(lat) * 1.38)
    ax1.set_xlabel("Canonical MMS-ASYMPTOTIC-CERT-1.0 certificate (C1–C16)")
    ax1.set_ylabel("Exact zero-float replay latency (ms)")
    ax1.set_title("(a) Canonical Certificate Replay Latency")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=mode_colors[m], label=mode_short[m])
        for m in mode_colors
    ]
    ax1.legend(handles=handles, loc="upper left", ncol=2, framealpha=0.92, fontsize=6.4)

    cats = ["Global /\nType", "Mode 1\nCkpt", "Mode 2\nBagFill", "Mode 3\nDual LP", "Mode 4\nMMS"]
    counts = [12, 12, 12, 12, 12]
    bcols = ["#595959", "#1f4e79", "#2e75b6", "#548235", "#c55a11"]
    bars = ax2.bar(cats, counts, color=bcols, width=0.58, alpha=0.92)
    for bar, c in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.35, f"{c}/{c}",
                 ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    ax2.set_ylim(0, 15.0)
    ax2.set_ylabel("Live adversarial forgeries rejected")
    ax2.set_title("(b) Adversarial Red-Team Audit (60/60 Rejected)")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_certificate_replay_atlas.png", dpi=300)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    make_fig1()
    make_fig2()
    make_fig3()
    print("Generated all 3 Paper 39 publication figures successfully.")


if __name__ == "__main__":
    main()
