# Machine-Checked Exact-Rational Checkpoint Rounding, Dual-Price Markov, and Goods/Chores $(1 \mp \varepsilon)$-MMS Allocation Certificates

**Author:** Lluis Eriksson (prepared with transparent AI assistance via Google Antigravity)
**Target Reference:** Zehan Lin, Shengxin Liu, Biaoshuai Tao, Xiaowei Wu, & Yuhao Zhang, *$(1 - \varepsilon)$-MMS Allocations*, `arXiv:2611.33052` (November 24, 2026).
**Licenses:** Manuscript (`paper/main.tex`, `paper/main.pdf`) under `CC-BY-4.0`; code, Lean 4 proofs, and exact certificates under `MIT`.

## Highlights
- **Zero-Dependency Lean 4 Kernel (`lean/AsymptoticMMS.lean`, Lean 4.35.0):** 18 constructive theorems, `0` `sorry`, `0` warnings, `0` `Classical.choice`, `0` `Lean.ofReduceBool` (2 zero-axiom `none`, 3 `[propext]`, 13 `[propext, Quot.sound]`).
- **Pure-Stdlib Zero-Float Verifier (`MMS-ASYMPTOTIC-CERT-1.0`, `src/mms_certifier.py`):** Four exact-$\mathbb{Q}$ modes auditing Algorithm 3.1 Step 1 checkpoint rounding & Lemma 3.5 support halving, Steps 2–3 $\lceil k/\lambda\rceil$ deadline assignment with Eq. (3.1) Abel summation & greedy bag-filling, Section 3.3 dual prices & Definition 3.7 Markov reserved-agent bounds, and tight exact-MMS goods/chores allocations with ordinal comparison queries.
- **Canonical Atlas & Adversarial Audit:** 16/16 canonical certificates replayed in `0.211–0.864 ms` and **60/60 live adversarial forgery attacks rejected**.

## Quick Reproduction
```bash
# 1. Build & audit Lean 4 kernel
cd lean && lake build && lake env lean AuditAxioms.lean && cd ..

# 2. Run deterministic end-to-end verification (Lean + 16 certs + 60/60 attacks + SHA-256 manifest)
python scripts/verify_all.py

# Optional: regenerate certificates, wall-clock benchmark JSONs, and figures
python scripts/run_all_experiments.py
python scripts/make_figures.py
```
