"""Generate all 16 canonical MMS-ASYMPTOTIC-CERT-1.0 certificates and 4 exact-Q
experimental studies for Paper 39: Lin, Liu, Tao, Wu, & Zhang (arXiv:2611.33052).

Author: Lluis Eriksson
License: MIT
"""
from __future__ import annotations

import json
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mms_certifier import SCHEMA_VERSION, verify_mms_certificate_exact
from src.mms_exact import (
    abel_summation_exact,
    dual_prefix_prices_exact,
    format_rat,
    reduced_mms_segments_exact,
    step2_deadline_assign_exact,
    step3_greedy_bag_fill_exact,
)

CERT_DIR = ROOT / "certificates"
DATA_DIR = ROOT / "data"


def _vec_json(v: Sequence[Fraction]) -> List[str]:
    return [format_rat(x) for x in v]


def _mat_json(M: Sequence[Sequence[Fraction]]) -> List[List[str]]:
    return [_vec_json(r) for r in M]


def _make_mode1(
    cert_id: str,
    n: int,
    m: int,
    sigma: Fraction,
    lam: Fraction,
    omega: Fraction,
    vals: List[List[Fraction]],
    cands: List[List[List[int]]],
    wb: List[List[Fraction]],
    wa: List[List[Fraction]],
    ckpts: List[int],
) -> Dict[str, Any]:
    f_before = sum(1 for i in range(n) for x in wb[i] if 0 < x < 1)
    f_after = sum(1 for i in range(n) for x in wa[i] if 0 < x < 1)
    w_loss = sum((x for i in range(n) for x in wb[i]), Fraction(0, 1)) - sum(
        (x for i in range(n) for x in wa[i]), Fraction(0, 1)
    )
    max_ratio = Fraction(0, 1)
    for j in range(1, m + 1):
        load_a = Fraction(0, 1)
        for i in range(n):
            for k, S in enumerate(cands[i]):
                cnt = sum(1 for p in S if p < j)
                if cnt:
                    load_a += wa[i][k] * cnt
        r_j = load_a / Fraction(j, 1)
        if r_j > max_ratio:
            max_ratio = r_j
    return {
        "schema": SCHEMA_VERSION,
        "mode": "checkpoint_rounding_prefix_cert",
        "cert_id": cert_id,
        "num_agents": n,
        "num_items": m,
        "sigma": format_rat(sigma),
        "lam": format_rat(lam),
        "omega": format_rat(omega),
        "valuations": _mat_json(vals),
        "candidate_bundles": cands,
        "weights_before": _mat_json(wb),
        "weights_after": _mat_json(wa),
        "checkpoints": ckpts,
        "claimed_frac_before": f_before,
        "claimed_frac_after": f_after,
        "claimed_weight_loss": format_rat(w_loss),
        "claimed_max_prefix_load_ratio": format_rat(max_ratio),
    }


def _make_mode2(
    cert_id: str,
    n: int,
    m: int,
    sigma: Fraction,
    lam: Fraction,
    vals: List[List[Fraction]],
    served: Dict[int, List[int]],
) -> Dict[str, Any]:
    assigned, unassigned = step2_deadline_assign_exact(served, lam, m)
    unserved = [i for i in range(n) if i not in served]
    min_pool_val = min(abel_summation_exact(vals[i], unassigned)[0] for i in unserved)
    step3_bags, _ = step3_greedy_bag_fill_exact(unassigned, unserved, vals, sigma)
    final_vals = []
    for i in range(n):
        b_i = assigned[i] if i in assigned else step3_bags[i]
        final_vals.append(sum((vals[i][p] for p in b_i), Fraction(0, 1)))
    return {
        "schema": SCHEMA_VERSION,
        "mode": "deadline_slot_and_bagfill_cert",
        "cert_id": cert_id,
        "num_agents": n,
        "num_items": m,
        "sigma": format_rat(sigma),
        "lam": format_rat(lam),
        "valuations": _mat_json(vals),
        "served_integral_bundles": {str(k): v for k, v in served.items()},
        "claimed_step2_assigned": {str(k): v for k, v in assigned.items()},
        "claimed_unassigned_pool": sorted(unassigned),
        "claimed_step3_bags": {str(k): v for k, v in step3_bags.items()},
        "claimed_min_pool_value": format_rat(min_pool_val),
        "claimed_min_final_value": format_rat(min(final_vals)),
    }


def _make_mode3(
    cert_id: str,
    n: int,
    m: int,
    sigma: Fraction,
    tau_ord: Fraction,
    vals: List[List[Fraction]],
    parts: List[List[List[int]]],
    alphas: List[Fraction],
    zetas: List[Fraction] | None = None,
) -> Dict[str, Any]:
    prices = dual_prefix_prices_exact(alphas)
    if zetas is None:
        zetas = []
        for i in range(n):
            z_i = Fraction(1, 1)
            for B in parts[i]:
                pB = sum((prices[pos] for pos in B), Fraction(0, 1))
                if pB < z_i:
                    z_i = pB
                for seg in reduced_mms_segments_exact(B, vals[i], sigma):
                    p_seg = sum((prices[pos] for pos in seg), Fraction(0, 1))
                    if pB - p_seg < z_i:
                        z_i = pB - p_seg
            zetas.append(z_i)
    lam = sum(zetas, Fraction(0, 1)) / Fraction(n, 1)
    delta = Fraction(n, 1) * (Fraction(1, 1) - lam)
    n_ord = [i for i in range(n) if Fraction(n, 1) * (Fraction(1, 1) - zetas[i]) <= tau_ord]
    n_res = [i for i in range(n) if i not in n_ord]
    cap_small = Fraction(1, 1) - sigma
    m2 = [pos for pos in range(m) if any(vals[i][pos] <= cap_small for i in n_ord)]
    m1 = [pos for pos in range(m) if pos not in m2]
    m2_price = sum((prices[pos] for pos in m2), Fraction(0, 1))
    return {
        "schema": SCHEMA_VERSION,
        "mode": "small_slack_dual_price_cert",
        "cert_id": cert_id,
        "num_agents": n,
        "num_items": m,
        "sigma": format_rat(sigma),
        "lam": format_rat(lam),
        "ordinary_threshold": format_rat(tau_ord),
        "valuations": _mat_json(vals),
        "mms_partitions": parts,
        "dual_zeta": _vec_json(zetas),
        "dual_alpha": _vec_json(alphas),
        "claimed_prices": _vec_json(prices),
        "claimed_delta": format_rat(delta),
        "claimed_ordinary_agents": n_ord,
        "claimed_reserved_agents": n_res,
        "claimed_prefix_M1": m1,
        "claimed_M2_price_sum": format_rat(m2_price),
    }


def _make_mode4(
    cert_id: str,
    setting: str,
    n: int,
    m: int,
    eps: Fraction,
    vals: List[List[Fraction]],
    witnesses: List[List[List[int]]],
    alloc: List[List[int]],
    query_specs: List[Tuple[int, List[int], List[int]]],
) -> Dict[str, Any]:
    benches = []
    for i in range(n):
        bvals = [sum((vals[i][p] for p in B), Fraction(0, 1)) for B in witnesses[i]]
        benches.append(min(bvals) if setting == "goods" else max(bvals))
    ag_vals = [sum((vals[i][p] for p in alloc[i]), Fraction(0, 1)) for i in range(n)]
    ratios = [ag_vals[i] / benches[i] for i in range(n)]
    worst = min(ratios) if setting == "goods" else max(ratios)
    queries = []
    for ag, ba, bb in query_specs:
        va = sum((vals[ag][p] for p in ba), Fraction(0, 1))
        vb = sum((vals[ag][p] for p in bb), Fraction(0, 1))
        if setting == "goods":
            pref = "a" if va > vb else ("b" if vb > va else "tie")
        else:
            pref = "a" if va < vb else ("b" if vb < va else "tie")
        queries.append({"agent": ag, "bundle_a": sorted(ba), "bundle_b": sorted(bb), "preferred": pref})
    return {
        "schema": SCHEMA_VERSION,
        "mode": "goods_and_chores_mms_allocation_cert",
        "cert_id": cert_id,
        "setting": setting,
        "num_agents": n,
        "num_items": m,
        "epsilon": format_rat(eps),
        "valuations": _mat_json(vals),
        "mms_witness_partitions": witnesses,
        "allocation": [sorted(b) for b in alloc],
        "comparison_queries": queries,
        "claimed_mms_benchmarks": _vec_json(benches),
        "claimed_agent_values": _vec_json(ag_vals),
        "claimed_worst_ratio": format_rat(worst),
    }


def _uniform_ido_vals(n: int, m: int, profile: List[Fraction]) -> List[List[Fraction]]:
    """Build heterogeneous IDO normalized valuations around a base profile."""
    assert len(profile) == m and sum(profile) == Fraction(n, 1)
    rows: List[List[Fraction]] = []
    for i in range(n):
        row = list(profile)
        if i % 2 == 1 and m >= 4 and row[0] + Fraction(1, 20) <= 1 and row[-1] >= Fraction(1, 20):
            cand = list(row)
            cand[0] += Fraction(1, 20)
            cand[-1] -= Fraction(1, 20)
            if all(cand[j] >= cand[j + 1] for j in range(m - 1)):
                row = cand
        rows.append(row)
    return rows


def build_all_certificates() -> List[Dict[str, Any]]:
    certs: List[Dict[str, Any]] = []

    # =========================================================================
    # Mode 1: 4 canonical Algorithm 3.1 Step 1 Checkpoint Rounding certificates
    # =========================================================================
    # 1a: n=3, m=8, f_before=4 -> f_after=2, 1 checkpoint at j=4
    v1a = [
        [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(2, 5), Fraction(3, 10), Fraction(3, 10), Fraction(1, 4), Fraction(1, 4)],
        [Fraction(11, 20), Fraction(1, 2), Fraction(1, 2), Fraction(7, 20), Fraction(3, 10), Fraction(3, 10), Fraction(1, 4), Fraction(1, 4)],
        [Fraction(1, 2), Fraction(1, 2), Fraction(9, 20), Fraction(9, 20), Fraction(3, 10), Fraction(3, 10), Fraction(1, 4), Fraction(1, 4)],
    ]
    c1a = [
        [[1, 6], [2, 7]],
        [[1, 7], [2, 6]],
        [[5, 6, 7]],
    ]
    wb1a = [[Fraction(1, 2), Fraction(1, 2)], [Fraction(1, 2), Fraction(1, 2)], [Fraction(1)]]
    wa1a = [[Fraction(1, 2), Fraction(0)], [Fraction(1, 2), Fraction(0)], [Fraction(1)]]
    certs.append(_make_mode1("m1_ckpt_3ag_8it_exact_halving", 3, 8, Fraction(3, 4), Fraction(7, 8), Fraction(1, 4), v1a, c1a, wb1a, wa1a, [4]))

    # 1b: n=4, m=10, f_before=6 -> f_after=0 (full rounding to integral!), checkpoint j=5
    v1b_base = [Fraction(3, 5), Fraction(1, 2), Fraction(1, 2), Fraction(2, 5), Fraction(2, 5),
                Fraction(2, 5), Fraction(2, 5), Fraction(3, 10), Fraction(3, 10), Fraction(1, 5)]
    v1b = _uniform_ido_vals(4, 10, v1b_base)
    c1b = [
        [[1, 6], [2, 7]],
        [[2, 8], [3, 6]],
        [[4, 5], [5, 6]],
        [[7, 8, 9]],
    ]
    wb1b = [
        [Fraction(1, 2), Fraction(1, 2)],
        [Fraction(1, 2), Fraction(1, 2)],
        [Fraction(1, 2), Fraction(1, 2)],
        [Fraction(1)],
    ]
    wa1b = [
        [Fraction(0), Fraction(1)],
        [Fraction(0), Fraction(1)],
        [Fraction(0), Fraction(0)],
        [Fraction(1)],
    ]
    certs.append(_make_mode1("m1_ckpt_4ag_10it_full_integral", 4, 10, Fraction(3, 4), Fraction(9, 10), Fraction(1, 4), v1b, c1b, wb1b, wa1b, [5]))

    # 1c: n=5, m=12, f_before=8 -> f_after=4, 2 checkpoints [4, 8]
    v1c_base = [Fraction(3, 5), Fraction(3, 5), Fraction(1, 2), Fraction(1, 2), Fraction(2, 5), Fraction(2, 5),
                Fraction(2, 5), Fraction(2, 5), Fraction(7, 20), Fraction(7, 20), Fraction(1, 4), Fraction(1, 4)]
    v1c = _uniform_ido_vals(5, 12, v1c_base)
    c1c = [
        [[1, 8], [2, 9]],
        [[2, 10], [3, 11]],
        [[4, 7], [5, 8]],
        [[6, 7], [6, 8]],
        [[9, 10, 11]],
    ]
    wb1c = [
        [Fraction(1, 2), Fraction(1, 2)],
        [Fraction(1, 2), Fraction(1, 2)],
        [Fraction(1, 2), Fraction(1, 2)],
        [Fraction(1, 2), Fraction(1, 2)],
        [Fraction(1)],
    ]
    wa1c = [
        [Fraction(1, 2), Fraction(0)],
        [Fraction(1, 2), Fraction(0)],
        [Fraction(1, 2), Fraction(0)],
        [Fraction(1, 2), Fraction(0)],
        [Fraction(1)],
    ]
    certs.append(_make_mode1("m1_ckpt_5ag_12it_two_ckpts", 5, 12, Fraction(7, 10), Fraction(11, 12), Fraction(1, 4), v1c, c1c, wb1c, wa1c, [4, 8]))

    # 1d: n=6, m=14, f_before=8 -> f_after=2, 2 checkpoints [5, 10], small weight loss = 1
    v1d_base = [Fraction(3, 5), Fraction(3, 5), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2),
                Fraction(2, 5), Fraction(2, 5), Fraction(2, 5), Fraction(2, 5), Fraction(3, 10), Fraction(3, 10), Fraction(3, 10), Fraction(3, 10)]
    v1d = _uniform_ido_vals(6, 14, v1d_base)
    c1d = [
        [[1, 10], [2, 11]],
        [[3, 12], [4, 13]],
        [[5, 9], [6, 8]],
        [[6, 9], [7, 8]],
        [[5, 13]],
        [[11, 12, 13]],
    ]
    wb1d = [
        [Fraction(1, 2), Fraction(1, 2)],
        [Fraction(1, 2), Fraction(1, 2)],
        [Fraction(1, 2), Fraction(1, 2)],
        [Fraction(1, 2), Fraction(1, 2)],
        [Fraction(1)],
        [Fraction(1)],
    ]
    wa1d = [
        [Fraction(0), Fraction(1)],
        [Fraction(0), Fraction(1)],
        [Fraction(1, 2), Fraction(1, 2)],
        [Fraction(0), Fraction(0)],
        [Fraction(1)],
        [Fraction(1)],
    ]
    certs.append(_make_mode1("m1_ckpt_6ag_14it_tight_omega", 6, 14, Fraction(3, 4), Fraction(13, 14), Fraction(1, 4), v1d, c1d, wb1d, wa1d, [5, 10]))

    # =========================================================================
    # Mode 2: 4 canonical Algorithm 3.1 Steps 2-3 Deadline & Bag-Filling certs
    # =========================================================================
    # 2a: n=3, m=9, sigma=3/4, lam=2/3 (Delta=1), 2 served Step 2, 1 served Step 3
    v2a = [
        [Fraction(1, 2), Fraction(1, 2), Fraction(2, 5), Fraction(2, 5), Fraction(3, 10), Fraction(3, 10), Fraction(1, 5), Fraction(1, 5), Fraction(1, 5)],
        [Fraction(11, 20), Fraction(1, 2), Fraction(2, 5), Fraction(2, 5), Fraction(3, 10), Fraction(3, 10), Fraction(1, 5), Fraction(1, 5), Fraction(3, 20)],
        [Fraction(1, 2), Fraction(9, 20), Fraction(9, 20), Fraction(2, 5), Fraction(3, 10), Fraction(3, 10), Fraction(1, 5), Fraction(1, 5), Fraction(1, 5)],
    ]
    # Shift Agent 1's last two items strictly right ([2, 6, 8]) so Step 2 strictly reassigns slots leftward (6->5, 8->7)!
    served2a = {0: [1, 4], 1: [2, 6, 8]}
    certs.append(_make_mode2("m2_bagfill_3ag_9it_lam_2_3", 3, 9, Fraction(3, 4), Fraction(2, 3), v2a, served2a))

    # 2b: n=4, m=12, sigma=4/5, lam=3/4 (Delta=1), 3 served Step 2, 1 served Step 3
    v2b_base = [Fraction(1, 2), Fraction(1, 2), Fraction(2, 5), Fraction(2, 5), Fraction(2, 5), Fraction(7, 20),
                Fraction(7, 20), Fraction(3, 10), Fraction(1, 5), Fraction(1, 5), Fraction(1, 5), Fraction(1, 5)]
    v2b = _uniform_ido_vals(4, 12, v2b_base)
    served2b = {0: [1, 3], 1: [2, 5, 10], 2: [6, 7, 9]}
    certs.append(_make_mode2("m2_bagfill_4ag_12it_lam_3_4", 4, 12, Fraction(4, 5), Fraction(3, 4), v2b, served2b))

    # 2c: n=5, m=15, sigma=7/10, lam=3/5 (Delta=2), 3 served Step 2, 2 served Step 3
    v2c_base = [Fraction(11, 20), Fraction(1, 2), Fraction(2, 5), Fraction(2, 5), Fraction(2, 5),
                Fraction(7, 20), Fraction(7, 20), Fraction(7, 20), Fraction(3, 10), Fraction(3, 10),
                Fraction(1, 4), Fraction(1, 4), Fraction(1, 5), Fraction(1, 5), Fraction(1, 5)]
    v2c = _uniform_ido_vals(5, 15, v2c_base)
    served2c = {0: [1, 4], 1: [3, 6], 2: [8, 9, 11]}
    certs.append(_make_mode2("m2_bagfill_5ag_15it_lam_3_5", 5, 15, Fraction(7, 10), Fraction(3, 5), v2c, served2c))

    # 2d: n=6, m=18, sigma=3/4, lam=2/3 (Delta=2), 4 served Step 2, 2 served Step 3
    v2d_base = [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(2, 5), Fraction(2, 5), Fraction(2, 5),
                Fraction(7, 20), Fraction(7, 20), Fraction(7, 20), Fraction(3, 10), Fraction(3, 10), Fraction(3, 10),
                Fraction(1, 4), Fraction(1, 4), Fraction(1, 4), Fraction(1, 5), Fraction(1, 5), Fraction(1, 5)]
    v2d = _uniform_ido_vals(6, 18, v2d_base)
    served2d = {0: [1, 4], 1: [2, 5], 2: [7, 8, 14], 3: [10, 11, 13]}
    certs.append(_make_mode2("m2_bagfill_6ag_18it_lam_2_3", 6, 18, Fraction(3, 4), Fraction(2, 3), v2d, served2d))

    # =========================================================================
    # Mode 3: 4 canonical Section 3.3 Small-Slack Dual LP Price certificates
    # =========================================================================
    # 3a: n=3, m=6, sigma=3/4 (small threshold <= 1/4), heavy prefix [0,1,2] vs small tail [3,4,5]
    v3a = [
        [Fraction(4, 5), Fraction(4, 5), Fraction(4, 5), Fraction(1, 5), Fraction(1, 5), Fraction(1, 5)],
        [Fraction(4, 5), Fraction(4, 5), Fraction(4, 5), Fraction(1, 5), Fraction(1, 5), Fraction(1, 5)],
        [Fraction(3, 4), Fraction(3, 4), Fraction(3, 4), Fraction(1, 4), Fraction(1, 4), Fraction(1, 4)],
    ]
    p3a = [[[0, 3], [1, 4], [2, 5]] for _ in range(3)]
    # alpha: j=2 (pos 2, weight 3) gets 9/10 -> prefix [0,1,2] has price 9/10 + 1/20 = 19/20, tail [3,4,5] has price 1/20
    # Check sum (j+1)*alpha_j = 3*(9/10) + 6*(1/20) = 27/10 + 3/10 = 3 = n!
    a3a = [Fraction(0), Fraction(0), Fraction(9, 10), Fraction(0), Fraction(0), Fraction(1, 20)]
    certs.append(_make_mode3("m3_dual_3ag_6it_heavy_prefix", 3, 6, Fraction(3, 4), Fraction(1, 2), v3a, p3a, a3a))

    # 3b: n=4, m=8, sigma=4/5 (small <= 1/5), 3 ordinary + 1 reserved agent!
    # Agents 0,1,2 pair heavy [0..3] with small [4..7]; Agent 3 has flatter MMS partition [[0,7],[1,6],[2,5],[3,4]]
    # with alpha concentrated on [0,1,2] so Agent 3 has larger (1 - zeta_3) -> classified RESERVED!
    v3b = [
        [Fraction(9, 10), Fraction(9, 10), Fraction(9, 10), Fraction(9, 10), Fraction(1, 10), Fraction(1, 10), Fraction(1, 10), Fraction(1, 10)],
        [Fraction(9, 10), Fraction(9, 10), Fraction(9, 10), Fraction(9, 10), Fraction(1, 10), Fraction(1, 10), Fraction(1, 10), Fraction(1, 10)],
        [Fraction(4, 5), Fraction(4, 5), Fraction(4, 5), Fraction(4, 5), Fraction(1, 5), Fraction(1, 5), Fraction(1, 5), Fraction(1, 5)],
        [Fraction(3, 4), Fraction(3, 4), Fraction(3, 4), Fraction(1, 2), Fraction(1, 2), Fraction(1, 4), Fraction(1, 4), Fraction(1, 4)],
    ]
    p3b = [
        [[0, 4], [1, 5], [2, 6], [3, 7]],
        [[0, 4], [1, 5], [2, 6], [3, 7]],
        [[0, 4], [1, 5], [2, 6], [3, 7]],
        [[0, 7], [1, 6], [2, 5], [3, 4]],
    ]
    # alpha: j=3 gets 9/10 (weight 4 -> 36/10), j=7 gets 1/20 (weight 8 -> 4/10): total = 4!
    # Wait: for agent 3, items 5,6,7 have val 1/4 > 1/5 so NO small items! Make agent 3 have small items [5,6,7] with val 1/5:
    v3b[3] = [Fraction(4, 5), Fraction(4, 5), Fraction(4, 5), Fraction(1, 2), Fraction(1, 2), Fraction(1, 5), Fraction(1, 5), Fraction(1, 5)]
    # And give slightly higher price on j=7 (e.g. alpha_2 = 3/10, alpha_3 = 1/2, alpha_7 = 11/80 -> check sum: 3*3/10 + 4*1/2 + 8*11/80 = 0.9 + 2.0 + 1.1 = 4.0!)
    # Check prices: p[0..2] = 0.3 + 0.5 + 0.1375 = 0.9375, p[3] = 0.6375, p[4..7] = 0.1375.
    # For agents 0,1,2: bundle [3,7] has p = 0.775, minus seg [7] (0.1375) -> zeta = 0.6375!
    # Wait: make agent 3's bundle [3,4] have SMALLER price (or give agent 3 an asymmetric partition):
    # Better: alpha = [0, 0, 0, 9/10, 0, 0, 0, 1/20] -> p[0..3] = 19/20, p[4..7] = 1/20.
    # For agents 0,1,2: every bundle [k, k+4] has p=1, minus small seg [k+4] (1/20) -> zeta_i = 19/20 -> n*(1-zeta_i) = 4*(1/20) = 1/5!
    # For agent 3: bundle [3,4] has NO small items (v=1/2, 1/2), p([3,4]) = 19/20 + 1/20 = 1;
    # wait: give agent 3 bundle [4,5,6,7]? No, v_3 = [1, 4/5, 4/5, 4/5, 1/5, 1/5, 1/5, 0] -> sum = 4!
    # Partition for agent 3: [[0], [1, 4], [2, 5], [3, 6, 7]]:
    #   - [0]: p = 19/20, no small items -> p = 19/20
    #   - [3, 6, 7]: p = 19/20 + 1/20 + 1/20 = 21/20, small items {6,7} have v=1/5+0=1/5 <= 1/5 (1 seg, p=2/20) -> p - p_seg = 19/20!
    # To make Agent 3 strictly RESERVED (smaller zeta_3): put small price bump at j=7 AND give Agent 3 a bundle of ALL small items!
    # Look how clean: v3b[3] = [1, 1, 1, 1/5, 1/5, 1/5, 1/5, 1/5] (sum = 4, IDO, <= 1)!!
    # Partition P_3 = [[0], [1], [2], [3, 4, 5, 6, 7]] (all 4 bundles have value 1)!
    # With alpha = [0, 0, 0, 9/10, 0, 0, 0, 1/20] -> p[0..3]=19/20, p[4..7]=1/20:
    #   - For agents 0,1,2: P_i = [[0,4],[1,5],[2,6],[3,7]] -> zeta_i = 19/20 -> n(1-zeta_i) = 1/5!
    #   - For agent 3: bundle [3,4,5,6,7] has p = 19/20 + 4/20 = 23/20, and small segment [3] has p = 19/20!!
    #     So p(B) - p([3]) = 4/20 = 1/5 -> zeta_3 = 1/5 -> n(1-zeta_3) = 16/5 = 3.2!!
    #   - With ordinary_threshold tau_ord = 1/2:
    #     n_ord = [0, 1, 2] (1/5 <= 1/2), n_res = [3] (16/5 > 1/2)!
    #     Check Markov: |N_r| * tau_ord = 1 * 1/2 = 1/2 <= n * Delta = 4 * (3/20 + 4/5) = 4 * 19/20 = 19/5!!
    #     AND M1 = [0, 1, 2, 3] (all i in N_o have v_i(e) >= 4/5 > 1/5)!!
    v3b[3] = [Fraction(1), Fraction(1), Fraction(1), Fraction(1, 5), Fraction(1, 5), Fraction(1, 5), Fraction(1, 5), Fraction(1, 5)]
    p3b[3] = [[0], [1], [2], [3, 4, 5, 6, 7]]
    a3b = [Fraction(0), Fraction(0), Fraction(0), Fraction(9, 10), Fraction(0), Fraction(0), Fraction(0), Fraction(1, 20)]
    certs.append(_make_mode3("m3_dual_4ag_8it_one_reserved", 4, 8, Fraction(4, 5), Fraction(1, 2), v3b, p3b, a3b))

    # 3c: n=5, m=10, sigma=4/5, two-tier alpha spectrum (alpha_4 > 0, alpha_7 > 0, alpha_9 > 0)
    v3c = [
        [Fraction(9, 10)] * 5 + [Fraction(1, 10)] * 5,
        [Fraction(9, 10)] * 5 + [Fraction(1, 10)] * 5,
        [Fraction(4, 5)] * 5 + [Fraction(1, 5)] * 5,
        [Fraction(4, 5)] * 5 + [Fraction(1, 5)] * 5,
        [Fraction(4, 5)] * 5 + [Fraction(1, 5)] * 5,
    ]
    p3c = [[[k, k + 5] for k in range(5)] for _ in range(5)]
    # 5*alpha_4 + 8*alpha_7 + 10*alpha_9 = 5: choose alpha_4 = 4/5 (4.0), alpha_7 = 1/16 (0.5), alpha_9 = 1/20 (0.5) -> total = 5.0!
    a3c = [Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(4, 5),
           Fraction(0), Fraction(0), Fraction(1, 16), Fraction(0), Fraction(1, 20)]
    certs.append(_make_mode3("m3_dual_5ag_10it_two_tier_alpha", 5, 10, Fraction(4, 5), Fraction(1), v3c, p3c, a3c))

    # 3d: n=6, m=12, sigma=5/6 (small <= 1/6), 5 ordinary + 1 reserved, tiny M2 price sum
    v3d = [
        [Fraction(11, 12)] * 6 + [Fraction(1, 12)] * 6 for _ in range(5)
    ] + [
        [Fraction(1)] * 5 + [Fraction(1, 7)] * 7  # Wait: sum = 5 + 1 = 6, IDO, <= 1!
    ]
    p3d = [[[k, k + 6] for k in range(6)] for _ in range(5)] + [
        [[0], [1], [2], [3], [4], [5, 6, 7, 8, 9, 10, 11]]
    ]
    # 6*alpha_5 + 12*alpha_11 = 6: choose alpha_5 = 23/24 (23/4), alpha_11 = 1/48 (1/4) -> total = 6!
    a3d = [Fraction(0)] * 5 + [Fraction(23, 24)] + [Fraction(0)] * 5 + [Fraction(1, 48)]
    certs.append(_make_mode3("m3_dual_6ag_12it_tight_small_slack", 6, 12, Fraction(5, 6), Fraction(1, 2), v3d, p3d, a3d))

    # =========================================================================
    # Mode 4: 4 canonical Goods (1-eps)-MMS & Chores (1+eps)-MMS Allocation certs
    # =========================================================================
    # 4a: Goods, n=3, m=9, eps=1/20 (95%-MMS), exact balanced witnesses (mu_i = 20)
    vg4a = [
        [Fraction(10), Fraction(9), Fraction(8), Fraction(7), Fraction(6), Fraction(6), Fraction(5), Fraction(5), Fraction(4)],  # sum=60, avg=20: [0,3,8]? 10+7+4=21! Let's make exact 20-splits:
    ]
    # Exact 3-way 20-splits:
    # Agent 0: [10, 8, 7, 6, 6, 6, 6, 6, 5] -> sum=60; W0 = [[0,3,8]? 10+6+4] Wait: [10, 9, 8, 7, 6, 6, 5, 5, 4]:
    #   [0, 5, 8] = 10+6+4=20; [1, 4, 6] = 9+6+5=20; [2, 3, 7] = 8+7+5=20! EXACT!
    # Agent 1: [9, 9, 8, 7, 7, 6, 5, 5, 4] -> sum=60:
    #   [0, 5, 6] = 9+6+5=20; [1, 3, 8] = 9+7+4=20; [2, 4, 7] = 8+7+5=20! EXACT!
    # Agent 2: [11, 8, 7, 7, 6, 6, 5, 5, 5] -> sum=60:
    #   [0, 7, 8]? 11+5+5=21 -> use [10, 8, 8, 7, 6, 6, 5, 5, 5] (sum=60):
    #   [0, 6, 7]=10+5+5=20; [1, 3, 8]=8+7+5=20; [2, 4, 5]=8+6+6=20! EXACT!
    vg4a = [
        [Fraction(10), Fraction(9), Fraction(8), Fraction(7), Fraction(6), Fraction(6), Fraction(5), Fraction(5), Fraction(4)],
        [Fraction(9), Fraction(9), Fraction(8), Fraction(7), Fraction(7), Fraction(6), Fraction(5), Fraction(5), Fraction(4)],
        [Fraction(10), Fraction(8), Fraction(8), Fraction(7), Fraction(6), Fraction(6), Fraction(5), Fraction(5), Fraction(5)],
    ]
    wg4a = [
        [[0, 5, 8], [1, 4, 6], [2, 3, 7]],
        [[0, 5, 6], [1, 3, 8], [2, 4, 7]],
        [[0, 6, 7], [1, 3, 8], [2, 4, 5]],
    ]
    # Allocation attaining >= 19/20 * 20 = 19 for all 3 (specifically 19, 20, 21 -> worst ratio = 19/20 = 0.95 EXACT!):
    # Ag0: [0, 6, 8] = 10+5+4 = 19 (19/20); Ag1: [1, 3, 7] = 9+7+5 = 21 (21/20); Ag2: [2, 4, 5] = 8+6+6 = 20 (1)!
    alloc4a = [[0, 6, 8], [1, 3, 7], [2, 4, 5]]
    q4a = [
        (0, [0, 6, 8], [0, 5, 8]),
        (0, [1, 4, 6], [2, 3, 7]),
        (1, [1, 3, 7], [0, 5, 6]),
        (1, [2, 4, 7], [1, 3, 8]),
        (2, [2, 4, 5], [0, 6, 7]),
        (2, [0, 1], [2, 3, 4]),
    ]
    certs.append(_make_mode4("m4_goods_3ag_9it_95pct_mms", "goods", 3, 9, Fraction(1, 20), vg4a, wg4a, alloc4a, q4a))

    # 4b: Goods, n=4, m=12, eps=1/10, heterogeneous valuations (non-IDO allowed in Mode 4!) achieving >= 1-MMS!
    # Each agent has sum = 40, avg = 10, exact 10-split witness, and complementary valuations yield >= 10 each!
    vg4b = [
        [Fraction(6), Fraction(5), Fraction(4), Fraction(4), Fraction(4), Fraction(3), Fraction(3), Fraction(3), Fraction(2), Fraction(2), Fraction(2), Fraction(2)],
        [Fraction(2), Fraction(2), Fraction(6), Fraction(5), Fraction(4), Fraction(4), Fraction(4), Fraction(3), Fraction(3), Fraction(3), Fraction(2), Fraction(2)],
        [Fraction(2), Fraction(2), Fraction(2), Fraction(2), Fraction(6), Fraction(5), Fraction(4), Fraction(4), Fraction(4), Fraction(3), Fraction(3), Fraction(3)],
        [Fraction(3), Fraction(3), Fraction(2), Fraction(2), Fraction(2), Fraction(2), Fraction(6), Fraction(5), Fraction(4), Fraction(4), Fraction(4), Fraction(3)],
    ]
    # Check each row sums to 40: 6+5+4*3+3*3+2*4 = 11+12+9+8 = 40!
    # Exact 10-split for base [6,5,4,4,4,3,3,3,2,2,2,2]:
    #   [0,8,9] = 6+2+2 = 10
    #   [1,5,10] = 5+3+2 = 10
    #   [2,3,11] = 4+4+2 = 10
    #   [4,6,7] = 4+3+3 = 10
    base_w = [[0, 8, 9], [1, 5, 10], [2, 3, 11], [4, 6, 7]]
    wg4b = []
    for shift in [0, 2, 4, 6]:
        # Wait: row 3 is shifted by 6? Let's verify shifts:
        # row 0: [6,5,4,4,4,3,3,3,2,2,2,2] (shift 0)
        # row 1: shift right by 2 -> [2,2,6,5,4,4,4,3,3,3,2,2]
        # row 2: shift right by 4 -> [2,2,2,2,6,5,4,4,4,3,3,3]
        # row 3: shift right by 6 -> [3,3,2,2,2,2,6,5,4,4,4,3] (EXACT!)
        wg4b.append([[(p + shift) % 12 for p in B] for B in base_w])
    # Allocation giving each agent their top items:
    # Ag0: [0, 1] (11), Ag1: [2, 3] (11), Ag2: [4, 5, 11] (6+5+3=14), Ag3: [6, 7, 8, 9, 10] (6+5+4+4+4=23)!
    alloc4b = [[0, 1, 10], [2, 3, 11], [4, 5, 9], [6, 7, 8]]
    # Check values: Ag0: 6+5+2=13; Ag1: 6+5+2=13; Ag2: 6+5+3=14; Ag3: 6+5+4=15 -> all >= 13/10 >= 9/10!
    q4b = [
        (0, [0, 1, 10], [0, 8, 9]),
        (0, [2, 3, 11], [4, 6, 7]),
        (1, [2, 3, 11], [0, 1, 10]),
        (1, [4, 5, 9], [6, 7, 8]),
        (2, [4, 5, 9], [0, 1, 10]),
        (2, [6, 7, 8], [2, 3, 11]),
        (3, [6, 7, 8], [0, 1, 10]),
        (3, [4, 5, 9], [2, 3, 11]),
    ]
    certs.append(_make_mode4("m4_goods_4ag_12it_exact_1_mms", "goods", 4, 12, Fraction(1, 10), vg4b, wg4b, alloc4b, q4b))

    # 4c: Chores, n=3, m=9, eps=1/20 (105%-MMS chores!), balanced exact MMS witness (mu_i = 20)
    # Reuse exact balanced cost matrices from 4a:
    # Allocation gives costs: Ag0: [0, 5, 7]=10+6+5=21 (21/20 = 1.05 EXACT!); Ag1: [1, 4, 8]=9+7+4=20 (1); Ag2: [2, 3, 6]=8+7+5=20 (1)?
    # Check disjointness: [0,5,7], [1,4,8], [2,3,6] -> covers 0..8!
    # Check costs: Ag0: 10+6+5=21 <= 21; Ag1: 9+7+4=20 <= 21; Ag2: 8+7+5=20 <= 21! Worst ratio = 21/20 EXACT!
    alloc4c = [[0, 5, 7], [1, 4, 8], [2, 3, 6]]
    q4c = [
        (0, [0, 5, 8], [0, 5, 7]),  # chores: lower cost 20 preferred over 21 -> "a"
        (0, [1, 4, 6], [2, 3, 7]),  # tie 20 vs 20
        (1, [1, 4, 8], [0, 1, 2]),
        (1, [2, 4, 7], [0, 5, 6]),
        (2, [2, 3, 6], [0, 1, 2]),
        (2, [6, 7, 8], [0, 1, 2]),
    ]
    certs.append(_make_mode4("m4_chores_3ag_9it_105pct_mms", "chores", 3, 9, Fraction(1, 20), vg4a, wg4a, alloc4c, q4c))

    # 4d: Chores, n=4, m=10, eps=1/10 (110%-MMS chores!), single-dominant-chore exact witness (mu_i = max_item = 10 > avg_i = 8)!
    # Each agent has one dominant chore of cost 10 and 9 smaller chores summing to 22 (total=32, avg=8 < 10)!
    # Witness partition puts the dominant chore alone (cost 10) and splits the remaining 22 into 3 bundles <= 10 (e.g., 8, 7, 7)!
    # Hence max_k c_i(W_k) = 10 = max_item_i -> EXACT MMS_i = 10!
    vc4d = [
        [Fraction(10), Fraction(4), Fraction(3), Fraction(3), Fraction(2), Fraction(2), Fraction(2), Fraction(2), Fraction(2), Fraction(2)],
        [Fraction(4), Fraction(10), Fraction(3), Fraction(3), Fraction(2), Fraction(2), Fraction(2), Fraction(2), Fraction(2), Fraction(2)],
        [Fraction(3), Fraction(4), Fraction(10), Fraction(3), Fraction(2), Fraction(2), Fraction(2), Fraction(2), Fraction(2), Fraction(2)],
        [Fraction(3), Fraction(3), Fraction(4), Fraction(10), Fraction(2), Fraction(2), Fraction(2), Fraction(2), Fraction(2), Fraction(2)],
    ]
    wc4d = [
        [[0], [1, 2, 4], [3, 5, 6], [7, 8, 9]],  # costs: 10, 4+3+2=9, 3+2+2=7, 2+2+2=6 -> max=10!
        [[1], [0, 2, 4], [3, 5, 6], [7, 8, 9]],  # costs: 10, 4+3+2=9, 3+2+2=7, 6 -> max=10!
        [[2], [1, 0, 4], [3, 5, 6], [7, 8, 9]],  # costs: 10, 4+3+2=9, 7, 6 -> max=10!
        [[3], [2, 0, 4], [1, 5, 6], [7, 8, 9]],  # costs: 10, 4+3+2=9, 7, 6 -> max=10!
    ]
    # Allocation: each agent avoids their dominant chore, getting cost <= 11 (ratio <= 11/10)!
    # Ag0: [1, 2, 4] (4+3+2=9); Ag1: [0, 3, 5] (4+3+2=9); Ag2: [6, 7, 8, 9] (8); Ag3: [2]? wait need disjoint partition of 0..9:
    # Ag0: [1, 4, 5] (4+2+2=8); Ag1: [2, 6, 7] (3+2+2=7); Ag2: [3, 8, 9] (3+2+2=7); Ag3: [0] (3)!
    # Even better: give Ag0 [1, 2, 4, 5] = 4+3+2+2 = 11 (exact 11/10 boundary!), Ag1 [0, 3] = 7, Ag2 [6, 7, 8] = 6, Ag3 [9] = 2 -> wait sum=10 items!
    alloc4d = [[1, 2, 4, 5], [0, 3, 6], [7, 8], [9]]
    # Check costs:
    # Ag0: 4+3+2+2 = 11 (11/10 = 1+eps EXACT!)
    # Ag1: 4+3+2 = 9 (9/10)
    # Ag2: 2+2 = 4 (2/5)
    # Ag3: 2 (1/5)
    q4d = [
        (0, [1, 2, 4, 5], [0]),        # cost 11 vs 10 -> "b"
        (0, [7, 8, 9], [1, 2]),        # cost 6 vs 7 -> "a"
        (1, [0, 3, 6], [1]),           # cost 9 vs 10 -> "a"
        (1, [2, 3], [7, 8, 9]),        # cost 6 vs 6 -> "tie"
        (2, [7, 8], [2]),              # cost 4 vs 10 -> "a"
        (2, [0, 1], [3, 4, 5]),        # cost 7 vs 7 -> "tie"
        (3, [9], [3]),                 # cost 2 vs 10 -> "a"
        (3, [0, 1, 2], [3]),           # cost 10 vs 10 -> "tie"
    ]
    certs.append(_make_mode4("m4_chores_4ag_10it_dominant_chore", "chores", 4, 10, Fraction(1, 10), vc4d, wc4d, alloc4d, q4d))

    return certs


def run_experiments() -> None:
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    certs = build_all_certificates()
    replay_rows: List[Dict[str, Any]] = []
    for cert in certs:
        out_path = CERT_DIR / f"{cert['cert_id']}.json"
        out_path.write_text(json.dumps(cert, indent=2) + "\n", encoding="utf-8")
        verify_mms_certificate_exact(cert)
        times_ms: List[float] = []
        res: Dict[str, Any] = {}
        for _ in range(5):
            t0 = time.perf_counter()
            res = verify_mms_certificate_exact(cert)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        times_ms.sort()
        replay_rows.append(
            {
                "cert_id": cert["cert_id"],
                "mode": cert["mode"],
                "verify_ms": round(times_ms[2], 3),
                "summary": res,
            }
        )

    (DATA_DIR / "certificate_replay_summary.json").write_text(
        json.dumps({"count": len(replay_rows), "certificates": replay_rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    # =========================================================================
    # Exp 1: Algorithm 3.1 Step 1 Support Halving & Cumulative Weight Loss Trajectory
    # =========================================================================
    exp1_rows = []
    f_cur = 1024
    cum_loss = Fraction(0, 1)
    omega_ex = Fraction(1, 32)  # L = 32 -> 1/(2L) = 1/64, use omega = 1/32
    for step in range(1, 9):
        ckpts_cap = f_cur // 4
        step_loss = omega_ex * Fraction(f_cur, 1)
        cum_loss += step_loss
        f_next = f_cur // 2
        exp1_rows.append(
            {
                "round": step,
                "frac_before": f_cur,
                "frac_after": f_next,
                "max_checkpoints": ckpts_cap,
                "step_loss_dec": float(step_loss),
                "cum_loss_dec": float(cum_loss),
                "geometric_cap_dec": float(2 * omega_ex * 1024),
            }
        )
        f_cur = f_next
    (DATA_DIR / "exp1_support_halving_trajectory.json").write_text(
        json.dumps({"omega": format_rat(omega_ex), "f0": 1024, "rows": exp1_rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    # =========================================================================
    # Exp 2: Step 2 Unassigned Prefix Reserve & Eq. (3.1) Pool Value vs Slack Delta
    # =========================================================================
    # Exact n=6, m=18 instance from 2d across prefix j=1..18
    c2d = next(c for c in certs if c["cert_id"] == "m2_bagfill_6ag_18it_lam_2_3")
    unassigned_2d = set(c2d["claimed_unassigned_pool"])
    lam_2d = Fraction(2, 3)
    exp2_rows = []
    for j in range(1, 19):
        u_pref = sum(1 for p in unassigned_2d if p < j)
        lb_pref = float((Fraction(1) - lam_2d) * j)
        exp2_rows.append(
            {
                "prefix_j": j,
                "unassigned_count": u_pref,
                "reserve_lb": round(lb_pref, 4),
                "assigned_count": j - u_pref,
                "cap_lam_j": round(float(lam_2d * j), 4),
            }
        )
    (DATA_DIR / "exp2_prefix_reserve_profile.json").write_text(
        json.dumps({"n": 6, "m": 18, "lam": "2/3", "rows": exp2_rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    # =========================================================================
    # Exp 3: Section 3.3 Dual Price Spectrum & Reserved-Agent Markov Decay O(1/L^2)
    # =========================================================================
    exp3_rows = []
    for L in [2, 4, 8, 16, 32, 64, 128]:
        # Theorem 11 exact bound: |N_r|/n < 1536 / L^2, and asymptotic loss 1/L
        exp3_rows.append(
            {
                "L": L,
                "eps_target_dec": round(1.0 / L, 5),
                "R_slack_cap": 768 * L * L,
                "markov_reserved_frac_bound": round(min(1.0, 1536.0 / (L * L)), 6),
                "terminal_loss_160L": 160 * L,
            }
        )
    (DATA_DIR / "exp3_markov_and_asymptotic_decay.json").write_text(
        json.dumps({"rows": exp3_rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    # =========================================================================
    # Exp 4: Historical & Asymptotic MMS Approximation Landmarks (Goods & Chores)
    # =========================================================================
    exp4_rows = [
        {"stage": 1, "goods_label": "Kurokawa et al. (2014)", "goods": 2 / 3, "chores_label": "Aziz et al. (2017)", "chores": 4 / 3},
        {"stage": 2, "goods_label": "Ghodsi et al. (2018)", "goods": 3 / 4, "chores_label": "Huang-Lu (2021)", "chores": 11 / 9},
        {"stage": 3, "goods_label": "Akrami-Garg (2024)", "goods": 0.750782, "chores_label": "Huang-Segal-Halevi (2023)", "chores": 13 / 11},
        {"stage": 4, "goods_label": "Huang-Zhou (2026)", "goods": 7 / 9, "chores_label": "Huang-Segal-Halevi (2023)", "chores": 13 / 11},
        {"stage": 5, "goods_label": "Lin et al. (2026, 1-eps)", "goods": 0.99, "chores_label": "Lin et al. (2026, 1+eps)", "chores": 1.01},
    ]
    (DATA_DIR / "exp4_mms_landmark_progression.json").write_text(
        json.dumps({"rows": exp4_rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Generated {len(certs)} canonical MMS-ASYMPTOTIC-CERT-1.0 certificates and 4 experiment JSONs.")
    for r in replay_rows:
        print(f"  {r['cert_id']:38s}  {r['verify_ms']:6.3f} ms  ({r['mode']})")


if __name__ == "__main__":
    run_experiments()
