"""Zero-floating-point exact-Q certificate verifier: MMS-ASYMPTOTIC-CERT-1.0.
Audits all four algorithmic and dual pillars of
Zehan Lin, Shengxin Liu, Biaoshuai Tao, Xiaowei Wu, & Yuhao Zhang
(arXiv:2611.33052, Nov 24, 2026) "(1 - eps)-MMS Allocations":
  - Mode 1: checkpoint_rounding_prefix_cert (Algorithm 3.1 Step 1, Lemmas 3.4 & 3.5)
  - Mode 2: deadline_slot_and_bagfill_cert (Algorithm 3.1 Steps 2-3, Eq. 3.1 Abel summation & Lemma 3.6)
  - Mode 3: small_slack_dual_price_cert (Section 3.3 dual prices, Def. 3.7, Lemma 3.8 & Eq. 3.2)
  - Mode 4: goods_and_chores_mms_allocation_cert ((1-eps)-MMS goods & (1+eps)-MMS chores + ordinal comparisons)

Author: Lluis Eriksson
License: MIT
"""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Any, Dict, List, Set

from src.mms_exact import (
    abel_summation_exact,
    dual_prefix_prices_exact,
    floor_mul_rat,
    format_rat,
    reduced_mms_segments_exact,
    step2_deadline_assign_exact,
    step3_greedy_bag_fill_exact,
)

SCHEMA_VERSION = "MMS-ASYMPTOTIC-CERT-1.0"
_CANON_INT_RE = re.compile(r"^(0|-?[1-9][0-9]*)$")
_CANON_FRAC_RE = re.compile(r"^-?[1-9][0-9]*/[1-9][0-9]*$")
MAX_RAT_STR_LEN = 1024
MAX_INT_BITS = 2048


class MMSCertificateError(ValueError):
    """Raised whenever an MMS-ASYMPTOTIC-CERT-1.0 certificate fails any invariant."""


def _assert_strict_json_types(obj: Any, path: str = "$") -> None:
    t = type(obj)
    if t is int:
        if obj.bit_length() > MAX_INT_BITS:
            raise MMSCertificateError(f"Integer bit-length > {MAX_INT_BITS} at {path}")
        return
    if t is str or t is bool:
        return
    if t is list:
        for i, v in enumerate(obj):
            _assert_strict_json_types(v, f"{path}[{i}]")
        return
    if t is dict:
        for k, v in obj.items():
            if type(k) is not str:
                raise MMSCertificateError(f"Non-string dict key at {path}")
            _assert_strict_json_types(v, f"{path}.{k}")
        return
    raise MMSCertificateError(f"Forbidden JSON type {t.__name__} at {path}")


def _parse_int(val: Any, name: str) -> int:
    if type(val) is not int:
        raise MMSCertificateError(f"{name} must be strict int, got {type(val).__name__}")
    if val.bit_length() > MAX_INT_BITS:
        raise MMSCertificateError(f"{name} integer too large")
    return val


def _parse_rat(val: Any, name: str) -> Fraction:
    if type(val) is int:
        if val.bit_length() > MAX_INT_BITS:
            raise MMSCertificateError(f"{name} integer too large")
        return Fraction(val, 1)
    if type(val) is str:
        if len(val) > MAX_RAT_STR_LEN:
            raise MMSCertificateError(f"{name} rational string too long")
        if _CANON_INT_RE.match(val):
            iv = int(val)
            if iv.bit_length() > MAX_INT_BITS:
                raise MMSCertificateError(f"{name} integer bit-length > {MAX_INT_BITS}")
            return Fraction(iv, 1)
        if _CANON_FRAC_RE.match(val):
            num_s, den_s = val.split("/")
            num, den = int(num_s), int(den_s)
            if num.bit_length() > MAX_INT_BITS or den.bit_length() > MAX_INT_BITS:
                raise MMSCertificateError(f"{name} rational bit-length > {MAX_INT_BITS}")
            if den < 2:
                raise MMSCertificateError(f"{name} denominator must be >= 2: {val!r}")
            q = Fraction(num, den)
            if q.numerator != num or q.denominator != den:
                raise MMSCertificateError(f"{name} unreduced fraction: {val!r}")
            return q
        raise MMSCertificateError(f"{name} non-canonical rational string: {val!r}")
    raise MMSCertificateError(f"{name} invalid type {type(val).__name__}")


def _parse_exact_keys(cert: Dict[str, Any], req: Set[str], mode: str) -> None:
    keys = set(cert.keys())
    if keys != req:
        raise MMSCertificateError(f"{mode} key mismatch: {sorted(keys ^ req)}")


def _parse_valuations(raw: Any, n: int, m: int, require_ido: bool = True, require_norm: bool = True) -> List[List[Fraction]]:
    if type(raw) is not list or len(raw) != n:
        raise MMSCertificateError(f"valuations must have length {n}")
    vals: List[List[Fraction]] = []
    for i in range(n):
        row_raw = raw[i]
        if type(row_raw) is not list or len(row_raw) != m:
            raise MMSCertificateError(f"valuations[{i}] must have length {m}")
        row = [_parse_rat(row_raw[j], f"valuations[{i}][{j}]") for j in range(m)]
        for j, v in enumerate(row):
            if v < 0:
                raise MMSCertificateError(f"Negative valuation at ({i},{j})")
            if require_norm and v > Fraction(1, 1):
                raise MMSCertificateError(f"Normalized item value > 1 at ({i},{j})")
            if require_ido and j > 0 and row[j] > row[j - 1]:
                raise MMSCertificateError(f"IDO ordering violated for agent {i} at position {j}")
        if require_norm and sum(row, Fraction(0, 1)) != Fraction(n, 1):
            raise MMSCertificateError(f"Normalized valuation sum != n={n} for agent {i}")
        vals.append(row)
    return vals


def _parse_bundle(raw: Any, m: int, name: str) -> List[int]:
    if type(raw) is not list:
        raise MMSCertificateError(f"{name} must be list")
    items = [_parse_int(v, name) for v in raw]
    if len(set(items)) != len(items) or any(not (0 <= x < m) for x in items):
        raise MMSCertificateError(f"Invalid item indices in {name}")
    return sorted(items)


# =============================================================================
# Mode 1: Algorithm 3.1 Step 1 — Checkpoint Rounding & Support Halving (Lemmas 3.4-3.5)
# =============================================================================


def _verify_mode1_checkpoint_rounding(cert: Dict[str, Any]) -> Dict[str, Any]:
    req = {
        "schema",
        "mode",
        "cert_id",
        "num_agents",
        "num_items",
        "sigma",
        "lam",
        "omega",
        "valuations",
        "candidate_bundles",
        "weights_before",
        "weights_after",
        "checkpoints",
        "claimed_frac_before",
        "claimed_frac_after",
        "claimed_weight_loss",
        "claimed_max_prefix_load_ratio",
    }
    _parse_exact_keys(cert, req, "Mode 1")
    n = _parse_int(cert["num_agents"], "num_agents")
    m = _parse_int(cert["num_items"], "num_items")
    if not (2 <= n <= 16 and 2 <= m <= 64):
        raise MMSCertificateError("Require 2 <= n <= 16 and 2 <= m <= 64")
    sigma = _parse_rat(cert["sigma"], "sigma")
    lam = _parse_rat(cert["lam"], "lam")
    omega = _parse_rat(cert["omega"], "omega")
    if not (Fraction(0, 1) < sigma < Fraction(1, 1)):
        raise MMSCertificateError("Require 0 < sigma < 1")
    if not (Fraction(0, 1) < lam <= Fraction(1, 1)):
        raise MMSCertificateError("Require 0 < lam <= 1")
    if not (Fraction(0, 1) < omega < Fraction(1, 2)):
        raise MMSCertificateError("Require 0 < omega < 1/2")

    vals = _parse_valuations(cert["valuations"], n, m, require_ido=True, require_norm=True)

    raw_b = cert["candidate_bundles"]
    raw_wb = cert["weights_before"]
    raw_wa = cert["weights_after"]
    if type(raw_b) is not list or len(raw_b) != n or type(raw_wb) is not list or len(raw_wb) != n or type(raw_wa) is not list or len(raw_wa) != n:
        raise MMSCertificateError("candidate_bundles / weights lengths must equal n")

    bundles: List[List[List[int]]] = []
    wb: List[List[Fraction]] = []
    wa: List[List[Fraction]] = []
    for i in range(n):
        if type(raw_b[i]) is not list or not (1 <= len(raw_b[i]) <= 256):
            raise MMSCertificateError(f"candidate_bundles[{i}] must have length in [1, 256]")
        b_i = [_parse_bundle(raw_b[i][k], m, f"B[{i}][{k}]") for k in range(len(raw_b[i]))]
        for k, S in enumerate(b_i):
            if not S or sum((vals[i][p] for p in S), Fraction(0, 1)) < sigma:
                raise MMSCertificateError(f"Candidate bundle B[{i}][{k}] below sigma={sigma}")
        if type(raw_wb[i]) is not list or len(raw_wb[i]) != len(b_i):
            raise MMSCertificateError(f"weights_before[{i}] length mismatch")
        if type(raw_wa[i]) is not list or len(raw_wa[i]) != len(b_i):
            raise MMSCertificateError(f"weights_after[{i}] length mismatch")
        wb_i = [_parse_rat(x, f"wb[{i}]") for x in raw_wb[i]]
        wa_i = [_parse_rat(x, f"wa[{i}]") for x in raw_wa[i]]
        if any(x < 0 or x > 1 for x in wb_i) or any(x < 0 or x > 1 for x in wa_i):
            raise MMSCertificateError(f"Weights out of [0,1] for agent {i}")
        if sum(wb_i, Fraction(0, 1)) > Fraction(1, 1) or sum(wa_i, Fraction(0, 1)) > Fraction(1, 1):
            raise MMSCertificateError(f"Agent weight sum > 1 for agent {i}")
        for k in range(len(b_i)):
            if wb_i[k] == 0 and wa_i[k] != 0:
                raise MMSCertificateError(f"Zero weight became nonzero at agent {i}, bundle {k}")
            if wb_i[k] == 1 and wa_i[k] != 1:
                raise MMSCertificateError(f"Integral weight 1 altered at agent {i}, bundle {k}")
        bundles.append(b_i)
        wb.append(wb_i)
        wa.append(wa_i)

    # Check fractional support sizes & Lemma 3.5 halving |F'| <= |F| / 2 and |C| <= |F| / 4
    f_before = sum(1 for i in range(n) for x in wb[i] if 0 < x < 1)
    f_after = sum(1 for i in range(n) for x in wa[i] if 0 < x < 1)
    if f_before < 4:
        raise MMSCertificateError("Require initial fractional support |F| >= 4 for checkpoint step")
    if 2 * f_after > f_before:
        raise MMSCertificateError(f"Support halving violated: |F'|={f_after} > |F|/2={f_before}/2")

    raw_ck = cert["checkpoints"]
    if type(raw_ck) is not list:
        raise MMSCertificateError("checkpoints must be list")
    ckpts = [_parse_int(v, "checkpoints") for v in raw_ck]
    if not ckpts or len(set(ckpts)) != len(ckpts) or ckpts != sorted(ckpts) or any(not (1 <= j <= m) for j in ckpts):
        raise MMSCertificateError("Invalid non-empty strictly increasing 1-indexed checkpoints")
    if 4 * len(ckpts) > f_before:
        raise MMSCertificateError(f"Too many checkpoints: {len(ckpts)} > |F|/4")
    ckpt_set = set(ckpts)

    # Verify total weight loss <= omega * |F|
    sum_before = sum((x for i in range(n) for x in wb[i]), Fraction(0, 1))
    sum_after = sum((x for i in range(n) for x in wa[i]), Fraction(0, 1))
    w_loss = sum_before - sum_after
    if w_loss < 0 or w_loss > omega * Fraction(f_before, 1):
        raise MMSCertificateError(f"Weight loss {w_loss} outside [0, omega*|F|={omega * f_before}]")

    # Verify prefix load constraints <= lam * j BEFORE and AFTER rounding (Lemma 3.4)
    # and exact checkpoint prefix-load conservation at every j in checkpoints
    max_ratio = Fraction(0, 1)
    for j in range(1, m + 1):
        load_b = Fraction(0, 1)
        load_a = Fraction(0, 1)
        for i in range(n):
            for k, S in enumerate(bundles[i]):
                cnt = sum(1 for p in S if p < j)
                if cnt:
                    load_b += wb[i][k] * cnt
                    load_a += wa[i][k] * cnt
        cap = lam * Fraction(j, 1)
        if load_b > cap or load_a > cap:
            raise MMSCertificateError(f"Prefix load violated at j={j}: before={load_b}, after={load_a}, cap={cap}")
        if j in ckpt_set and load_a > load_b:
            raise MMSCertificateError(f"Checkpoint prefix-load conservation violated at j={j}: after={load_a} > before={load_b}")
        r_j = load_a / Fraction(j, 1)
        if r_j > max_ratio:
            max_ratio = r_j

    if _parse_int(cert["claimed_frac_before"], "claimed_frac_before") != f_before:
        raise MMSCertificateError(f"claimed_frac_before mismatch: exact={f_before}")
    if _parse_int(cert["claimed_frac_after"], "claimed_frac_after") != f_after:
        raise MMSCertificateError(f"claimed_frac_after mismatch: exact={f_after}")
    if _parse_rat(cert["claimed_weight_loss"], "claimed_weight_loss") != w_loss:
        raise MMSCertificateError(f"claimed_weight_loss mismatch: exact={w_loss}")
    if _parse_rat(cert["claimed_max_prefix_load_ratio"], "claimed_max_prefix_load_ratio") != max_ratio:
        raise MMSCertificateError(f"claimed_max_prefix_load_ratio mismatch: exact={max_ratio}")

    return {
        "verified": True,
        "mode": cert["mode"],
        "frac_before": f_before,
        "frac_after": f_after,
        "weight_loss": format_rat(w_loss),
        "max_prefix_ratio": format_rat(max_ratio),
    }


# =============================================================================
# Mode 2: Algorithm 3.1 Steps 2 & 3 — Deadline Slot Assignment, Eq. 3.1 & Bag Filling
# =============================================================================


def _verify_mode2_deadline_bagfill(cert: Dict[str, Any]) -> Dict[str, Any]:
    req = {
        "schema",
        "mode",
        "cert_id",
        "num_agents",
        "num_items",
        "sigma",
        "lam",
        "valuations",
        "served_integral_bundles",
        "claimed_step2_assigned",
        "claimed_unassigned_pool",
        "claimed_step3_bags",
        "claimed_min_pool_value",
        "claimed_min_final_value",
    }
    _parse_exact_keys(cert, req, "Mode 2")
    n = _parse_int(cert["num_agents"], "num_agents")
    m = _parse_int(cert["num_items"], "num_items")
    if not (2 <= n <= 16 and 2 <= m <= 64):
        raise MMSCertificateError("Require 2 <= n <= 16 and 2 <= m <= 64")
    sigma = _parse_rat(cert["sigma"], "sigma")
    lam = _parse_rat(cert["lam"], "lam")
    if not (Fraction(0, 1) < sigma < Fraction(1, 1)) or not (Fraction(0, 1) < lam < Fraction(1, 1)):
        raise MMSCertificateError("Require 0 < sigma < 1 and 0 < lam < 1")

    vals = _parse_valuations(cert["valuations"], n, m, require_ido=True, require_norm=True)

    raw_srv = cert["served_integral_bundles"]
    if type(raw_srv) is not dict:
        raise MMSCertificateError("served_integral_bundles must be dict")
    served: Dict[int, List[int]] = {}
    for k_str, b_raw in raw_srv.items():
        if not _CANON_INT_RE.match(k_str):
            raise MMSCertificateError(f"Invalid agent key {k_str!r}")
        ag = int(k_str)
        if not (0 <= ag < n) or ag in served:
            raise MMSCertificateError(f"Out-of-range or duplicate agent key {ag}")
        b = _parse_bundle(b_raw, m, f"served[{ag}]")
        if sum((vals[ag][p] for p in b), Fraction(0, 1)) < sigma:
            raise MMSCertificateError(f"Served bundle for agent {ag} below sigma")
        served[ag] = b

    # Check prefix constraints <= lam * j on selected integral bundles
    for j in range(1, m + 1):
        load_j = sum(1 for b in served.values() for p in b if p < j)
        if Fraction(load_j, 1) > lam * Fraction(j, 1):
            raise MMSCertificateError(f"Integral prefix constraint violated at j={j}: {load_j} > {lam * j}")

    # Replay Step 2 ceil(k/lam) deadline assignment
    try:
        assigned, unassigned = step2_deadline_assign_exact(served, lam, m)
    except ValueError as exc:
        raise MMSCertificateError(str(exc)) from exc

    # Verify prefix reserve |U cap M[j]| >= j - floor(lam*j) >= (1-lam)*j for every j
    for j in range(1, m + 1):
        used_j = j - sum(1 for p in unassigned if p < j)
        if used_j > floor_mul_rat(lam, j):
            raise MMSCertificateError(f"Prefix assigned count exceeded floor(lam*j) at j={j}")

    # Verify Eq. (3.1) Abel summation identity & pool value >= Delta = n*(1-lam) for every unserved agent
    delta_slack = Fraction(n, 1) * (Fraction(1, 1) - lam)
    unserved = [i for i in range(n) if i not in served]
    if not unserved:
        raise MMSCertificateError("Mode 2 requires at least 1 unserved agent to audit Step 3 bag-filling")

    min_pool_val = None
    for i in unserved:
        lhs, rhs = abel_summation_exact(vals[i], unassigned)
        if lhs != rhs:
            raise MMSCertificateError(f"Eq. (3.1) Abel summation mismatch for agent {i}")
        if lhs < delta_slack:
            raise MMSCertificateError(f"Pool value {lhs} fell below Delta={delta_slack} for agent {i}")
        if min_pool_val is None or lhs < min_pool_val:
            min_pool_val = lhs

    # Replay Step 3 greedy bag-filling
    try:
        step3_bags, _leftover = step3_greedy_bag_fill_exact(unassigned, unserved, vals, sigma)
    except ValueError as exc:
        raise MMSCertificateError(str(exc)) from exc

    # Verify global allocation disjointness and sigma-MMS guarantee for ALL n agents
    seen_items: Set[int] = set()
    min_final_val = None
    for i in range(n):
        b_i = assigned[i] if i in assigned else step3_bags[i]
        for p in b_i:
            if p in seen_items:
                raise MMSCertificateError(f"Duplicate item {p} in final allocation")
            seen_items.add(p)
        v_i = sum((vals[i][p] for p in b_i), Fraction(0, 1))
        if v_i < sigma:
            raise MMSCertificateError(f"Agent {i} final value {v_i} < sigma={sigma}")
        if min_final_val is None or v_i < min_final_val:
            min_final_val = v_i

    # Exact parity checks
    exp_s2 = {str(k): v for k, v in assigned.items()}
    if cert["claimed_step2_assigned"] != exp_s2:
        raise MMSCertificateError("claimed_step2_assigned mismatch")
    if cert["claimed_unassigned_pool"] != sorted(unassigned):
        raise MMSCertificateError("claimed_unassigned_pool mismatch")
    exp_s3 = {str(k): v for k, v in step3_bags.items()}
    if cert["claimed_step3_bags"] != exp_s3:
        raise MMSCertificateError("claimed_step3_bags mismatch")
    if _parse_rat(cert["claimed_min_pool_value"], "claimed_min_pool_value") != min_pool_val:
        raise MMSCertificateError(f"claimed_min_pool_value mismatch: exact={min_pool_val}")
    if _parse_rat(cert["claimed_min_final_value"], "claimed_min_final_value") != min_final_val:
        raise MMSCertificateError(f"claimed_min_final_value mismatch: exact={min_final_val}")

    return {
        "verified": True,
        "mode": cert["mode"],
        "served_step2": len(served),
        "served_step3": len(unserved),
        "min_pool_value": format_rat(min_pool_val),
        "min_final_value": format_rat(min_final_val),
    }


# =============================================================================
# Mode 3: Section 3.3 Small-Slack Dual LP Prices, Def. 3.7 & Lemma 3.8 Prefix M1
# =============================================================================


def _verify_mode3_dual_prices(cert: Dict[str, Any]) -> Dict[str, Any]:
    req = {
        "schema",
        "mode",
        "cert_id",
        "num_agents",
        "num_items",
        "sigma",
        "lam",
        "ordinary_threshold",
        "valuations",
        "mms_partitions",
        "dual_zeta",
        "dual_alpha",
        "claimed_prices",
        "claimed_delta",
        "claimed_ordinary_agents",
        "claimed_reserved_agents",
        "claimed_prefix_M1",
        "claimed_M2_price_sum",
    }
    _parse_exact_keys(cert, req, "Mode 3")
    n = _parse_int(cert["num_agents"], "num_agents")
    m = _parse_int(cert["num_items"], "num_items")
    if not (2 <= n <= 16 and 2 <= m <= 64):
        raise MMSCertificateError("Require 2 <= n <= 16 and 2 <= m <= 64")
    sigma = _parse_rat(cert["sigma"], "sigma")
    lam = _parse_rat(cert["lam"], "lam")
    tau_ord = _parse_rat(cert["ordinary_threshold"], "ordinary_threshold")
    if not (Fraction(0, 1) < sigma < Fraction(1, 1)) or not (Fraction(0, 1) < lam <= Fraction(1, 1)) or tau_ord <= 0:
        raise MMSCertificateError("Invalid sigma / lam / ordinary_threshold")

    vals = _parse_valuations(cert["valuations"], n, m, require_ido=True, require_norm=True)

    # Verify explicit MMS partitions P_i: each is a partition of [m] into n bundles of value 1
    raw_parts = cert["mms_partitions"]
    if type(raw_parts) is not list or len(raw_parts) != n:
        raise MMSCertificateError("mms_partitions length mismatch")
    parts: List[List[List[int]]] = []
    for i in range(n):
        pi_raw = raw_parts[i]
        if type(pi_raw) is not list or len(pi_raw) != n:
            raise MMSCertificateError(f"mms_partitions[{i}] must have n={n} bundles")
        cov: Set[int] = set()
        pi: List[List[int]] = []
        for k in range(n):
            b = _parse_bundle(pi_raw[k], m, f"P[{i}][{k}]")
            if cov.intersection(b):
                raise MMSCertificateError(f"Overlapping MMS bundle in P[{i}]")
            cov.update(b)
            if sum((vals[i][p] for p in b), Fraction(0, 1)) != Fraction(1, 1):
                raise MMSCertificateError(f"MMS bundle P[{i}][{k}] does not have normalized value 1")
            pi.append(b)
        if cov != set(range(m)):
            raise MMSCertificateError(f"mms_partitions[{i}] does not cover all m={m} items")
        parts.append(pi)

    # Parse dual variables zeta_i <= 1 and alpha_j >= 0
    raw_z = cert["dual_zeta"]
    raw_a = cert["dual_alpha"]
    if type(raw_z) is not list or len(raw_z) != n or type(raw_a) is not list or len(raw_a) != m:
        raise MMSCertificateError("dual_zeta / dual_alpha length mismatch")
    zetas = [_parse_rat(raw_z[i], f"dual_zeta[{i}]") for i in range(n)]
    alphas = [_parse_rat(raw_a[j], f"dual_alpha[{j}]") for j in range(m)]
    if any(z < Fraction(0, 1) or z > Fraction(1, 1) for z in zetas):
        raise MMSCertificateError("Dual zeta_i must lie in [0, 1]")
    if any(a < Fraction(0, 1) for a in alphas):
        raise MMSCertificateError("Dual alpha_j must be >= 0")

    # Verify exact item prices p(e_k) = sum_{j>=k} alpha_j and double-sum identity p(M) = sum_j j*alpha_j = n
    prices = dual_prefix_prices_exact(alphas)
    weighted_a = sum((Fraction(j + 1, 1) * alphas[j] for j in range(m)), Fraction(0, 1))
    total_p = sum(prices, Fraction(0, 1))
    if total_p != weighted_a or total_p != Fraction(n, 1):
        raise MMSCertificateError(f"Dual price normalization failed: sum p={total_p}, sum j*alpha={weighted_a}, expected {n}")

    # Verify strong duality sum(zeta) == n*lam and sum(1 - zeta) == Delta = n*(1 - lam)
    delta = Fraction(n, 1) * (Fraction(1, 1) - lam)
    if sum(zetas, Fraction(0, 1)) != Fraction(n, 1) * lam:
        raise MMSCertificateError("Strong duality sum(zeta_i) == n*lam violated")

    # Verify dual constraints on all MMS bundles B and reduced MMS bundles B \ B' + Eq. (3.2)
    cap_small = Fraction(1, 1) - sigma
    # exact integer ceiling ceil(2 / (1 - sigma))
    two_over_cap = Fraction(2, 1) / cap_small
    ceil_2_cap = (two_over_cap.numerator + two_over_cap.denominator - 1) // two_over_cap.denominator
    seg_cap = 1 + ceil_2_cap
    for i in range(n):
        small_price_sum = Fraction(0, 1)
        for B in parts[i]:
            pB = sum((prices[pos] for pos in B), Fraction(0, 1))
            if pB < zetas[i]:
                raise MMSCertificateError(f"Dual constraint p(B) >= zeta_{i} violated")
            segs = reduced_mms_segments_exact(B, vals[i], sigma)
            if len(segs) > seg_cap:
                raise MMSCertificateError(f"Reduced MMS segment count exceeded {seg_cap}")
            for seg in segs:
                p_seg = sum((prices[pos] for pos in seg), Fraction(0, 1))
                if pB - p_seg < zetas[i]:
                    raise MMSCertificateError(f"Dual constraint p(B \\ B') >= zeta_{i} violated")
                small_price_sum += p_seg
        # Eq. (3.2): sum_{e: v_i(e) <= 1-sigma} p(e) <= (1 + ceil(2/(1-sigma))) * n * (1 - zeta_i)
        if small_price_sum > Fraction(seg_cap * n, 1) * (Fraction(1, 1) - zetas[i]):
            raise MMSCertificateError(f"Eq. (3.2) small-item price bound violated for agent {i}")

    # Partition ordinary vs reserved (Def. 3.7) & verify Markov bound |N_r| * tau_ord <= n * Delta
    n_ord = [i for i in range(n) if Fraction(n, 1) * (Fraction(1, 1) - zetas[i]) <= tau_ord]
    n_res = [i for i in range(n) if i not in n_ord]
    if not n_ord:
        raise MMSCertificateError("No ordinary agents found under ordinary_threshold")
    if Fraction(len(n_res), 1) * tau_ord > Fraction(n, 1) * delta:
        raise MMSCertificateError("Markov reserved-agent bound violated")

    # Compute exact valuable prefix M1 = {e : for all i in N_o, v_i(e) > 1 - sigma} and M2 = M \ M1
    m2 = [pos for pos in range(m) if any(vals[i][pos] <= cap_small for i in n_ord)]
    m1 = [pos for pos in range(m) if pos not in m2]
    # Because valuations are IDO, M1 must be a prefix {0, 1, ..., |M1|-1}
    if m1 != list(range(len(m1))):
        raise MMSCertificateError("M1 is not a valid IDO prefix")
    m2_price_sum = sum((prices[pos] for pos in m2), Fraction(0, 1))

    if [_parse_rat(x, "claimed_prices") for x in cert["claimed_prices"]] != prices:
        raise MMSCertificateError("claimed_prices mismatch")
    if _parse_rat(cert["claimed_delta"], "claimed_delta") != delta:
        raise MMSCertificateError(f"claimed_delta mismatch: exact={delta}")
    if cert["claimed_ordinary_agents"] != n_ord:
        raise MMSCertificateError("claimed_ordinary_agents mismatch")
    if cert["claimed_reserved_agents"] != n_res:
        raise MMSCertificateError("claimed_reserved_agents mismatch")
    if cert["claimed_prefix_M1"] != m1:
        raise MMSCertificateError("claimed_prefix_M1 mismatch")
    if _parse_rat(cert["claimed_M2_price_sum"], "claimed_M2_price_sum") != m2_price_sum:
        raise MMSCertificateError(f"claimed_M2_price_sum mismatch: exact={m2_price_sum}")

    return {
        "verified": True,
        "mode": cert["mode"],
        "delta": format_rat(delta),
        "ordinary_count": len(n_ord),
        "reserved_count": len(n_res),
        "M1_len": len(m1),
        "M2_price_sum": format_rat(m2_price_sum),
    }


# =============================================================================
# Mode 4: Goods (1-eps)-MMS & Chores (1+eps)-MMS Allocation + Bundle Comparison Audit
# =============================================================================


def _verify_mode4_mms_allocation(cert: Dict[str, Any]) -> Dict[str, Any]:
    req = {
        "schema",
        "mode",
        "cert_id",
        "setting",
        "num_agents",
        "num_items",
        "epsilon",
        "valuations",
        "mms_witness_partitions",
        "allocation",
        "comparison_queries",
        "claimed_mms_benchmarks",
        "claimed_agent_values",
        "claimed_worst_ratio",
    }
    _parse_exact_keys(cert, req, "Mode 4")
    setting = cert["setting"]
    if setting not in ("goods", "chores"):
        raise MMSCertificateError(f"setting must be 'goods' or 'chores', got {setting!r}")
    n = _parse_int(cert["num_agents"], "num_agents")
    m = _parse_int(cert["num_items"], "num_items")
    if not (2 <= n <= 16 and 2 <= m <= 64):
        raise MMSCertificateError("Require 2 <= n <= 16 and 2 <= m <= 64")
    eps = _parse_rat(cert["epsilon"], "epsilon")
    if not (Fraction(0, 1) < eps < Fraction(1, 1)):
        raise MMSCertificateError("Require 0 < epsilon < 1")

    vals = _parse_valuations(cert["valuations"], n, m, require_ido=False, require_norm=False)

    # Verify MMS witness partitions and exact benchmark mu_i
    raw_w = cert["mms_witness_partitions"]
    if type(raw_w) is not list or len(raw_w) != n:
        raise MMSCertificateError("mms_witness_partitions length mismatch")
    benchmarks: List[Fraction] = []
    for i in range(n):
        pi_raw = raw_w[i]
        if type(pi_raw) is not list or len(pi_raw) != n:
            raise MMSCertificateError(f"mms_witness_partitions[{i}] must have {n} bundles")
        cov: Set[int] = set()
        b_vals: List[Fraction] = []
        for k in range(n):
            b = _parse_bundle(pi_raw[k], m, f"W[{i}][{k}]")
            if cov.intersection(b):
                raise MMSCertificateError(f"Overlapping bundle in W[{i}]")
            cov.update(b)
            b_vals.append(sum((vals[i][p] for p in b), Fraction(0, 1)))
        if cov != set(range(m)):
            raise MMSCertificateError(f"mms_witness_partitions[{i}] does not cover [m]")
        mu_i = min(b_vals) if setting == "goods" else max(b_vals)
        if mu_i <= 0:
            raise MMSCertificateError(f"Nonpositive MMS benchmark mu_{i}={mu_i}")
        # Soundness guard against witness-weakening forgery:
        # For goods: min_k v_i(W_k) <= MMS_i <= avg_i, so min_k == avg_i certifies MMS_i = mu_i EXACTLY.
        # For chores: max(avg_i, max_e c_i(e)) <= MMS_i <= max_k c_i(W_k), so equality certifies MMS_i = mu_i EXACTLY.
        avg_i = sum(vals[i], Fraction(0, 1)) / Fraction(n, 1)
        if setting == "goods":
            if mu_i != avg_i:
                raise MMSCertificateError(
                    f"Goods witness W[{i}] is not tight (min={mu_i} != avg={avg_i}); cannot certify exact MMS_i"
                )
        else:
            lb_chore_i = max(avg_i, max(vals[i]))
            if mu_i != lb_chore_i:
                raise MMSCertificateError(
                    f"Chores witness W[{i}] is not tight (max={mu_i} != max(avg,max_item)={lb_chore_i}); cannot certify exact MMS_i"
                )
        benchmarks.append(mu_i)

    # Verify integral allocation X = (X_0, ..., X_{n-1}) partitions [m]
    raw_alloc = cert["allocation"]
    if type(raw_alloc) is not list or len(raw_alloc) != n:
        raise MMSCertificateError("allocation length mismatch")
    alloc_cov: Set[int] = set()
    agent_vals: List[Fraction] = []
    ratios: List[Fraction] = []
    for i in range(n):
        bi = _parse_bundle(raw_alloc[i], m, f"allocation[{i}]")
        if alloc_cov.intersection(bi):
            raise MMSCertificateError(f"Duplicate item in allocation[{i}]")
        alloc_cov.update(bi)
        vi = sum((vals[i][p] for p in bi), Fraction(0, 1))
        agent_vals.append(vi)
        r_i = vi / benchmarks[i]
        ratios.append(r_i)
        if setting == "goods":
            if vi < (Fraction(1, 1) - eps) * benchmarks[i]:
                raise MMSCertificateError(f"Agent {i} goods ratio {r_i} < 1 - eps = {1 - eps}")
        else:
            if vi > (Fraction(1, 1) + eps) * benchmarks[i]:
                raise MMSCertificateError(f"Agent {i} chores ratio {r_i} > 1 + eps = {1 + eps}")
    if alloc_cov != set(range(m)):
        raise MMSCertificateError("allocation does not partition all m items")

    # Verify comparison-based oracle queries (Section 1.1 ordinal bundle comparisons)
    raw_q = cert["comparison_queries"]
    if type(raw_q) is not list or not (1 <= len(raw_q) <= 64):
        raise MMSCertificateError("comparison_queries must have length in [1, 64]")
    for qi, qobj in enumerate(raw_q):
        if type(qobj) is not dict or set(qobj.keys()) != {"agent", "bundle_a", "bundle_b", "preferred"}:
            raise MMSCertificateError(f"comparison_queries[{qi}] schema mismatch")
        ag = _parse_int(qobj["agent"], f"query[{qi}].agent")
        if not (0 <= ag < n):
            raise MMSCertificateError(f"query[{qi}].agent out of range")
        ba = _parse_bundle(qobj["bundle_a"], m, f"query[{qi}].bundle_a")
        bb = _parse_bundle(qobj["bundle_b"], m, f"query[{qi}].bundle_b")
        pref = qobj["preferred"]
        if pref not in ("a", "b", "tie"):
            raise MMSCertificateError(f"query[{qi}].preferred invalid: {pref!r}")
        va = sum((vals[ag][p] for p in ba), Fraction(0, 1))
        vb = sum((vals[ag][p] for p in bb), Fraction(0, 1))
        if setting == "goods":
            exact_pref = "a" if va > vb else ("b" if vb > va else "tie")
        else:
            # For chores, lower cost is strictly preferred!
            exact_pref = "a" if va < vb else ("b" if vb < va else "tie")
        if pref != exact_pref:
            raise MMSCertificateError(f"comparison_queries[{qi}] forged preference: claimed={pref}, exact={exact_pref}")

    worst_ratio = min(ratios) if setting == "goods" else max(ratios)

    if [_parse_rat(x, "claimed_mms_benchmarks") for x in cert["claimed_mms_benchmarks"]] != benchmarks:
        raise MMSCertificateError("claimed_mms_benchmarks mismatch")
    if [_parse_rat(x, "claimed_agent_values") for x in cert["claimed_agent_values"]] != agent_vals:
        raise MMSCertificateError("claimed_agent_values mismatch")
    if _parse_rat(cert["claimed_worst_ratio"], "claimed_worst_ratio") != worst_ratio:
        raise MMSCertificateError(f"claimed_worst_ratio mismatch: exact={worst_ratio}")

    return {
        "verified": True,
        "mode": cert["mode"],
        "setting": setting,
        "epsilon": format_rat(eps),
        "worst_ratio": format_rat(worst_ratio),
        "queries_audited": len(raw_q),
    }


def verify_mms_certificate_exact(cert: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level zero-float entry point for MMS-ASYMPTOTIC-CERT-1.0."""
    if type(cert) is not dict:
        raise MMSCertificateError("Certificate root must be a JSON object")
    _assert_strict_json_types(cert)
    if cert.get("schema") != SCHEMA_VERSION:
        raise MMSCertificateError(f"Unsupported schema: {cert.get('schema')!r}")
    cid = cert.get("cert_id")
    if type(cid) is not str or not cid:
        raise MMSCertificateError("cert_id must be non-empty string")
    mode = cert.get("mode")
    if mode == "checkpoint_rounding_prefix_cert":
        return _verify_mode1_checkpoint_rounding(cert)
    if mode == "deadline_slot_and_bagfill_cert":
        return _verify_mode2_deadline_bagfill(cert)
    if mode == "small_slack_dual_price_cert":
        return _verify_mode3_dual_prices(cert)
    if mode == "goods_and_chores_mms_allocation_cert":
        return _verify_mode4_mms_allocation(cert)
    raise MMSCertificateError(f"Unknown verification mode: {mode!r}")
