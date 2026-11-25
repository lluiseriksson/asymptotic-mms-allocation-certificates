"""Pure-stdlib zero-floating-point exact-Q primitives for Paper 39:
Zehan Lin, Shengxin Liu, Biaoshuai Tao, Xiaowei Wu, & Yuhao Zhang
(arXiv:2611.33052, Nov 24, 2026) "(1 - eps)-MMS Allocations".

Author: Lluis Eriksson
License: MIT
"""
from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Sequence, Set, Tuple


def format_rat(q: Fraction) -> str:
    """Canonical reduced rational string ('n' if denominator == 1 else 'n/d')."""
    if q.denominator == 1:
        return str(q.numerator)
    return f"{q.numerator}/{q.denominator}"


def ceil_div_rat(k: int, lam: Fraction) -> int:
    """Exact integer ceiling ceil(k / lam) for integer k >= 1 and rational lam > 0."""
    if k < 1 or lam <= 0:
        raise ValueError("Require k >= 1 and lam > 0")
    # k / (ln / ld) = (k * ld) / ln
    num = k * lam.denominator
    den = lam.numerator
    return (num + den - 1) // den


def floor_mul_rat(lam: Fraction, j: int) -> int:
    """Exact integer floor floor(lam * j) for rational lam >= 0 and integer j >= 0."""
    if lam < 0 or j < 0:
        raise ValueError("Require lam >= 0 and j >= 0")
    return (lam.numerator * j) // lam.denominator


def abel_summation_exact(vals: Sequence[Fraction], subset: Set[int]) -> Tuple[Fraction, Fraction]:
    """Compute both sides of Eq. (3.1) over 0-indexed items:
      LHS = sum_{j in subset} vals[j]
      RHS = sum_{j=0}^{m-1} |subset cap {0..j}| * (vals[j] - vals[j+1])  (with vals[m]=0).
    Returns (LHS, RHS)."""
    m = len(vals)
    lhs = sum((vals[j] for j in subset), Fraction(0, 1))
    rhs = Fraction(0, 1)
    running = 0
    for j in range(m):
        if j in subset:
            running += 1
        nxt = vals[j + 1] if j + 1 < m else Fraction(0, 1)
        rhs += Fraction(running, 1) * (vals[j] - nxt)
    return lhs, rhs


def step2_deadline_assign_exact(
    selected_bundles: Dict[int, List[int]],
    lam: Fraction,
    m: int,
) -> Tuple[Dict[int, List[int]], Set[int]]:
    """Exact replay of Algorithm 3.1 Step 2:
    Given integral selected bundles {agent_i: [0-indexed item positions]} satisfying
    prefix loads <= lam * j, sorts all requests by deadline (1-indexed j = pos + 1),
    assigns the ceil(k / lam)-th item (1-indexed, so 0-indexed slot = ceil(k/lam) - 1)
    to the k-th request, and returns (assigned_bundles, unassigned_pool_set)."""
    requests: List[Tuple[int, int]] = []  # (deadline_1idx, agent_i)
    for agent_i in sorted(selected_bundles.keys()):
        for pos in sorted(selected_bundles[agent_i]):
            if not (0 <= pos < m):
                raise ValueError(f"Item position {pos} out of range [0, {m - 1}]")
            requests.append((pos + 1, agent_i))
    requests.sort(key=lambda pair: (pair[0], pair[1]))

    assigned: Dict[int, List[int]] = {i: [] for i in selected_bundles.keys()}
    used_slots: Set[int] = set()
    prev_slot = 0
    for k, (deadline, agent_i) in enumerate(requests, start=1):
        slot_1idx = ceil_div_rat(k, lam)
        if slot_1idx <= prev_slot:
            raise ValueError("Assigned slots failed strict monotonicity")
        if slot_1idx > deadline or slot_1idx > m:
            raise ValueError(f"Slot {slot_1idx} exceeded deadline {deadline} or m={m}")
        prev_slot = slot_1idx
        slot_0idx = slot_1idx - 1
        used_slots.add(slot_0idx)
        assigned[agent_i].append(slot_0idx)

    unassigned = set(range(m)) - used_slots
    return assigned, unassigned


def step3_greedy_bag_fill_exact(
    unassigned_pool: Set[int],
    unserved_agents: Sequence[int],
    valuations: Sequence[Sequence[Fraction]],
    sigma: Fraction,
) -> Tuple[Dict[int, List[int]], List[int]]:
    """Exact replay of Algorithm 3.1 Step 3 greedy bag-filling:
    Repeatedly scans remaining pool in IDO order (ascending 0-indexed position),
    stops at the shortest prefix B' where max_{i in remaining} v_i(B') >= sigma,
    assigns B' to the smallest-index qualifying agent, and removes B' from pool."""
    pool = sorted(unassigned_pool)
    waiting = list(unserved_agents)
    allocated: Dict[int, List[int]] = {}

    while waiting:
        bag: List[int] = []
        chosen_agent = -1
        for pos in pool:
            bag.append(pos)
            # Check qualifying agents in deterministic index order
            qual = [
                ag
                for ag in waiting
                if sum((valuations[ag][p] for p in bag), Fraction(0, 1)) >= sigma
            ]
            if qual:
                chosen_agent = min(qual)
                break
        if chosen_agent == -1:
            raise ValueError("Step 3 bag-filling exhausted pool before serving all agents")
        allocated[chosen_agent] = list(bag)
        bag_set = set(bag)
        pool = [p for p in pool if p not in bag_set]
        waiting.remove(chosen_agent)

    return allocated, pool


def dual_prefix_prices_exact(alphas: Sequence[Fraction]) -> List[Fraction]:
    """Exact Section 3.3 dual item prices p(e_k) = sum_{j=k}^{m-1} alpha_j."""
    m = len(alphas)
    prices = [Fraction(0, 1)] * m
    running = Fraction(0, 1)
    for k in range(m - 1, -1, -1):
        running += alphas[k]
        prices[k] = running
    return prices


def reduced_mms_segments_exact(
    bundle: Sequence[int],
    vals_i: Sequence[Fraction],
    sigma: Fraction,
) -> List[List[int]]:
    """Exact Section 3.1 greedy consecutive segmentation of items e in bundle with v_i(e) <= 1 - sigma."""
    cap = Fraction(1, 1) - sigma
    small_items = [pos for pos in sorted(bundle) if vals_i[pos] <= cap]
    if not small_items:
        return []
    segs: List[List[int]] = []
    cur: List[int] = []
    cur_val = Fraction(0, 1)
    for pos in small_items:
        v = vals_i[pos]
        if cur and cur_val + v > cap:
            segs.append(cur)
            cur = [pos]
            cur_val = v
        else:
            cur.append(pos)
            cur_val += v
    if cur:
        segs.append(cur)
    return segs
