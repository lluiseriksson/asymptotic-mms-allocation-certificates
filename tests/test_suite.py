"""Comprehensive pytest & 60-attack live adversarial red-team suite for Paper 39:
MMS-ASYMPTOTIC-CERT-1.0 (Lin, Liu, Tao, Wu, & Zhang, arXiv:2611.33052).

Author: Lluis Eriksson
License: MIT
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mms_certifier import MMSCertificateError, verify_mms_certificate_exact

CERT_DIR = ROOT / "certificates"


def _load(name: str) -> Dict[str, Any]:
    return json.loads((CERT_DIR / f"{name}.json").read_text(encoding="utf-8"))


def test_all_16_canonical_certificates_verify() -> None:
    paths = sorted(CERT_DIR.glob("*.json"))
    assert len(paths) == 16, f"Expected 16 certificates, found {len(paths)}"
    for p in paths:
        res = verify_mms_certificate_exact(json.loads(p.read_text(encoding="utf-8")))
        assert res["verified"] is True


def _mut(base: Dict[str, Any], fn: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
    c = copy.deepcopy(base)
    fn(c)
    return c


def build_60_adversarial_attacks() -> List[Tuple[str, Any]]:
    m1 = _load("m1_ckpt_3ag_8it_exact_halving")
    m2 = _load("m2_bagfill_3ag_9it_lam_2_3")
    m3 = _load("m3_dual_4ag_8it_one_reserved")
    m4g = _load("m4_goods_3ag_9it_95pct_mms")
    m4c = _load("m4_chores_3ag_9it_105pct_mms")

    attacks: List[Tuple[str, Any]] = [
        # --- Global Schema / Type / Rational Hardening (12 attacks) ---
        ("g01_non_dict_root", [1, 2, 3]),
        ("g02_wrong_schema", _mut(m1, lambda c: c.__setitem__("schema", "MMS-ASYMPTOTIC-CERT-2.0"))),
        ("g03_empty_cert_id", _mut(m1, lambda c: c.__setitem__("cert_id", ""))),
        ("g04_unknown_mode", _mut(m1, lambda c: c.__setitem__("mode", "bogus"))),
        ("g05_float_leak", _mut(m1, lambda c: c.__setitem__("sigma", 0.75))),
        ("g06_none_injection", _mut(m1, lambda c: c.__setitem__("sigma", None))),
        ("g07_bool_as_int", _mut(m1, lambda c: c.__setitem__("num_agents", True))),
        ("g08_unreduced_fraction", _mut(m1, lambda c: c.__setitem__("sigma", "6/8"))),
        ("g09_den_one_fraction", _mut(m1, lambda c: c["weights_before"][2].__setitem__(0, "1/1"))),
        ("g10_neg_zero_rat", _mut(m1, lambda c: c["weights_after"][0].__setitem__(1, "-0"))),
        ("g11_oversized_rat_str", _mut(m1, lambda c: c.__setitem__("sigma", "1/" + "9" * 1200))),
        ("g12_huge_int_bitlen", _mut(m1, lambda c: c.__setitem__("num_agents", 1 << 3000))),
        # --- Mode 1: Algorithm 3.1 Step 1 Checkpoint Rounding (12 attacks) ---
        ("m1_01_extra_key", _mut(m1, lambda c: c.__setitem__("extra", 1))),
        ("m1_02_ido_violation", _mut(m1, lambda c: (
            c["valuations"][0].__setitem__(0, "1/2"),
            c["valuations"][0].__setitem__(1, "3/5"),
        ))),
        ("m1_03_norm_sum_violation", _mut(m1, lambda c: c["valuations"][0].__setitem__(7, "1/3"))),
        ("m1_04_bundle_below_sigma", _mut(m1, lambda c: c["candidate_bundles"][0].__setitem__(0, [6, 7]))),
        ("m1_05_support_not_halved", _mut(m1, lambda c: (
            c["weights_after"][0].__setitem__(1, "1/4"),
            c.__setitem__("claimed_frac_after", 3),
        ))),
        ("m1_06_too_many_ckpts", _mut(m1, lambda c: c.__setitem__("checkpoints", [2, 4]))),
        ("m1_07_unsorted_ckpts", _mut(_load("m1_ckpt_5ag_12it_two_ckpts"), lambda c: c.__setitem__("checkpoints", [8, 4]))),
        ("m1_08_ckpt_load_increased", _mut(m1, lambda c: (
            c["weights_after"][0].__setitem__(0, "3/4"),
            c["weights_after"][0].__setitem__(1, "1/4"),
        ))),
        ("m1_09_integral_altered", _mut(m1, lambda c: c["weights_after"][2].__setitem__(0, "1/2"))),
        ("m1_10_weight_loss_exceeds_omega", _mut(m1, lambda c: c.__setitem__("omega", "1/8"))),
        ("m1_11_prefix_load_exceeds_lam", _mut(m1, lambda c: c.__setitem__("lam", "1/2"))),
        ("m1_12_wrong_claimed_loss", _mut(m1, lambda c: c.__setitem__("claimed_weight_loss", "0"))),
        # --- Mode 2: Algorithm 3.1 Steps 2-3 Deadline & Bag-Filling (12 attacks) ---
        ("m2_01_extra_key", _mut(m2, lambda c: c.__setitem__("extra", 1))),
        ("m2_02_lam_one", _mut(m2, lambda c: c.__setitem__("lam", "1"))),
        ("m2_03_served_below_sigma", _mut(m2, lambda c: c["served_integral_bundles"].__setitem__("0", [7, 8]))),
        ("m2_04_prefix_cap_violated", _mut(m2, lambda c: c["served_integral_bundles"].__setitem__("0", [0, 1]))),
        ("m2_05_no_unserved_agents", _mut(m2, lambda c: c["served_integral_bundles"].__setitem__("2", [3, 6, 8]))),
        ("m2_06_invalid_agent_key", _mut(m2, lambda c: c["served_integral_bundles"].__setitem__("9", [4, 5]))),
        ("m2_07_bagfill_starvation", _mut(m2, lambda c: c.__setitem__("sigma", "99/100"))),
        ("m2_08_wrong_step2_assigned", _mut(m2, lambda c: c["claimed_step2_assigned"].__setitem__("0", [0, 1]))),
        ("m2_09_wrong_unassigned_pool", _mut(m2, lambda c: c["claimed_unassigned_pool"].append(0))),
        ("m2_10_wrong_step3_bags", _mut(m2, lambda c: c["claimed_step3_bags"].__setitem__("2", [0]))),
        ("m2_11_wrong_min_pool_val", _mut(m2, lambda c: c.__setitem__("claimed_min_pool_value", "1"))),
        ("m2_12_wrong_min_final_val", _mut(m2, lambda c: c.__setitem__("claimed_min_final_value", "1"))),
        # --- Mode 3: Section 3.3 Dual Prices, Def. 3.7 & Lemma 3.8 (12 attacks) ---
        ("m3_01_extra_key", _mut(m3, lambda c: c.__setitem__("extra", 1))),
        ("m3_02_mms_bundle_not_unit", _mut(m3, lambda c: (
            c["mms_partitions"][0].__setitem__(0, [0, 1]),
            c["mms_partitions"][0].__setitem__(1, [4, 5]),
        ))),
        ("m3_03_mms_partition_overlap", _mut(m3, lambda c: c["mms_partitions"][0].__setitem__(1, [0, 5]))),
        ("m3_04_zeta_above_one", _mut(m3, lambda c: c["dual_zeta"].__setitem__(0, "11/10"))),
        ("m3_05_negative_alpha", _mut(m3, lambda c: c["dual_alpha"].__setitem__(0, "-1/10"))),
        ("m3_06_price_sum_not_n", _mut(m3, lambda c: c["dual_alpha"].__setitem__(7, "1/10"))),
        ("m3_07_strong_duality_broken", _mut(m3, lambda c: c.__setitem__("lam", "1/2"))),
        ("m3_08_reduced_mms_dual_violated", _mut(m3, lambda c: c["dual_zeta"].__setitem__(3, "1/2"))),
        ("m3_09_no_ordinary_agents", _mut(m3, lambda c: c.__setitem__("ordinary_threshold", "1/100"))),
        ("m3_10_wrong_ordinary_list", _mut(m3, lambda c: c.__setitem__("claimed_ordinary_agents", [0, 1, 2, 3]))),
        ("m3_11_wrong_prefix_M1", _mut(m3, lambda c: c.__setitem__("claimed_prefix_M1", [0, 1]))),
        ("m3_12_wrong_M2_price_sum", _mut(m3, lambda c: c.__setitem__("claimed_M2_price_sum", "0"))),
        # --- Mode 4: Goods (1-eps)-MMS & Chores (1+eps)-MMS + Bundle Queries (12 attacks) ---
        ("m4_01_bad_setting", _mut(m4g, lambda c: c.__setitem__("setting", "mixed"))),
        ("m4_02_eps_out_of_range", _mut(m4g, lambda c: c.__setitem__("epsilon", "1"))),
        ("m4_03_goods_weak_witness_forgery", _mut(m4g, lambda c: (
            c["mms_witness_partitions"][0].__setitem__(0, [8]),
            c["mms_witness_partitions"][0].__setitem__(1, [0, 1, 2, 3]),
            c["mms_witness_partitions"][0].__setitem__(2, [4, 5, 6, 7]),
        ))),
        ("m4_04_chores_weak_witness_forgery", _mut(m4c, lambda c: (
            c["mms_witness_partitions"][0].__setitem__(0, [0, 1, 2, 3]),
            c["mms_witness_partitions"][0].__setitem__(1, [4, 5, 6]),
            c["mms_witness_partitions"][0].__setitem__(2, [7, 8]),
        ))),
        ("m4_05_alloc_duplicate_item", _mut(m4g, lambda c: c["allocation"][1].append(0))),
        ("m4_06_alloc_missing_item", _mut(m4g, lambda c: c["allocation"][0].pop())),
        ("m4_07_goods_mms_ratio_violated", _mut(m4g, lambda c: c.__setitem__("epsilon", "1/100"))),
        ("m4_08_chores_mms_ratio_violated", _mut(m4c, lambda c: c.__setitem__("epsilon", "1/100"))),
        ("m4_09_forged_goods_query", _mut(m4g, lambda c: c["comparison_queries"][0].__setitem__("preferred", "a"))),
        ("m4_10_forged_chores_query", _mut(m4c, lambda c: c["comparison_queries"][0].__setitem__("preferred", "b"))),
        ("m4_11_empty_queries", _mut(m4g, lambda c: c.__setitem__("comparison_queries", []))),
        ("m4_12_wrong_worst_ratio", _mut(m4g, lambda c: c.__setitem__("claimed_worst_ratio", "1"))),
    ]
    assert len(attacks) == 60
    return attacks


def test_60_live_adversarial_attacks_rejected() -> None:
    attacks = build_60_adversarial_attacks()
    rejected = 0
    for name, payload in attacks:
        try:
            verify_mms_certificate_exact(payload)
        except MMSCertificateError:
            rejected += 1
        else:
            raise AssertionError(f"Adversarial attack {name!r} was NOT rejected!")
    assert rejected == 60


def test_exact_no_float_source_audit() -> None:
    for rel in ["src/mms_exact.py", "src/mms_certifier.py"]:
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert "float(" not in src, f"float() found in {rel}"
        assert "math." not in src, f"math. found in {rel}"
        assert "numpy" not in src, f"numpy found in {rel}"
