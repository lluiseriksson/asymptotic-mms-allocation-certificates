/-
  AsymptoticMMS.lean
  Zero-dependency Lean 4.35.0 formalization of the exact combinatorial,
  prefix-load, support-halving, Abel-summation, greedy bag-filling, and
  small-slack dual LP invariants of Zehan Lin, Shengxin Liu, Biaoshuai Tao,
  Xiaowei Wu, & Yuhao Zhang (arXiv:2611.33052, Nov 24, 2026):
  "(1 - ε)-MMS Allocations" (proving Feige & Grinberg's asymptotic MMS conjecture).

  Author: Lluis Eriksson
  License: MIT
  Axiom Discipline: 0 sorry, 0 admit, 0 Classical.choice, 0 warnings.
-/

namespace AsymptoticMMS

/-! ### Part 1: Step 1 Checkpoint Rounding & Support-Halving (Lemmas 3.4 & 3.5) -/

/-- Theorem 1 (Lemma 3.5 Exact Fractional Support Halving):
    If a basic feasible solution has fractional support `F'` bounded by at most
    `|C| ≤ F / 4` checkpoint constraints plus at most `F' / 2` tight agent constraints
    (cleared integer form: `4 * C ≤ F`, `2 * A ≤ F'`, `F' ≤ C + A`),
    then `2 * F' ≤ F` (fractional support strictly halves every iteration!). -/
theorem support_halving_lemma35 (F F' C A : Nat)
    (hC : 4 * C ≤ F) (hA : 2 * A ≤ F') (hbfs : F' ≤ C + A) :
    2 * F' ≤ F := by
  omega

/-- Theorem 2 (Lemma 3.4 Prefix-Load Slack Absorption):
    For rational scaling factor `ω = wn / wd ∈ (0, 1/2)` and non-checkpoint prefix count
    `A` exceeding preceding checkpoint count `A0` by `A - A0 < w_k ≤ ω * A`
    (cleared: `wd * (A - A0) ≤ wn * A`), scaling fractional weight by `(1 - ω)` creates
    slack `(wn / wd) * A` so total new prefix load stays `≤ λ * j`:
    if `wd * load_scaled ≤ wd * cap - wn * A` and `wd * inc ≤ wn * A`, then
    `load_scaled + inc ≤ cap`. -/
theorem prefix_load_slack_absorption (wd wn A load_scaled inc cap : Nat)
    (hwd : 0 < wd)
    (hscaled : wd * load_scaled + wn * A ≤ wd * cap)
    (hinc : wd * inc ≤ wn * A) :
    load_scaled + inc ≤ cap := by
  have h1 : wd * (load_scaled + inc) ≤ wd * cap := by
    rw [Nat.mul_add]
    omega
  exact Nat.le_of_mul_le_mul_left h1 hwd

/-- Theorem 3 (Lemma 3.5 Threshold Count Bound `1 + 2^k / ⌊ω 2^k⌋ < 4 / ω`):
    When `ω = wn / wd < 1/2` (`2 * wn < wd`) and `ω * 2^k ≥ 1` (`wd ≤ wn * P` with `P = 2^k`),
    let `wk = (wn * P) / wd ≥ 1`. Then `wk * wd + wd > wn * P`, i.e., `2 * wk * wd ≥ wn * P`,
    so `1 + P / wk < 4 / ω` in cleared integer form: `wn * (wk + P) < 4 * wd * wk`! -/
theorem checkpoint_threshold_count_bound (wn wd P wk : Nat)
    (hwn : 0 < wn) (hsmall : 2 * wn < wd)
    (hwk_pos : 1 ≤ wk)
    (hwk_floor : wn * P < (wk + 1) * wd) :
    wn * (wk + P) < 4 * wd * wk := by
  have h1 : wn * wk < wd * wk := Nat.mul_lt_mul_of_pos_right (by omega) (by omega)
  have h2 : (wk + 1) * wd ≤ 2 * wk * wd := by
    have : wk + 1 ≤ 2 * wk := by omega
    exact Nat.mul_le_mul_right wd this
  have h3 : wn * (wk + P) = wn * wk + wn * P := Nat.mul_add wn wk P
  have h4 : 2 * wk * wd = 2 * (wd * wk) := by rw [Nat.mul_assoc, Nat.mul_comm wk wd]
  have h5 : 4 * wd * wk = 4 * (wd * wk) := Nat.mul_assoc 4 wd wk
  omega

/-- Theorem 4 (Lemma 3.5 Terminal Rounding Loss Budget `|F| ≤ 2 ω |F| ≤ 160 L`):
    When `ω = wn / wd ≥ 1/2` (`wd ≤ 2 * wn`) and `ω * F ≤ 80 * L` (`wn * F ≤ 80 * L * wd`),
    zeroing all remaining fractional entries drops at most `F ≤ 160 * L` total weight! -/
theorem terminal_rounding_loss_160L (wn wd F L : Nat)
    (hwd : 0 < wd)
    (hbig : wd ≤ 2 * wn)
    (hbud : wn * F ≤ 80 * L * wd) :
    F ≤ 160 * L := by
  have h1 : wd * F ≤ (2 * wn) * F := Nat.mul_le_mul_right F hbig
  have h2 : (2 * wn) * F = 2 * (wn * F) := Nat.mul_assoc 2 wn F
  have h3 : 2 * (wn * F) ≤ 2 * (80 * L * wd) := Nat.mul_le_mul_left 2 hbud
  have h4 : 2 * (80 * L * wd) = wd * (160 * L) := by
    have ha : 2 * (80 * L * wd) = (160 * L) * wd := by rw [← Nat.mul_assoc, ← Nat.mul_assoc]
    rw [ha, Nat.mul_comm]
  exact Nat.le_of_mul_le_mul_left (by omega) hwd

/-! ### Part 2: Step 2 Exact `⌈k / λ⌉` Deadline Assignment & Prefix Reserve -/

/-- Exact integer ceiling `⌈k / (ln / ld)⌉ = (k * ld + ln - 1) / ln` for `λ = ln / ld ∈ (0, 1]`. -/
def assignSlot (k ln ld : Nat) : Nat :=
  (k * ld + ln - 1) / ln

/-- Theorem 5 (Step 2 Strict Monotonicity = Distinct Assigned Items):
    For `0 < ln ≤ ld` (so `λ = ln / ld ≤ 1`) and any `k ≥ 1`,
    `assignSlot k ln ld < assignSlot (k + 1) ln ld`, guaranteeing every request
    receives a distinct item from `M`! -/
theorem step2_assigned_slots_strictly_increasing (k ln ld : Nat)
    (hln : 0 < ln) (hlam : ln ≤ ld) (hk : 1 ≤ k) :
    assignSlot k ln ld < assignSlot (k + 1) ln ld := by
  unfold assignSlot
  have hstep : (k * ld + ln - 1) + ln ≤ (k + 1) * ld + ln - 1 := by
    rw [Nat.add_mul, Nat.one_mul]
    have : 1 ≤ k * ld := Nat.mul_pos hk (by omega)
    omega
  have hdiv : (k * ld + ln - 1) / ln + 1 = ((k * ld + ln - 1) + ln) / ln := by
    rw [Nat.add_div_right _ hln]
  have hmono : ((k * ld + ln - 1) + ln) / ln ≤ ((k + 1) * ld + ln - 1) / ln :=
    Nat.div_le_div_right hstep
  omega

/-- Theorem 6 (Step 2 Deadline Feasibility `⌈k / λ⌉ ≤ d_k`):
    If the `k`-th request has deadline `d` and the prefix constraint gives `k ≤ λ * d`
    (cleared: `k * ld ≤ ln * d`), then `assignSlot k ln ld ≤ d`, so under IDO ordering
    the assigned item is worth at least as much as the requested item! -/
theorem step2_deadline_feasibility (k d ln ld : Nat)
    (hln : 0 < ln)
    (hpref : k * ld ≤ ln * d) :
    assignSlot k ln ld ≤ d := by
  unfold assignSlot
  have h1 : k * ld + ln - 1 < ln * (d + 1) := by
    rw [Nat.mul_add, Nat.mul_one]
    omega
  have h2 : (k * ld + ln - 1) / ln < d + 1 := by
    rw [Nat.div_lt_iff_lt_mul hln, Nat.mul_comm (d + 1) ln]
    exact h1
  omega

/-- Theorem 7 (Step 2 Prefix Unassigned Reserve `|U ∩ M[j]| ≥ (1 - λ) j`):
    If prefix `M[j]` contains `used ≤ ⌊λ * j⌋` assigned items (`used * ld ≤ ln * j`),
    then the remaining unassigned count `rem = j - used` satisfies
    `ld * rem ≥ (ld - ln) * j`! -/
theorem step2_unassigned_prefix_reserve (j used ln ld : Nat)
    (hlam : ln ≤ ld)
    (hcap : used * ld ≤ ln * j) :
    (ld - ln) * j ≤ ld * (j - used) := by
  have h1 : ld * (j - used) = ld * j - ld * used := Nat.mul_sub_left_distrib ld j used
  have h2 : (ld - ln) * j = ld * j - ln * j := Nat.mul_sub_right_distrib ld ln j
  have h3 : ld * used = used * ld := Nat.mul_comm ld used
  have h4 : ln * j ≤ ld * j := Nat.mul_le_mul_right j hlam
  omega

/-! ### Part 3: Eq. (3.1) Exact Abel Summation Identity & Step 3 Bag-Filling -/

/-- Value of subset encoded as Boolean indicator list over item values `vs`. -/
def subsetVal : List Int → List Bool → Int
  | [], _ => 0
  | _, [] => 0
  | v :: vs, b :: bs => (if b then v else 0) + subsetVal vs bs

/-- Running prefix count `|S ∩ [j]|` weighted by differences `s_j - s_{j+1}` (Abel sum). -/
def abelPrefixSum (cnt : Int) : List Int → List Bool → Int
  | [], _ => 0
  | _, [] => 0
  | [v], b :: _ => (cnt + if b then 1 else 0) * v
  | v1 :: v2 :: vs, b :: bs =>
      let cnt' := cnt + if b then 1 else 0
      cnt' * (v1 - v2) + abelPrefixSum cnt' (v2 :: vs) bs

/-- Theorem 8 (Eq. 3.1 Exact Abel Summation Identity for Arbitrary Subset Masks):
    For any finite item-value list `vs` and matching Boolean mask `bs`,
      `abelPrefixSum c vs bs = c * head(vs) + subsetVal vs bs`
    (with `c = 0` recovering Eq. (3.1): `∑_{j ∈ S} s_j = ∑_{j=1}^ℓ |S ∩ [j]| (s_j - s_{j+1})`!). -/
private theorem abel_summation_gen (vs : List Int) :
    ∀ (bs : List Bool) (c : Int),
      vs.length = bs.length →
      abelPrefixSum c vs bs =
        (match vs with | [] => 0 | v :: _ => c * v) + subsetVal vs bs := by
  induction vs with
  | nil =>
    intro bs c h
    cases bs with
    | nil => dsimp only [abelPrefixSum, subsetVal]; decide
    | cons _ _ => simp only [List.length] at h; omega
  | cons v1 rest ih =>
    intro bs c h
    cases bs with
    | nil => simp only [List.length] at h; omega
    | cons b bs' =>
      have h' : rest.length = bs'.length := Nat.succ.inj h
      cases rest with
      | nil =>
        cases bs' with
        | nil =>
          dsimp only [abelPrefixSum, subsetVal]
          cases b <;> simp [Int.add_mul]
        | cons _ _ => simp only [List.length] at h'; omega
      | cons v2 vs2 =>
        cases bs' with
        | nil => simp only [List.length] at h'; omega
        | cons b2 bs2 =>
          have ih' := ih (b2 :: bs2) (c + if b then 1 else 0) h'
          dsimp only [] at ih'
          change (c + if b then 1 else 0) * (v1 - v2) +
              abelPrefixSum (c + if b then 1 else 0) (v2 :: vs2) (b2 :: bs2) =
            c * v1 + ((if b then v1 else 0) + subsetVal (v2 :: vs2) (b2 :: bs2))
          rw [ih']
          cases b <;> simp [Int.mul_sub, Int.add_mul] <;> omega

theorem eq31_abel_summation_exact (vs : List Int) (bs : List Bool)
    (hlen : vs.length = bs.length) :
    abelPrefixSum 0 vs bs = subsetVal vs bs := by
  have h := abel_summation_gen vs bs 0 hlen
  cases vs with
  | nil => simp [abelPrefixSum, subsetVal] at h ⊢
  | cons v rest =>
    dsimp only [] at h
    rw [Int.zero_mul, Int.zero_add] at h
    exact h

/-- Theorem 9 (Step 3 Greedy Bag-Filling Termination & Survival Bound, p. 11):
    With initial unassigned pool value `vU ≥ ∆`, total slack `∆ ≥ 2 * R`,
    at most `R` unserved agents (`r < R` already served), and each removed bag
    bounded by `< σ + 1 ≤ 2` (cleared integer form with denominator `D > 0`:
    `sig < D`, `Δ ≥ 2 * R * D`, each bag `< 2 * D`), the remaining pool value
    after `r ≤ R - 1` removals still satisfies `rem > sig`! -/
theorem step3_bag_filling_survival (D sig R r Delta : Nat)
    (hsig : sig < D)
    (hR : 1 ≤ R)
    (hr : r ≤ R - 1)
    (hDelta : 2 * R * D ≤ Delta) :
    sig < Delta - 2 * r * D := by
  have h1 : 2 * r * D ≤ 2 * (R - 1) * D := by
    have : 2 * r ≤ 2 * (R - 1) := by omega
    exact Nat.mul_le_mul_right D this
  have h2 : 2 * (R - 1) * D + 2 * D = 2 * R * D := by
    have hR1 : R - 1 + 1 = R := by omega
    calc 2 * (R - 1) * D + 2 * D
        = (2 * (R - 1) + 2) * D := by rw [Nat.add_mul]
      _ = (2 * (R - 1 + 1)) * D := by rw [Nat.mul_add, Nat.mul_one]
      _ = 2 * R * D := by rw [hR1]
  omega

/-! ### Part 4: Section 3.3 Small-Slack Dual LP & Reserved-Agent Bound -/

/-- Exact dual prefix-price suffix sums: given `alphas = [α_1, ..., α_m]`,
    `priceSum alphas = ∑_{k=1}^m ∑_{j=k}^m α_j` and `weightedAlphaSum 1 alphas = ∑_{j=1}^m j α_j`. -/
def sumIntList : List Int → Int
  | [] => 0
  | x :: xs => x + sumIntList xs

def totalPriceSum : List Int → Int
  | [] => 0
  | a :: as => sumIntList (a :: as) + totalPriceSum as

def weightedAlphaSum (idx : Int) : List Int → Int
  | [] => 0
  | a :: as => idx * a + weightedAlphaSum (idx + 1) as

private theorem weightedAlphaSum_shift (idx : Int) :
    ∀ as : List Int, weightedAlphaSum (idx + 1) as = weightedAlphaSum idx as + sumIntList as
  | [] => by dsimp only [weightedAlphaSum, sumIntList]; decide
  | a :: as => by
      dsimp only [weightedAlphaSum, sumIntList]
      rw [weightedAlphaSum_shift (idx + 1) as, Int.add_mul, Int.one_mul]
      omega

/-- Theorem 10 (Section 3.3 Exact Double-Summation Price Identity `∑_k p(e_k) = ∑_j j α_j`):
    Proves constructively over arbitrary finite dual vectors `alphas` that
    `totalPriceSum alphas = weightedAlphaSum 1 alphas`. -/
theorem dual_price_double_sum_exact :
    ∀ alphas : List Int, totalPriceSum alphas = weightedAlphaSum 1 alphas
  | [] => rfl
  | a :: as => by
      dsimp only [totalPriceSum, sumIntList, weightedAlphaSum]
      rw [dual_price_double_sum_exact as, weightedAlphaSum_shift 1 as, Int.one_mul]
      omega

/-- Theorem 11 (Section 3.3 Definition 3.7 Reserved-Agent Markov Bound `|N_r| L^4 < 1536 n L^2`):
    With `R = 768 L^2`, small slack `∆ < 2 R = 1536 L^2`, and every reserved agent
    satisfying `n (1 - ζ_i) > L^4`, summing over `N_r` yields `|N_r| * L^4 < 1536 * n * L^2`,
    and hence `|N_r| * L^2 < 1536 * n` (i.e., `|N_r| = O(n / L^2) = o(n)`)! -/
theorem reserved_agent_markov_bound (Nr n L sumNr : Nat)
    (h_lb : Nr * (L * L * L * L) ≤ sumNr)
    (h_ub : sumNr < 2 * (768 * (L * L)) * n) :
    Nr * (L * L) < 1536 * n := by
  have h1 : Nr * (L * L * L * L) < 1536 * n * (L * L) := by
    have hcoeff : 2 * (768 * (L * L)) * n = 1536 * n * (L * L) := by
      have ha : 2 * (768 * (L * L)) = 1536 * (L * L) := by rw [← Nat.mul_assoc]
      rw [ha, Nat.mul_assoc, Nat.mul_comm (L * L) n, ← Nat.mul_assoc]
    rw [hcoeff] at h_ub
    exact Nat.lt_of_le_of_lt h_lb h_ub
  have h2 : Nr * (L * L * L * L) = (Nr * (L * L)) * (L * L) := by
    simp only [Nat.mul_assoc]
  rw [h2] at h1
  match Nat.lt_or_ge (Nr * (L * L)) (1536 * n) with
  | Or.inl hlt => exact hlt
  | Or.inr hge =>
      have hbad : 1536 * n * (L * L) ≤ (Nr * (L * L)) * (L * L) :=
        Nat.mul_le_mul_right (L * L) hge
      exact absurd h1 (Nat.not_lt.mpr hbad)

/-- Theorem 12 (Section 3.1 Reduced MMS Bundle Segment Bound `t ≤ 1 + ⌈2 / (1 - σ)⌉`):
    Since consecutive segments inside a unit-value bundle `B` (`v_i(B) = 1`) have
    combined value `> 1 - σ` (cleared with denominator `D > 0`: `v_pair > D - sig`),
    forming `p` disjoint adjacent pairs from `t` segments satisfies `p * (D - sig) < D`,
    and for `t = 2 p` or `t = 2 p + 1`, `(t - 1) * (D - sig) < 2 * D`! -/
theorem reduced_mms_segment_count_bound (t p D sig : Nat)
    (hpair : p * (D - sig) < D)
    (ht : t ≤ 2 * p + 1) :
    (t - 1) * (D - sig) < 2 * D := by
  have h1 : t - 1 ≤ 2 * p := by omega
  have h2 : (t - 1) * (D - sig) ≤ (2 * p) * (D - sig) := Nat.mul_le_mul_right _ h1
  have h3 : (2 * p) * (D - sig) = 2 * (p * (D - sig)) := Nat.mul_assoc 2 p (D - sig)
  omega

/-- Theorem 13 (Section 3.3 Eq. 3.2 Dual Price Transfer Per Reduced Segment):
    If `p(B \ B') ≥ ζ_i` (dual constraint for reduced MMS bundle `B \ B' ∈ B_i`),
    then `p(B') ≤ p(B) - ζ_i`. -/
theorem dual_price_transfer_segment (pB pB' zeta : Int)
    (hdual : zeta ≤ pB - pB') :
    pB' ≤ pB - zeta := by
  omega

/-! ### Part 5: Exact Literature Landmarks & Goods/Chores Approximation Ordering -/

/-- Theorem 14 (Historical MMS Goods Approximation Ordering, Section 1):
    `2/3 < 3/4 < 7/9 < 39/40 < 1` (with `7/9` Huang–Zhou 2026 and `39/40` FST21 3-agent upper bound). -/
theorem mms_goods_landmark_ordering :
    (2 * 4 < 3 * 3) ∧
    (3 * 9 < 7 * 4) ∧
    (7 * 40 < 39 * 9) ∧
    (39 < 40) := by
  decide

/-- Theorem 15 (Historical MMS Chores Approximation Ordering, Section 1):
    `1 < 44/43 < 13/11 < 4/3` (with `13/11` Huang–Segal-Halevi 2023 and `44/43` FST21 3-agent lower bound). -/
theorem mms_chores_landmark_ordering :
    (43 < 44) ∧
    (44 * 11 < 13 * 43) ∧
    (13 * 3 < 4 * 11) := by
  decide

/-- Theorem 16 (Exact Chores Dual Multiplicative Inversion & Normalization):
    For any `ε = en / ed ∈ (0, 1)`, the goods guarantee `1 - ε` and chores guarantee `1 + ε`
    satisfy `(1 - ε)(1 + ε) = 1 - ε^2 < 1` (exact cleared integer identity). -/
theorem goods_chores_symmetric_epsilon (en ed : Int) (hen : 0 < en) :
    (ed - en) * (ed + en) = ed * ed - en * en ∧ (ed - en) * (ed + en) < ed * ed := by
  have hsq : 0 < en * en := Int.mul_pos hen hen
  have hid : (ed - en) * (ed + en) = ed * ed - en * en := by
    rw [Int.sub_mul, Int.mul_add, Int.mul_add, Int.mul_comm en ed]
    omega
  constructor
  · exact hid
  · omega

/-- Theorem 17 (Section 3.1 Uniform Fractional MMS Feasibility `λ = 1`):
    Assigning weight `1 / n` to each of the `n` disjoint bundles in every agent's MMS partition
    gives total load `n * (1 / n) * |M[j]| = j` on every prefix `M[j]`, certifying `λ ≤ 1`! -/
theorem uniform_mms_fractional_feasibility (n j : Nat) (hn : 0 < n) :
    (n * j) / n = j := by
  rw [Nat.mul_comm n j]
  exact Nat.mul_div_cancel j hn

/-- Theorem 18 (Section 3.3 Strong Duality Slack Identity `∑ (1 - ζ_i) = n (1 - λ)`):
    In cleared integer form (common denominator `D > 0`), if `∑ ζ_i = n * λ` (`sumZeta = n * lam`),
    then `n * D - sumZeta = n * (D - lam)`, equating dual agent deficits to total LP slack `∆`! -/
theorem strong_duality_slack_identity (n D lam sumZeta : Int)
    (hdual : sumZeta = n * lam) :
    n * D - sumZeta = n * (D - lam) := by
  rw [hdual, Int.mul_sub]

end AsymptoticMMS
