import Mathlib.Analysis.Convex.SpecificFunctions.Basic
import Mathlib.Analysis.Convex.SpecificFunctions.Pow
import Mathlib.Analysis.Convex.Deriv
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic

namespace BigBertha

open Real Set

theorem traction_fixed_point
    (meff mu W k Fres a : ℝ) (hden : meff - mu * k ≠ 0) :
    meff * a = mu * (W + k * a) - Fres ↔ a = (mu * W - Fres) / (meff - mu * k) := by
  rw [eq_div_iff hden]
  constructor
  · intro h; nlinarith [h]
  · intro h; nlinarith [h]

theorem traction_denominator_pos_of
    (meff mu k : ℝ) (hmeff : 0 < meff) (hmu : 0 ≤ mu) (hk : 0 ≤ k)
    (h : mu * k < meff) : 0 < meff - mu * k := by linarith

theorem open_diff_le_independent (F₁ F₂ : ℝ) :
    2 * min F₁ F₂ ≤ F₁ + F₂ := by
  rcases le_total F₁ F₂ with h | h
  · rw [min_eq_left h]; linarith
  · rw [min_eq_right h]; linarith

theorem open_diff_eq_iff (F₁ F₂ : ℝ) :
    2 * min F₁ F₂ = F₁ + F₂ ↔ F₁ = F₂ := by
  rcases le_total F₁ F₂ with h | h
  · rw [min_eq_left h]; constructor <;> intro hh <;> linarith
  · rw [min_eq_right h]; constructor <;> intro hh <;> linarith

theorem load_transfer_strictly_destroys_grip
    (c p N Δ : ℝ) (hc : 0 < c) (hp₀ : 0 < p) (hp₁ : p < 1)
    (hN : 0 < N) (hΔ : Δ ≠ 0) (hlo : 0 < N / 2 - Δ) (hhi : 0 < N / 2 + Δ) :
    c * (N / 2 - Δ) ^ p + c * (N / 2 + Δ) ^ p < 2 * (c * (N / 2) ^ p) := by
  have hconc : StrictConcaveOn ℝ (Ici (0 : ℝ)) fun x : ℝ => x ^ p :=
    Real.strictConcaveOn_rpow hp₀ hp₁
  have hxmem : (N / 2 - Δ) ∈ Ici (0 : ℝ) := le_of_lt hlo
  have hymem : (N / 2 + Δ) ∈ Ici (0 : ℝ) := le_of_lt hhi
  have hne : (N / 2 - Δ) ≠ (N / 2 + Δ) := by
    intro hcon
    apply hΔ
    linarith [hcon]
  have key := hconc.2 hxmem hymem hne (by norm_num : (0:ℝ) < 1/2)
      (by norm_num : (0:ℝ) < 1/2) (by norm_num : (1:ℝ)/2 + 1/2 = 1)
  simp only [smul_eq_mul] at key
  have hmid : (1:ℝ)/2 * (N/2 - Δ) + 1/2 * (N/2 + Δ) = N/2 := by ring
  rw [hmid] at key
  nlinarith [key, hc]

theorem load_transfer_destroys_grip_general
    (f : ℝ → ℝ) (s : Set ℝ) (hconv : Convex ℝ s)
    (hconc : StrictConcaveOn ℝ s f)
    (N Δ : ℝ) (hΔ : Δ ≠ 0)
    (hlo : (N / 2 - Δ) ∈ s) (hhi : (N / 2 + Δ) ∈ s) :
    f (N / 2 - Δ) + f (N / 2 + Δ) < 2 * f (N / 2) := by
  have hne : (N / 2 - Δ) ≠ (N / 2 + Δ) := by
    intro hcon; apply hΔ; linarith [hcon]
  have key := hconc.2 hlo hhi hne (by norm_num : (0:ℝ) < 1/2)
      (by norm_num : (0:ℝ) < 1/2) (by norm_num : (1:ℝ)/2 + 1/2 = 1)
  simp only [smul_eq_mul] at key
  have hmid : (1:ℝ)/2 * (N/2 - Δ) + 1/2 * (N/2 + Δ) = N/2 := by ring
  rw [hmid] at key
  linarith [key]

theorem pac2002_peak_force_strictly_concave
    (pDx1 d Fz0 : ℝ) (hd : 0 < d) (hFz0 : 0 < Fz0) :
    StrictConcaveOn ℝ Set.univ
      (fun Fz : ℝ => (pDx1 - d * ((Fz - Fz0) / Fz0)) * Fz) := by
  have hFz0' : Fz0 ≠ 0 := ne_of_gt hFz0
  have hrw : (fun Fz : ℝ => (pDx1 - d * ((Fz - Fz0) / Fz0)) * Fz)
           = fun Fz : ℝ => (-(d / Fz0)) * Fz ^ 2 + (pDx1 + d) * Fz := by
    funext Fz; field_simp; ring
  rw [hrw]
  have ha : -(d / Fz0) < 0 := by
    have : 0 < d / Fz0 := div_pos hd hFz0
    linarith
  refine ⟨convex_univ, ?_⟩
  intro x _ y _ hxy s t hs ht hst
  simp only [smul_eq_mul]
  have hne : x - y ≠ 0 := sub_ne_zero.mpr hxy
  have hsq : 0 < (x - y) ^ 2 := by positivity
  have hdF : 0 < d / Fz0 := div_pos hd hFz0
  have ht1 : t = 1 - s := by linarith
  subst ht1

  have gap :
      (-(d / Fz0)) * (s * x + (1 - s) * y) ^ 2 + (pDx1 + d) * (s * x + (1 - s) * y)
        - (s * ((-(d / Fz0)) * x ^ 2 + (pDx1 + d) * x)
           + (1 - s) * ((-(d / Fz0)) * y ^ 2 + (pDx1 + d) * y))
      = (d / Fz0) * (s * (1 - s) * (x - y) ^ 2) := by ring
  have hpos : 0 < (d / Fz0) * (s * (1 - s) * (x - y) ^ 2) := by
    apply mul_pos hdF
    apply mul_pos (mul_pos hs ht) hsq |>.trans_le
    apply le_of_eq; ring
  linarith [gap, hpos]

theorem ellipse_boundary
    (mux muy Fz Fy : ℝ) (hFz : 0 < Fz) (hmux : 0 < mux) (hmuy : 0 < muy)
    (hfeas : |Fy| ≤ muy * Fz) :
    ((mux * Fz * Real.sqrt (1 - (Fy / (muy * Fz)) ^ 2)) / (mux * Fz)) ^ 2
      + (Fy / (muy * Fz)) ^ 2 = 1 := by
  have hmuFz : (0:ℝ) < mux * Fz := mul_pos hmux hFz
  have hmuyFz : (0:ℝ) < muy * Fz := mul_pos hmuy hFz
  have hb := abs_le.mp hfeas
  have hratio : (Fy / (muy * Fz)) ^ 2 ≤ 1 := by
    rw [div_pow, div_le_one (by positivity)]
    nlinarith [hb.1, hb.2, hmuyFz]
  have hnonneg : (0:ℝ) ≤ 1 - (Fy / (muy * Fz)) ^ 2 := by linarith
  rw [mul_div_cancel_left₀ _ (ne_of_gt hmuFz), Real.sq_sqrt hnonneg]
  ring

theorem power_transfer_le (V R I : Real) (hR : 0 < R) :
    (V - I * R) * I <= V ^ 2 / (4 * R) := by
  have key : V ^ 2 / (4 * R) - (V - I * R) * I = R * (I - V / (2 * R)) ^ 2 := by
    field_simp
    ring
  nlinarith [sq_nonneg (I - V / (2 * R)), hR, key]

theorem power_transfer_eq_at_opt (V R : Real) (hR : 0 < R) :
    (V - (V / (2 * R)) * R) * (V / (2 * R)) = V ^ 2 / (4 * R) := by
  field_simp
  ring

theorem steady_pitch_load_transfer
    (F_f F_r l_f l_r m_s a h_s : Real)
    (hL : l_f + l_r ≠ 0)
    (heave : F_f + F_r = 0)
    (pitch : l_f * F_f - l_r * F_r + m_s * a * h_s = 0) :
    F_r = (m_s * a * h_s) / (l_f + l_r) := by
  have hf : F_f = -F_r := by linarith
  rw [hf] at pitch
  rw [eq_div_iff hL]
  nlinarith [pitch]

theorem sprung_plus_unsprung_eq_total
    (m_s m_u h_s h_u : Real) (hm : m_s + m_u ≠ 0) :
    m_s * h_s + m_u * h_u
      = (m_s + m_u) * ((m_s * h_s + m_u * h_u) / (m_s + m_u)) := by
  rw [mul_div_cancel₀ _ hm]

theorem closure_matches_quasistatic
    (m_s m_u h_s h_u a L : Real) (hL : L ≠ 0) (hm : m_s + m_u ≠ 0) :
    (m_s * a * h_s) / L + (m_u * a * h_u) / L
      = ((m_s + m_u) * ((m_s * h_s + m_u * h_u) / (m_s + m_u))) * a / L := by
  rw [mul_div_cancel₀ _ hm]
  field_simp

theorem traction_limited_blind_to_powertrain
    (F_PT F_TR delta : Real) (hgap : F_TR < F_PT) (hd : |delta| < F_PT - F_TR) :
    min (F_PT + delta) F_TR = min F_PT F_TR := by
  have h1 : F_TR < F_PT + delta := by
    have := abs_lt.mp hd
    linarith [this.1]
  rw [min_eq_right (le_of_lt h1), min_eq_right (le_of_lt hgap)]

theorem powertrain_limited_blind_to_traction
    (F_PT F_TR delta : Real) (hgap : F_PT < F_TR) (hd : |delta| < F_TR - F_PT) :
    min F_PT (F_TR + delta) = min F_PT F_TR := by
  have h1 : F_PT < F_TR + delta := by
    have := abs_lt.mp hd
    linarith [this.1]
  rw [min_eq_left (le_of_lt h1), min_eq_left (le_of_lt hgap)]

theorem blind_to_powertrain_iff_traction_limited (F_PT F_TR : Real) :
    (∀ delta : Real, 0 < delta → min (F_PT + delta) F_TR = min F_PT F_TR)
      ↔ F_TR ≤ F_PT := by
  constructor
  · intro h
    by_contra hcon
    push_neg at hcon
    have hd := h (F_TR - F_PT) (by linarith)
    rw [min_eq_left (le_of_lt hcon)] at hd
    have : F_PT + (F_TR - F_PT) = F_TR := by ring
    rw [this, min_self] at hd
    linarith
  · intro h delta hdelta
    rw [min_eq_right h, min_eq_right (by linarith : F_TR ≤ F_PT + delta)]

theorem transition_is_one_sided (F delta : Real) (hdelta : 0 < delta) :
    min (F + delta) F = F ∧ min (F - delta) F = F - delta := by
  constructor
  · exact min_eq_right (by linarith)
  · exact min_eq_left (by linarith)

theorem envelope_root_unique (Φ : Real → Real) (c : Real) (hc : 0 < c)
    (hΦ : ∀ a b, a < b → Φ b - Φ a ≤ -c * (b - a))
    (a₁ a₂ : Real) (h₁ : Φ a₁ = 0) (h₂ : Φ a₂ = 0) : a₁ = a₂ := by
  rcases lt_trichotomy a₁ a₂ with h | h | h
  · have := hΦ a₁ a₂ h
    rw [h₁, h₂] at this
    nlinarith
  · exact h
  · have := hΦ a₂ a₁ h
    rw [h₁, h₂] at this
    nlinarith

theorem envelope_feasible_initial_segment (Φ : Real → Real) (c : Real) (hc : 0 < c)
    (hΦ : ∀ a b, a < b → Φ b - Φ a ≤ -c * (b - a))
    (a b : Real) (hab : a ≤ b) (hb : 0 ≤ Φ b) : 0 ≤ Φ a := by
  rcases eq_or_lt_of_le hab with h | h
  · rw [h]; exact hb
  · have := hΦ a b h
    nlinarith

theorem envelope_root_lipschitz (Φ₁ Φ₂ : Real → Real) (c ε : Real) (hc : 0 < c)
    (hΦ : ∀ a b, a < b → Φ₁ b - Φ₁ a ≤ -c * (b - a))
    (hclose : ∀ x, |Φ₁ x - Φ₂ x| ≤ ε)
    (a₁ a₂ : Real) (h₁ : Φ₁ a₁ = 0) (h₂ : Φ₂ a₂ = 0) : |a₁ - a₂| ≤ ε / c := by
  rw [le_div_iff₀ hc]
  have hx := abs_le.mp (hclose a₂)
  rcases lt_trichotomy a₁ a₂ with h | h | h
  · have := hΦ a₁ a₂ h
    rw [abs_of_neg (by linarith), h₁] at *
    nlinarith [hx.1, hx.2]
  · rw [h, sub_self, abs_zero]
    nlinarith [abs_nonneg (Φ₁ a₂ - Φ₂ a₂), hclose a₂]
  · have := hΦ a₂ a₁ h
    rw [abs_of_pos (by linarith), h₁] at *
    nlinarith [hx.1, hx.2]

theorem envelope_comparison (Φ₁ Φ₂ : Real → Real)
    (hdom : ∀ a, Φ₁ a ≤ Φ₂ a)
    (a₁ a₂ : Real) (h₁ : 0 ≤ Φ₁ a₁) (hmax₂ : ∀ a, 0 ≤ Φ₂ a → a ≤ a₂) : a₁ ≤ a₂ :=
  hmax₂ a₁ (le_trans h₁ (hdom a₁))

theorem mass_margin_identity (T Λ F_aero m r J a : Real)
    (hroot : T - (F_aero + m * r) - (m + J) * a = 0) :
    (T - Λ) - m * r - m * a = J * a + F_aero - Λ := by
  linarith

theorem load_sensitivity_deficit_nonneg (φ : Real → Real) (N d : Real)
    (hsuper : ∀ x, φ x ≤ φ N + d * (x - N)) (hzero : φ 0 = 0) :
    0 ≤ φ N - d * N := by
  have := hsuper 0
  rw [hzero] at this
  linarith

theorem grip_lower_bound (F N μ : Real) (hN : 0 < N) (hold : F ≤ μ * N) : F / N ≤ μ := by
  rw [div_le_iff₀ hN]; linarith

theorem grip_upper_bound_from_slip (F N μ : Real) (hN : 0 < N) (slip : μ * N < F) :
    μ < F / N := by
  rw [lt_div_iff₀ hN]; linarith

theorem grip_set_upward_closed {ι : Type*} (F N : ι → Real) (hN : ∀ t, 0 ≤ N t)
    (μ μ' : Real) (hμ : ∀ t, F t ≤ μ * N t) (hle : μ ≤ μ') : ∀ t, F t ≤ μ' * N t := by
  intro t
  have := hμ t
  nlinarith [hN t]

theorem grip_not_identified_without_binding {ι : Type*} (F N : ι → Real)
    (hN : ∀ t, 0 ≤ N t) (μ : Real) (hμ : ∀ t, F t ≤ μ * N t) :
    ∃ μ', μ' ≠ μ ∧ ∀ t, F t ≤ μ' * N t :=
  ⟨μ + 1, by linarith, grip_set_upward_closed F N hN μ (μ + 1) hμ (by linarith)⟩

theorem grip_pinned_by_binding (F N μ₁ μ₂ : Real) (hN : 0 < N)
    (h₁ : F = μ₁ * N) (h₂ : F = μ₂ * N) : μ₁ = μ₂ := by
  have : (μ₁ - μ₂) * N = 0 := by linarith
  rcases mul_eq_zero.mp this with h | h
  · linarith
  · linarith

theorem road_load_confounding (A₁ B₁ C₁ w₁ G₁ A₂ B₂ C₂ w₂ G₂ v : Real)
    (hC : C₁ = C₂) (hB : B₁ + 2 * C₁ * w₁ = B₂ + 2 * C₂ * w₂)
    (hA : A₁ + C₁ * w₁ ^ 2 + G₁ = A₂ + C₂ * w₂ ^ 2 + G₂) :
    A₁ + B₁ * v + C₁ * (v + w₁) ^ 2 + G₁ = A₂ + B₂ * v + C₂ * (v + w₂) ^ 2 + G₂ := by
  linear_combination hA + v * hB + v ^ 2 * hC

theorem box_vertex_bounds (f : Real → Real → Real)
    (h₁ : ∀ x x' y, x ≤ x' → f x y ≤ f x' y) (h₂ : ∀ x y y', y ≤ y' → f x y ≤ f x y')
    (x y xl xh yl yh : Real) (hx : xl ≤ x ∧ x ≤ xh) (hy : yl ≤ y ∧ y ≤ yh) :
    f xl yl ≤ f x y ∧ f x y ≤ f xh yh :=
  ⟨le_trans (h₁ xl x yl hx.1) (h₂ x yl y hy.1), le_trans (h₁ x xh y hx.2) (h₂ xh y yh hy.2)⟩

theorem monotone_preserves_order (g : Real → Real) (hg : Monotone g) (x x' : Real)
    (h : x ≤ x') : g x ≤ g x' := hg h

theorem forward_step_monotone (cap₁ cap₂ v₁ v₂ a₁ a₂ ds : Real)
    (hcap : cap₁ ≤ cap₂) (hv0 : 0 ≤ v₁) (hv : v₁ ≤ v₂) (ha : a₁ ≤ a₂) (hds : 0 ≤ ds) :
    min cap₁ (Real.sqrt (v₁ ^ 2 + 2 * a₁ * ds)) ≤ min cap₂ (Real.sqrt (v₂ ^ 2 + 2 * a₂ * ds)) := by
  apply min_le_min hcap
  apply Real.sqrt_le_sqrt
  nlinarith [mul_le_mul_of_nonneg_right ha hds]

theorem time_antitone_in_speed {n : Nat} (ds : Real) (hds : 0 ≤ ds) (v₁ v₂ : Fin n → Real)
    (hpos : ∀ i, 0 < v₁ i) (hle : ∀ i, v₁ i ≤ v₂ i) :
    (∑ i, ds / v₂ i) ≤ ∑ i, ds / v₁ i := by
  apply Finset.sum_le_sum
  intro i _
  exact div_le_div_of_nonneg_left hds (hpos i) (hle i)

theorem two_binding_samples_pin_load_line (K K' c c' a₁ a₂ : Real) (ha : a₁ ≠ a₂)
    (h₁ : K * (c - a₁) = K' * (c' - a₁)) (h₂ : K * (c - a₂) = K' * (c' - a₂)) :
    K = K' ∧ (K ≠ 0 → c = c') := by
  have hK : K = K' := by
    have h : (K - K') * (a₂ - a₁) = 0 := by linarith
    rcases mul_eq_zero.mp h with h | h
    · linarith
    · exact absurd (by linarith : a₁ = a₂) ha
  refine ⟨hK, fun hK0 => ?_⟩
  subst hK
  have : K * (c - c') = 0 := by linarith
  rcases mul_eq_zero.mp this with h | h
  · exact absurd h hK0
  · linarith

theorem brush_peak_sensitivity_identity (x : Real) :
    1 - (1 - x) ^ 3 - 3 * x * (1 - x) ^ 2 = x ^ 2 * (3 - 2 * x) := by
  ring

theorem brush_peak_sensitivity_nonneg (x : Real) (h0 : 0 ≤ x) (h1 : x ≤ 1) :
    0 ≤ x ^ 2 * (3 - 2 * x) := by
  have : 0 ≤ 3 - 2 * x := by linarith
  positivity

theorem chance_constrained_reduction (T : Real → Real) (hT : Antitone T)
    (t θ q : Real) (ht : T θ ≤ t) (hθ : θ ≤ q) : T q ≤ t :=
  le_trans (hT hθ) ht

theorem tolerated_set_upward_closed (feas : Real → Prop)
    (hmono : ∀ a b, a ≤ b → feas a → feas b) (μ μ' : Real) (h : feas μ) (hle : μ ≤ μ') :
    feas μ' :=
  hmono μ μ' hle h

theorem braking_excess_eq (v b : Real) (hv : 0 < v) (hb : 0 < b) :
    v / b - (v ^ 2 / (2 * b)) / v = v / (2 * b) := by
  field_simp
  ring

theorem spacing_optimum (c K s s' : Real) (hK : 0 < K) (hs : 0 < s) (hs' : 0 < s')
    (hopt : K * s' ^ 3 = 2 * c) : 3 / 2 * K * s' ≤ c / s ^ 2 + K * s := by
  have hs2 : 0 < s ^ 2 := by positivity
  rw [div_add' _ _ _ (ne_of_gt hs2), le_div_iff₀ hs2]
  nlinarith [sq_nonneg (s - s'), mul_pos hK hs, mul_pos hK hs', sq_nonneg s,
    mul_nonneg (mul_nonneg hK.le (sq_nonneg (s - s'))) (by linarith : (0:Real) ≤ 2 * s + s')]

theorem debt_rate_nonincreasing (adot a r τ : Real) (hadot : adot ≤ 0) (ha : 0 < a)
    (hr : r ≤ 1) (hτ : 0 < τ) : adot / a * (1 - r) - r / τ ≤ -(r / τ) := by
  have h1 : adot / a ≤ 0 := div_nonpos_of_nonpos_of_nonneg hadot ha.le
  have h2 : 0 ≤ 1 - r := by linarith
  nlinarith [mul_nonpos_of_nonpos_of_nonneg h1 h2]

theorem debt_rate_general (adot a r τ : Real) (ha : 0 < a) (hr0 : 0 ≤ r) (hr : r ≤ 1) :
    adot / a * (1 - r) - r / τ ≤ max (adot / a) 0 - r / τ := by
  have h2 : 0 ≤ 1 - r := by linarith
  have h3 : 1 - r ≤ 1 := by linarith
  rcases le_total 0 (adot / a) with h | h
  · rw [max_eq_left h]
    nlinarith [mul_le_mul_of_nonneg_left h3 h]
  · rw [max_eq_right h]
    nlinarith [mul_nonpos_of_nonpos_of_nonneg h h2]

theorem power_transfer_eq_iff (V R I : Real) (hR : 0 < R) :
    (V - I * R) * I = V ^ 2 / (4 * R) ↔ I = V / (2 * R) := by
  have key : V ^ 2 / (4 * R) - (V - I * R) * I = R * (I - V / (2 * R)) ^ 2 := by
    field_simp
    ring
  constructor
  · intro h
    have h0 : R * (I - V / (2 * R)) ^ 2 = 0 := by linarith
    rcases mul_eq_zero.mp h0 with h1 | h1
    · exact absurd h1 (ne_of_gt hR)
    · have := pow_eq_zero_iff (n := 2) (by norm_num) |>.mp h1
      linarith
  · intro h
    have h0 : R * (I - V / (2 * R)) ^ 2 = 0 := by rw [h, sub_self]; ring
    linarith

theorem box_vertex_bounds_n {n : Nat} (f : (Fin n → Real) → Real) (hf : Monotone f)
    (lo hi x : Fin n → Real) (hx : lo ≤ x ∧ x ≤ hi) : f lo ≤ f x ∧ f x ≤ f hi :=
  ⟨hf hx.1, hf hx.2⟩

theorem envelope_one_point (Φ : Real → Real) (c lo hi a₀ am : Real) (hc : 0 < c)
    (hΦ : ∀ a b, a < b → Φ b - Φ a ≤ -c * (b - a))
    (hmax : IsGreatest {a | lo ≤ a ∧ a ≤ hi ∧ 0 ≤ Φ a} am)
    (ha₀ : lo ≤ a₀ ∧ a₀ < hi) :
    (Φ a₀ = 0 → am = a₀) ∧ (Φ a₀ < 0 → am < a₀) ∧ (Continuous Φ → 0 < Φ a₀ → a₀ < am) := by
  obtain ⟨⟨_, _, ham⟩, hub⟩ := hmax
  refine ⟨fun h0 => ?_, fun hneg => ?_, fun hcont hpos => ?_⟩
  · have hle : a₀ ≤ am := hub ⟨ha₀.1, ha₀.2.le, h0.ge⟩
    rcases eq_or_lt_of_le hle with h | h
    · exact h.symm
    · have := hΦ a₀ am h
      nlinarith
  · by_contra hge
    push_neg at hge
    rcases eq_or_lt_of_le hge with h | h
    · rw [h] at hneg; linarith
    · have := hΦ a₀ am h
      nlinarith
  · have hev : ∀ᶠ x in nhds a₀, 0 < Φ x := hcont.continuousAt.eventually (lt_mem_nhds hpos)
    obtain ⟨ε, hε, hball⟩ := Metric.eventually_nhds_iff.mp hev
    set x := min (a₀ + ε / 2) hi with hxdef
    have hx1 : a₀ < x := lt_min (by linarith) ha₀.2
    have hx2 : x ≤ hi := min_le_right _ _
    have hdist : dist x a₀ < ε := by
      rw [Real.dist_eq, abs_of_pos (by linarith)]
      have : x ≤ a₀ + ε / 2 := min_le_left _ _
      linarith
    have hxpos : 0 < Φ x := hball hdist
    have hxle : x ≤ am := hub ⟨by linarith [ha₀.1], hx2, hxpos.le⟩
    linarith

end BigBertha

#print axioms BigBertha.traction_limited_blind_to_powertrain
#print axioms BigBertha.powertrain_limited_blind_to_traction
#print axioms BigBertha.blind_to_powertrain_iff_traction_limited
#print axioms BigBertha.transition_is_one_sided
#print axioms BigBertha.traction_fixed_point
#print axioms BigBertha.traction_denominator_pos_of
#print axioms BigBertha.sprung_plus_unsprung_eq_total
#print axioms BigBertha.open_diff_le_independent
#print axioms BigBertha.open_diff_eq_iff
#print axioms BigBertha.load_transfer_strictly_destroys_grip
#print axioms BigBertha.ellipse_boundary
#print axioms BigBertha.load_transfer_destroys_grip_general
#print axioms BigBertha.pac2002_peak_force_strictly_concave
#print axioms BigBertha.power_transfer_le
#print axioms BigBertha.power_transfer_eq_at_opt
#print axioms BigBertha.steady_pitch_load_transfer
#print axioms BigBertha.closure_matches_quasistatic
#print axioms BigBertha.envelope_root_unique
#print axioms BigBertha.envelope_feasible_initial_segment
#print axioms BigBertha.envelope_root_lipschitz
#print axioms BigBertha.envelope_comparison
#print axioms BigBertha.mass_margin_identity
#print axioms BigBertha.load_sensitivity_deficit_nonneg
#print axioms BigBertha.grip_lower_bound
#print axioms BigBertha.grip_upper_bound_from_slip
#print axioms BigBertha.grip_set_upward_closed
#print axioms BigBertha.grip_not_identified_without_binding
#print axioms BigBertha.grip_pinned_by_binding
#print axioms BigBertha.road_load_confounding
#print axioms BigBertha.box_vertex_bounds
#print axioms BigBertha.monotone_preserves_order
#print axioms BigBertha.forward_step_monotone
#print axioms BigBertha.time_antitone_in_speed
#print axioms BigBertha.two_binding_samples_pin_load_line
#print axioms BigBertha.brush_peak_sensitivity_identity
#print axioms BigBertha.brush_peak_sensitivity_nonneg
#print axioms BigBertha.chance_constrained_reduction
#print axioms BigBertha.tolerated_set_upward_closed
#print axioms BigBertha.braking_excess_eq
#print axioms BigBertha.spacing_optimum
#print axioms BigBertha.debt_rate_nonincreasing
#print axioms BigBertha.debt_rate_general
#print axioms BigBertha.power_transfer_eq_iff
#print axioms BigBertha.box_vertex_bounds_n
#print axioms BigBertha.envelope_one_point
