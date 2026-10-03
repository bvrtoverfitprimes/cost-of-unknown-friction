import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from magic_formula import MFTire
G = 9.80665

class Battery:

    def __init__(self, V_nom=360.0, R_int_BOL=0.138, R_int_EOL=0.341, capacity_Ah=66.2, soc=1.0, T_C=25.0, soh=1.0):
        self.V_nom = V_nom
        self.R_BOL, self.R_EOL = (R_int_BOL, R_int_EOL)
        self.capacity_Ah = capacity_Ah
        self.soc = soc
        self.T_C = T_C
        self.soh = soh

    def V_oc(self, soc=None):
        s = self.soc if soc is None else soc
        return 320.0 + 83.0 * np.clip(s, 0.0, 1.0) ** 0.85
    EA_OVER_K = 1651.0
    _R_T_MEASURED = {-6.4: 0.1835, 21.2: 0.1028, 37.6: 0.0914}

    def R_int(self, T_C=None):
        T = self.T_C if T_C is None else T_C
        R_age = self.R_BOL + (1 - self.soh) * (self.R_EOL - self.R_BOL)
        T_K = np.asarray(T, dtype=float) + 273.15
        return R_age * np.exp(self.EA_OVER_K * (1.0 / T_K - 1.0 / 298.15))

    def P_max_electrical(self, T_C=None, soc=None):
        return self.V_oc(soc) ** 2 / (4.0 * self.R_int(T_C))

    def solve_current(self, P_demand, T_C=None, soc=None):
        V, R = (self.V_oc(soc), self.R_int(T_C))
        disc = V * V - 4 * R * P_demand
        if disc < 0:
            return (None, None)
        I = (V - np.sqrt(disc)) / (2 * R)
        return (I, V - I * R)

    def step(self, dt, P_demand, T_C=None):
        I, V = self.solve_current(P_demand, T_C)
        if I is None:
            return (None, None)
        self.soc = max(0.0, self.soc - I * dt / 3600.0 / self.capacity_Ah)
        return (I, V)

class FuelTank:

    def __init__(self, mass_kg=50.0, bsfc_kg_per_kWh=0.3):
        self.mass = mass_kg
        self.bsfc = bsfc_kg_per_kWh

    def step(self, dt, P_W):
        burned = self.bsfc * (P_W / 1000.0) * (dt / 3600.0)
        self.mass = max(0.0, self.mass - burned)
        return burned

def physical_vs_realizable(tire, Fz, R_e, slip_margin=0.8):
    ks = np.linspace(0.0, 0.35, 701)
    fx = np.array([tire.Fx0(k, Fz) for k in ks])
    i = int(np.argmax(fx))
    k_peak, Fx_peak = (float(ks[i]), float(fx[i]))
    k_target = slip_margin * k_peak
    return (Fx_peak, float(tire.Fx0(k_target, Fz)), k_peak, k_target)

def main():
    print('=' * 92)
    print('STEP 7: POWERTRAIN AS A STATE + CONTROL LAYER')
    print('=' * 92)
    ok = []
    print('\n[1] MAXIMUM POWER TRANSFER  [EXACT]')
    print('  P(I) = (V_oc - I R)I is a downward parabola => I* = V_oc/(2R), P_max = V_oc^2/(4R)')
    b = Battery()
    V, R = (b.V_oc(1.0), b.R_int(25.0))
    I_star = V / (2 * R)
    P_analytic = V * V / (4 * R)
    Is = np.linspace(1, 2 * I_star, 20000)
    P_num = np.max((V - Is * R) * Is)
    rel = abs(P_num - P_analytic) / P_analytic
    print(f'\n  V_oc = {V:.1f} V, R_int = {R:.3f} ohm')
    print(f'  analytic P_max = {P_analytic / 1000:8.1f} kW at I* = {I_star:.0f} A')
    print(f'  numerical max  = {P_num / 1000:8.1f} kW')
    ok.append(rel < 1e-06)
    print(f"  agreement: {100 * rel:.2e}%  {('PASS' if ok[-1] else 'FAIL')}")
    print('\n[2] IS THE NISSAN LEAF BATTERY-LIMITED OR MOTOR-LIMITED?')
    print("  Rated motor power is 80 kW. Compare against the pack's electrical ceiling.")
    print(f"\n  {'condition':<28} {'R_int [ohm]':>12} {'P_max [kW]':>12} {'vs 80 kW motor':>18}")
    for label, soh, T in [('new pack, 25 C', 1.0, 25.0), ('new pack, 0 C', 1.0, 0.0), ('new pack, -20 C', 1.0, -20.0), ('end of life, 25 C', 0.0, 25.0), ('end of life, 0 C', 0.0, 0.0), ('end of life, -20 C', 0.0, -20.0)]:
        bb = Battery(soh=soh, T_C=T)
        Pm = bb.P_max_electrical()
        verdict = 'motor-limited' if Pm > 80000 else 'BATTERY-LIMITED'
        print(f'  {label:<28} {bb.R_int():12.3f} {Pm / 1000:12.1f} {verdict:>18}')
    bb_new = Battery(soh=1.0, T_C=25.0)
    ok.append(bb_new.P_max_electrical() > 80000)
    print(f'\n  A healthy warm pack has ~3x the ceiling the motor asks for, so the Leaf is')
    print(f'  motor-limited by design. Aged or cold, the battery becomes the binding')
    print(f'  constraint -- the SAME regime-switching structure as traction vs powertrain,')
    print(f"  now appearing inside the powertrain itself.  {('PASS' if ok[-1] else 'FAIL')}")
    print('\n[3] VOLTAGE SAG DELIVERING RATED 80 kW')
    print(f"  {'condition':<24} {'I [A]':>9} {'V_term [V]':>12} {'sag':>8} {'deliverable?':>14}")
    for label, soh, T in [('new, 25 C', 1.0, 25.0), ('new, 0 C', 1.0, 0.0), ('new, -20 C', 1.0, -20.0), ('EOL, 25 C', 0.0, 25.0), ('EOL, -20 C', 0.0, -20.0)]:
        bb = Battery(soh=soh, T_C=T)
        I, Vt = bb.solve_current(80000.0)
        if I is None:
            print(f"  {label:<24} {'--':>9} {'--':>12} {'--':>8} {'NO -- exceeds P_max':>14}")
            continue
        sag = 100 * (1 - Vt / bb.V_oc())
        print(f"  {label:<24} {I:9.0f} {Vt:12.1f} {sag:7.1f}% {'yes':>14}")
    print('\n[4] POWER FADE WITH STATE OF CHARGE  [EXACT via V_oc(SOC)]')
    print('  P_max scales with V_oc^2, so it falls quadratically as the pack empties.')
    print(f"\n  {'SOC':>6} {'V_oc [V]':>10} {'P_max [kW]':>12} {'vs full':>9}")
    bb = Battery()
    P_full = bb.P_max_electrical(soc=1.0)
    for s in [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]:
        Pm = bb.P_max_electrical(soc=s)
        print(f'  {s:6.1f} {bb.V_oc(s):10.1f} {Pm / 1000:12.1f} {100 * Pm / P_full:8.1f}%')
    print("  This is why an EV's acceleration degrades near empty even though the motor")
    print('  is unchanged -- and the model now produces that without any added rule.')
    print('\n[5] ICE: FUEL MASS IS A STATE, SO m_eff FALLS AS FUEL BURNS')
    tank = FuelTank(mass_kg=50.0)
    m_dry = 1520.0
    print(f"  {'time':>8} {'fuel [kg]':>11} {'total m [kg]':>13} {'a for 8 kN':>12}")
    for minutes in [0, 10, 20, 30, 45]:
        t = FuelTank(mass_kg=50.0)
        t.step(minutes * 60.0, 40000.0)
        m_tot = m_dry + t.mass
        print(f'  {minutes:6.0f}m {t.mass:11.2f} {m_tot:13.1f} {8000 / m_tot:11.3f}')
    print('  A 50 kg tank is ~3% of vehicle mass: small, but it is the same order as the')
    print('  rotational-inertia correction we were careful to include in Prop (Effective mass).')
    print("\n[6] PHYSICAL vs REALIZABLE a_max  (needs Step 6's instability threshold)")
    tire = MFTire()
    tire.Fz0 = 4000.0
    print('  Step 6: slip diverges above T* = R_e*Fx_peak, so a traction controller must')
    print('  target a slip setpoint strictly below the peak. What does that margin cost?')
    print(f"\n  {'slip margin':>12} {'kappa target':>13} {'Fx realizable':>14} {'vs physical':>12} {'stability':>11}")
    for margin in [1.0, 0.95, 0.9, 0.8, 0.7, 0.5]:
        Fx_p, Fx_r, k_pk, k_tg = physical_vs_realizable(tire, 4000.0, 0.33, margin)
        stab = 'AT LIMIT' if margin >= 1.0 else 'stable'
        print(f'  {margin:12.2f} {k_tg:13.4f} {Fx_r:14.0f} {100 * Fx_r / Fx_p:11.1f}% {stab:>11}')
    Fx_p, Fx_r, _, _ = physical_vs_realizable(tire, 4000.0, 0.33, 0.8)
    cost = 100 * (1 - Fx_r / Fx_p)
    ok.append(cost < 10.0)
    print(f'\n  A 20% slip margin costs only {cost:.1f}% of longitudinal force, because the')
    print('  Magic Formula curve is FLAT near its peak. That flatness is precisely why')
    print('  traction control is cheap in performance terms and why real vehicles run one.')
    print(f"  {('PASS' if ok[-1] else 'FAIL')} (margin cost should be small)")
    print('\n  a_max^physical is an upper bound no controller attains; a_max^realizable is')
    print('  what a real vehicle delivers. Reporting only the former overstates capability.')
    print('\n' + '=' * 92)
    print(f'VALIDATION: {sum(ok)}/{len(ok)} checks passed')
    return all(ok)
if __name__ == '__main__':
    main()
