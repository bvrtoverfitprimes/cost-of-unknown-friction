import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scipy.optimize import brentq, fsolve
from magic_formula import MFTire
from validate_acceleration import Vehicle, VEHICLES, MPH_TO_MS, LBF_TO_N, G

def wheel_slip_states(u, v_y, r, a_i, b_i, delta_i, omega_i, R_e):
    v_xi = u - r * b_i
    v_yi = v_y + r * a_i
    cd, sd = (np.cos(delta_i), np.sin(delta_i))
    v_long = v_xi * cd + v_yi * sd
    v_lat = -v_xi * sd + v_yi * cd
    if abs(v_long) < 1e-06:
        return (0.0, 0.0, v_long)
    alpha = -np.arctan2(v_lat, abs(v_long))
    kappa = (omega_i * R_e - v_long) / abs(v_long)
    return (alpha, kappa, v_long)

class PACVehicle:

    def __init__(self, base: Vehicle, mu_scale_from_base=True):
        self.b = base
        self.tire = MFTire()
        self.tire.Fz0 = base.mass * G / 4.0
        if mu_scale_from_base:
            self.tire.lambda_mux = base.mu / self.tire.pDx1
            self.tire.lambda_muy = base.mu / self.tire.pDy1

    def normal_loads(self, a_x, a_y=0.0):
        b = self.b
        W = b.mass * G
        transfer_long = b.mass * a_x * b.cg_height_m / b.wheelbase_m
        N_f = b.front_weight_frac * W - transfer_long
        N_r = (1 - b.front_weight_frac) * W + transfer_long
        track = 0.85 * b.wheelbase_m / 1.6
        dN = b.mass * a_y * b.cg_height_m / track if a_y else 0.0
        return {'FL': max(N_f / 2 - dN / 2, 1.0), 'FR': max(N_f / 2 + dN / 2, 1.0), 'RL': max(N_r / 2 - dN / 2, 1.0), 'RR': max(N_r / 2 + dN / 2, 1.0)}

    def driven_wheels(self):
        return {'FWD': ['FL', 'FR'], 'RWD': ['RL', 'RR']}.get(self.b.drive, ['FL', 'FR', 'RL', 'RR'])

    def _build_peak_table(self, Fz_max=30000.0, n=241):
        self._Fz_grid = np.linspace(1.0, Fz_max, n)
        ks = np.linspace(0.0, 0.35, 141)
        self._Fx_grid = np.array([max((self.tire.Fx0(k, Fz) for k in ks)) for Fz in self._Fz_grid])

    def peak_wheel_force(self, Fz, alpha=0.0):
        if alpha == 0.0:
            if not hasattr(self, '_Fz_grid'):
                self._build_peak_table()
            return float(np.interp(Fz, self._Fz_grid, self._Fx_grid))
        ks = np.linspace(0.0, 0.35, 141)
        return max((self.tire.Fx_combined(k, alpha, Fz) for k in ks))

    def a_max(self, v, gear=0, a_y=0.0, alphas=None):
        b = self.b
        F_res = b.road_load_N(v)
        F_pt = b.F_PT_gear(v, gear)
        if F_pt is None:
            F_pt = 0.0
        m_eff = b.m_eff(gear)
        driven = self.driven_wheels()
        alphas = alphas or {w: 0.0 for w in ['FL', 'FR', 'RL', 'RR']}

        def traction(a_x):
            N = self.normal_loads(a_x, a_y)
            return sum((self.peak_wheel_force(N[w], alphas[w]) for w in driven))

        def Psi(a_x):
            return m_eff * a_x - min(F_pt, traction(a_x)) + F_res
        try:
            a = brentq(Psi, -50.0, 50.0, xtol=1e-09)
        except ValueError:
            a = (F_pt - F_res) / m_eff
        regime = 'powertrain' if F_pt <= traction(a) else 'traction'
        return (a, regime)

    def simulate_0_60(self):
        b = self.b
        v_target, t, v, gear, dt = (60.0 * MPH_TO_MS, 0.0, 0.0, 0, 0.001)
        w_red = b.redline_rpm * 2 * np.pi / 60.0
        dist, t_ro, frac = (0.0, None, [])
        while v < v_target and t < 300.0:
            omega = b.total_ratio(gear) * v / b.r_wheel
            if omega >= w_red and gear < len(b.gear_ratios) - 1:
                gear += 1
                for _ in range(int(b.shift_time_s / dt)):
                    v = max(v - b.road_load_N(v) / b.m_eff(gear) * dt, 0.0)
                    t += dt
                continue
            a, regime = self.a_max(v, gear)
            frac.append(1.0 if regime == 'traction' else 0.0)
            v += max(a, 1e-06) * dt
            dist += v * dt
            t += dt
            if t_ro is None and dist >= 0.3048:
                t_ro = t
        return (t, t - (t_ro or 0.0), float(np.mean(frac)) if frac else 0.0)

def main():
    print('=' * 92)
    print('STEP 1: PAC2002 AS THE GOVERNING TIRE LAW (ellipse removed from the backbone)')
    print('=' * 92)
    print('The tire force now comes from the Magic Formula surface, not from mu*Fz.')
    print('Load sensitivity is intrinsic to PAC2002 (pDx2 < 0); the separate power law is')
    print('NOT applied on top -- that would double-count (verified: up to 7% error).')
    print()
    print(f"  {'vehicle':<38} {'ellipse':>9} {'PAC2002':>9} {'published':>10} {'MF err':>8} {'trac%':>7}")
    ellipse_ref = {'2013 Nissan Leaf (BEV, single-speed)': 10.16, '2013 Honda Civic 1.8 (FWD, 5MT)': 9.55, '2013 Corvette ZR1 (RWD, 638 hp)': 3.3}
    rows = []
    for base in VEHICLES:
        pv = PACVehicle(base)
        t60, t60_ro, frac_tr = pv.simulate_0_60()
        ell = ellipse_ref[base.name]
        err = 100 * (t60_ro - base.target_0_60_s) / base.target_0_60_s
        print(f'  {base.name:<38} {ell:9.2f} {t60_ro:9.2f} {base.target_0_60_s:10.2f} {err:+7.1f}% {100 * frac_tr:6.0f}%')
        rows.append((base.name, ell, t60_ro, base.target_0_60_s, err))
    errs = [abs(r[4]) for r in rows]
    print()
    print(f'  PAC2002-governed mean |error| = {np.mean(errs):.1f}%  (ellipse-governed was 1.1%)')
    print()
    print('-' * 92)
    print("WHY THE NUMBERS MOVE: MF peak force vs the ellipse's mu*Fz, per wheel")
    print('  (the ellipse assumes force scales linearly with load; MF does not)')
    zr1 = PACVehicle(VEHICLES[2])
    print(f"  {'Fz [N]':>8} {'ellipse mu*Fz':>14} {'MF peak':>10} {'diff':>8}")
    for Fz in [2000, 4000, 6000, 8000, 10000]:
        ell = VEHICLES[2].mu * Fz
        mf = zr1.peak_wheel_force(Fz)
        print(f'  {Fz:8.0f} {ell:14.0f} {mf:10.0f} {100 * (mf / ell - 1):+7.1f}%')
    print()
    print('  At the nominal (static) load the two agree by construction. Away from it they')
    print('  diverge, and a hard-launching RWD car operates far from nominal on the rear axle')
    print('  -- which is exactly where the ellipse was least trustworthy.')
    print()
    print('-' * 92)
    print("STEP 2: EXACT WHEEL SLIP KINEMATICS (replaces 'tell the tire what force to make')")
    b = VEHICLES[2]
    u, R_e = (25.0, b.r_wheel)
    l_f = b.wheelbase_m * b.front_weight_frac
    l_r = b.wheelbase_m - l_f
    track = 1.6
    geom = {'FL': (l_f, track / 2), 'FR': (l_f, -track / 2), 'RL': (-l_r, track / 2), 'RR': (-l_r, -track / 2)}
    print(f'  vehicle at u={u} m/s, steer=2 deg, yaw rate r=0.15 rad/s, sideslip v_y=-0.3 m/s')
    print(f"  {'wheel':>6} {'a_i':>7} {'b_i':>7} {'alpha [deg]':>12} {'kappa':>9} {'v_long':>8}")
    for w, (a_i, b_i) in geom.items():
        d = np.radians(2.0) if w.startswith('F') else 0.0
        omega = u / R_e * 1.02
        al, ka, vl = wheel_slip_states(u, -0.3, 0.15, a_i, b_i, d, omega, R_e)
        print(f'  {w:>6} {a_i:7.2f} {b_i:7.2f} {np.degrees(al):12.3f} {ka:9.4f} {vl:8.2f}')
    print()
    print('  Note the inside/outside asymmetry in both alpha and v_long: it arises purely from')
    print('  the r*b_i and r*a_i terms, with no distributional assumption imposed. This is what')
    print('  makes the equal-utilization closure unnecessary (Step 3).')
if __name__ == '__main__':
    main()
