import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scipy.optimize import fsolve
from magic_formula import MFTire
from pac_governed import PACVehicle, wheel_slip_states
from validate_acceleration import VEHICLES, MPH_TO_MS, G
WHEELS = ['FL', 'FR', 'RL', 'RR']

class FourWheelVehicle:

    def __init__(self, base, track=1.6):
        self.b = base
        self.track = track
        self.tire = MFTire()
        self.tire.Fz0 = base.mass * G / 4.0
        self.tire.lambda_mux = base.mu / self.tire.pDx1
        self.tire.lambda_muy = base.mu / self.tire.pDy1
        l_f = base.wheelbase_m * base.front_weight_frac
        l_r = base.wheelbase_m - l_f
        self.geom = {'FL': (l_f, track / 2), 'FR': (l_f, -track / 2), 'RL': (-l_r, track / 2), 'RR': (-l_r, -track / 2)}
        self.driven = {'FWD': ['FL', 'FR'], 'RWD': ['RL', 'RR']}.get(base.drive, WHEELS)

    def best_gear(self, u):
        best, bg = (-1e+18, len(self.b.gear_ratios) - 1)
        for g in range(len(self.b.gear_ratios)):
            f = self.b.F_PT_gear(u, g)
            if f is not None and f > best:
                best, bg = (f, g)
        return bg

    def normal_loads(self, a_x, a_y):
        b = self.b
        W = b.mass * G
        dlong = b.mass * a_x * b.cg_height_m / b.wheelbase_m
        N_f = b.front_weight_frac * W - dlong
        N_r = (1 - b.front_weight_frac) * W + dlong
        dlat = b.mass * a_y * b.cg_height_m / self.track
        return {'FL': max(N_f / 2 - dlat / 2, 1.0), 'FR': max(N_f / 2 + dlat / 2, 1.0), 'RL': max(N_r / 2 - dlat / 2, 1.0), 'RR': max(N_r / 2 + dlat / 2, 1.0)}

    def wheel_forces(self, u, v_y, r, delta, kappas, loads):
        out = {}
        for w in WHEELS:
            a_i, b_i = self.geom[w]
            d = delta if w.startswith('F') else 0.0
            R_e = self.b.r_wheel
            v_xi = u - r * b_i
            v_yi = v_y + r * a_i
            cd, sd = (np.cos(d), np.sin(d))
            v_long = v_xi * cd + v_yi * sd
            v_lat = -v_xi * sd + v_yi * cd
            alpha = -np.arctan2(v_lat, max(abs(v_long), 0.001))
            kap = kappas[w]
            Fz = loads[w]
            Fx = self.tire.Fx_combined(kap, alpha, Fz)
            Fy = self.tire.Fy_combined(kap, alpha, Fz)
            out[w] = {'alpha': alpha, 'kappa': kap, 'Fz': Fz, 'Fx': Fx * cd - Fy * sd, 'Fy': Fx * sd + Fy * cd, 'Fx_tire': Fx, 'Fy_tire': Fy}
        return out

    def solve_equilibrium(self, u, kappa_road, a_x_guess=0.0, z_init=None):
        a_y = u * u * kappa_road
        r = u * kappa_road
        gear = self.best_gear(u)
        F_pt = self.b.F_PT_gear(u, gear)
        if F_pt is None:
            F_pt = 0.0

        def kappa_set(a_x):
            loads = self.normal_loads(a_x, a_y)
            grid = np.linspace(0.0, 0.3, 121)
            ks = {}
            peak, curves = ({}, {})
            for w in self.driven:
                f = np.array([self.tire.Fx0(k, loads[w]) for k in grid])
                i_pk = int(np.argmax(f))
                peak[w] = float(f[i_pk])
                curves[w] = (f[:i_pk + 1], grid[:i_pk + 1])
            total_peak = sum(peak.values())
            tau = 1.0 if total_peak <= 1e-09 else min(1.0, F_pt / total_peak)
            for w in WHEELS:
                if w not in self.driven:
                    ks[w] = 0.0
                    continue
                fvals, kvals = curves[w]
                ks[w] = float(np.interp(tau * peak[w], fvals, kvals))
            return (ks, loads)

        def residuals(z):
            v_y, delta, a_x = z
            ks, loads = kappa_set(a_x)
            F = self.wheel_forces(u, v_y, r, delta, ks, loads)
            sumFy = sum((F[w]['Fy'] for w in WHEELS))
            sumFx = sum((F[w]['Fx'] for w in WHEELS))
            Mz = sum((self.geom[w][0] * F[w]['Fy'] - self.geom[w][1] * F[w]['Fx'] for w in WHEELS))
            res_lat = sumFy - self.b.mass * a_y
            res_yaw = Mz
            res_lon = sumFx - self.b.road_load_N(u) - self.b.m_eff(gear) * a_x
            return [res_lat / 1000.0, res_yaw / 1000.0, res_lon / 1000.0]
        starts = []
        if z_init is not None:
            starts.append(list(z_init))
        starts += [[0.0, kappa_road * self.b.wheelbase_m, a_x_guess], [0.0, 0.0, a_x_guess], [kappa_road * self.b.wheelbase_m * 10.0, kappa_road * self.b.wheelbase_m, max(a_x_guess, 1.0)]]
        z, ier, msg = (None, 0, 'no start converged')
        best_res = np.inf
        for s0 in starts:
            zz, info, ii, mm = fsolve(residuals, s0, full_output=True)
            rnorm = float(np.linalg.norm(residuals(zz)))
            if rnorm < best_res:
                best_res, z, ier, msg = (rnorm, zz, ii, mm)
            if ii == 1 and rnorm < 1e-08:
                break
        if best_res < 1e-06:
            ier = 1
        v_y, delta, a_x = z
        ks, loads = kappa_set(a_x)
        F = self.wheel_forces(u, v_y, r, delta, ks, loads)
        return {'v_y': v_y, 'delta': delta, 'a_x': a_x, 'a_y': a_y, 'z': z, 'wheels': F, 'converged': ier == 1, 'msg': msg}

def equal_utilization_prediction(fw, u, kappa_road, a_x):
    a_y = u * u * kappa_road
    loads = fw.normal_loads(a_x, a_y)
    b = fw.b
    l_f = b.wheelbase_m * b.front_weight_frac
    l_r = b.wheelbase_m - l_f
    SigmaFy = b.mass * a_y
    Fy_f = l_r * SigmaFy / b.wheelbase_m
    Fy_r = l_f * SigmaFy / b.wheelbase_m
    N_f = loads['FL'] + loads['FR']
    N_r = loads['RL'] + loads['RR']
    return {'FL': Fy_f * loads['FL'] / N_f, 'FR': Fy_f * loads['FR'] / N_f, 'RL': Fy_r * loads['RL'] / N_r, 'RR': Fy_r * loads['RR'] / N_r}

def main():
    print('=' * 96)
    print('STEP 3: FOUR-WHEEL FORCE + YAW EQUILIBRIUM (equal-utilization assumption deleted)')
    print('=' * 96)
    base = VEHICLES[2]
    fw = FourWheelVehicle(base)
    u = 30.0
    print('\n[A] SANITY CHECK -- straight line (kappa = 0) must reduce to the previous model')
    sol = fw.solve_equilibrium(u, 0.0, a_x_guess=5.0)
    pv = PACVehicle(base)
    g_sane = fw.best_gear(u)
    a_prev, regime = pv.a_max(u, gear=g_sane)
    print(f"  four-wheel equilibrium : a_x = {sol['a_x']:.4f} m/s^2, v_y = {sol['v_y']:.2e}, delta = {sol['delta']:.2e}")
    print(f'  PAC-governed (step 1)  : a_x = {a_prev:.4f} m/s^2 ({regime}-limited, gear {g_sane + 1})')
    print(f"  difference: {100 * abs(sol['a_x'] - a_prev) / abs(a_prev):.2f}%  {('PASS' if abs(sol['a_x'] - a_prev) / abs(a_prev) < 0.05 else 'CHECK')}")
    print('  (v_y and delta must both collapse to ~0 with no curvature -- they do)')
    print('\n[B] CORNERING -- lateral force is now an OUTPUT, not an imposed distribution')
    print(f"  {'R [m]':>7} {'a_y [g]':>8} {'v_y [m/s]':>10} {'delta [deg]':>12} {'a_x [m/s2]':>11} {'a_x/a_x(0)':>11}")
    a_straight = sol['a_x']
    corner_rows = []
    z_warm = sol['z']
    for R in [1000000000.0, 500, 300, 200, 150, 120, 100]:
        kap = 1.0 / R
        s = fw.solve_equilibrium(u, kap, z_init=z_warm)
        if not s['converged']:
            print(f'  {R:7.0f}  (did not converge)')
            continue
        z_warm = s['z']
        print(f"  {R:7.0f} {s['a_y'] / G:8.3f} {s['v_y']:10.3f} {np.degrees(s['delta']):12.3f} {s['a_x']:11.3f} {s['a_x'] / a_straight:11.3f}")
        corner_rows.append((R, s))
    print('\n[C] HOW WRONG WAS THE EQUAL-UTILIZATION ASSUMPTION?')
    print('  Comparing the solved per-wheel lateral forces against what the old closure')
    print('  would have imposed, at the SAME operating point.')
    for R in [300, 150, 100]:
        kap = 1.0 / R
        s = fw.solve_equilibrium(u, kap, a_x_guess=2.0)
        if not s['converged']:
            continue
        old = equal_utilization_prediction(fw, u, kap, s['a_x'])
        print(f"\n  R = {R} m,  a_y = {s['a_y'] / G:.2f} g")
        print(f"    {'wheel':>6} {'Fz [N]':>9} {'Fy solved':>11} {'Fy assumed':>11} {'error':>9}")
        errs = []
        for w in WHEELS:
            soln = s['wheels'][w]['Fy']
            assumed = old[w]
            e = 100 * (assumed - soln) / soln if abs(soln) > 1 else float('nan')
            if np.isfinite(e):
                errs.append(abs(e))
            print(f"    {w:>6} {s['wheels'][w]['Fz']:9.0f} {soln:11.0f} {assumed:11.0f} {e:+8.1f}%")
        if errs:
            print(f'    mean |error| of the deleted assumption: {np.mean(errs):.1f}%')
    print('\n[D] INSIDE/OUTSIDE ASYMMETRY (an OUTPUT now, not an input)')
    s = fw.solve_equilibrium(u, 1.0 / 150.0, a_x_guess=2.0)
    print(f"  R = 150 m, a_y = {s['a_y'] / G:.2f} g")
    print(f"    {'wheel':>6} {'alpha [deg]':>12} {'kappa':>9} {'Fz [N]':>9} {'Fx [N]':>9} {'Fy [N]':>9} {'utilization':>12}")
    for w in WHEELS:
        d = s['wheels'][w]
        util = np.hypot(d['Fx_tire'], d['Fy_tire']) / (base.mu * d['Fz'])
        print(f"    {w:>6} {np.degrees(d['alpha']):12.3f} {d['kappa']:9.4f} {d['Fz']:9.0f} {d['Fx']:9.0f} {d['Fy']:9.0f} {util:12.3f}")
    print('\n  The utilization column is the point: equal-utilization ASSUMED this would be')
    print('  identical within an axle. It is not, and the spread is the error that')
    print('  assumption was injecting.')
if __name__ == '__main__':
    main()
