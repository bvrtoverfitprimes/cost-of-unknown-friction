from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t4_uncertainty import fleet_mu_ratios
from t5_mintime import T_star
from u1_value_of_information import ROUTES, calibrated_prior, probe_info, probe_time_cost, route_family
from validate_acceleration import G, VEHICLES

class PriceCurve:

    def __init__(self, veh, kind):
        R = route_family(kind)
        L = R[0][-1]
        T1 = T_star(veh, 'ellipse', 1.0, None, R)[0]
        self.lr = np.linspace(np.log(0.55), 0.0, 46)
        self.p = np.array([(T_star(veh, 'ellipse', float(np.exp(x)), None, R)[0] - T1) / L for x in self.lr])
        self.kappa = float(-(self.p[-1] - self.p[-4]) / (self.lr[-1] - self.lr[-4]))
        pos = np.nonzero(self.p > 1e-07)[0]
        self.free_above = float(np.exp(self.lr[pos.max() + 1])) if len(pos) and pos.max() + 1 < len(self.lr) else 1.0

    def h(self, s, z):
        return np.interp(-z * np.asarray(s), self.lr, self.p)

def s_of_d(d, sigma, ell, s_p):
    rho = np.exp(-d / ell)
    return np.sqrt(np.maximum(sigma ** 2 - rho ** 2 * sigma ** 4 / (sigma ** 2 + s_p ** 2), 0.0))

def J(D, c_p, pc, z, sigma, ell, s_p, n=400):
    d = np.linspace(0.0, D, n + 1)
    h = pc.h(s_of_d(d, sigma, ell, s_p), z)
    return c_p / D + float(np.sum(0.5 * (h[1:] + h[:-1]) * np.diff(d))) / D

def D_star_asym(c_p, kappa, z, sigma, ell):
    if kappa <= 0:
        return np.inf
    return (3 * c_p * np.sqrt(ell) / (np.sqrt(2) * kappa * z * sigma)) ** (2 / 3)

def main() -> int:
    print('=' * 100)
    print('U1d -- PROBE SPACING UNDER SPATIALLY CHANGING FRICTION')
    print('=' * 100)
    ok = True
    sigma, _ = calibrated_prior(fleet_mu_ratios())
    z = norm.ppf(0.95)
    u_probe = float(np.exp(-z * sigma))
    s_p = 1.0 / np.sqrt(probe_info(u_probe))
    print(f'field sd sigma = {sigma:.3f} (fleet), probe residual sd s_p = {s_p:.4f}, z = {z:.3f}')
    record = {'sigma': sigma, 'z': float(z), 's_p': float(s_p), 'curves': [], 'optimum': [], 'threshold': [], 'scaling': []}
    print('\n[1] Price curve p(r) and its local slope kappa = -dp/dln r at r = 1')
    pcs = {}
    for veh in VEHICLES:
        for kind in ROUTES:
            pc = PriceCurve(veh, kind)
            pcs[veh.name, kind] = pc
            ok &= bool(np.all(np.diff(pc.p) <= 1e-12))
            print(f'    {veh.name[:26]:<26s} {kind:<9s} p(0.80) = {1000 * float(np.interp(np.log(0.8), pc.lr, pc.p)):6.3f} s/km   kappa = {1000 * pc.kappa:6.3f} s/km   price-free for r >= {pc.free_above:.2f}')
            record['curves'].append({'vehicle': veh.name, 'route': kind, 'p080': 1000 * float(np.interp(np.log(0.8), pc.lr, pc.p)), 'kappa': 1000 * float(pc.kappa), 'h0': 1000 * float(pc.h(sigma, z)), 'free_above': float(pc.free_above)})
    print('\n[2] Optimal spacing: numerical minimum of J(D) vs the asymptotic law')
    print(f"    {'vehicle':<10s} {'route':<9s} {'l [m]':>7s} {'c_p':>6s} {'D* num':>8s} {'D* law':>8s} {'D*/l':>6s} {'J* s/km':>8s} {'no-probe':>8s} {'probe?':>6s}")
    n_checked = 0
    for veh in VEHICLES:
        for kind in ROUTES:
            pc = pcs[veh.name, kind]
            vc = float(np.max(route_family(kind)[1]))
            c_p = probe_time_cost(vc, u_probe * veh.mu * G)
            h0 = float(pc.h(sigma, z))
            for ell in (1000.0, 5000.0, 25000.0):
                res = minimize_scalar(lambda D: J(D, c_p, pc, z, sigma, ell, s_p), bounds=(1.0, 20 * ell), method='bounded')
                Dn, Jn = (float(res.x), float(res.fun))
                worth = Jn < h0 * (1 - 1e-06) and Dn < 19 * ell
                Da = D_star_asym(c_p, pc.kappa, z, sigma, ell)
                if worth and Dn / ell < 0.12:
                    n_checked += 1
                    ok &= abs(Da - Dn) / Dn < 0.35
                Ds = f'{Dn:8.0f}' if worth else f"{'none':>8s}"
                rel = f'{Dn / ell:6.2f}' if worth else f"{'-':>6s}"
                record['optimum'].append({'vehicle': veh.name, 'route': kind, 'ell': ell, 'c_p': c_p, 'D_num': Dn if worth else None, 'D_law': float(Da), 'J': 1000 * min(Jn, h0), 'h0': 1000 * h0, 'worth': bool(worth), 'checked': bool(worth and Dn / ell < 0.12)})
                print(f'    {veh.name[5:15]:<10s} {kind:<9s} {ell:7.0f} {c_p:6.3f} {Ds} {Da:8.0f} {rel} {1000 * min(Jn, h0):8.3f} {1000 * h0:8.3f} {str(worth):>6s}')
    print(f'    asymptotic law checked against the numerical optimum in {n_checked} cases (D*/l < 0.12)')
    print('\n[2b] Threshold correlation length: probing pays at some spacing iff c_p < l * A, with')
    print('     A = int_0^inf [h0 - h(s(y))] dy and s(y) the planner sd at y = d / l, so l_min = c_p / A')
    agree, total, sharp = (0, 0, 0)
    for veh in VEHICLES:
        for kind in ROUTES:
            pc = pcs[veh.name, kind]
            vc = float(np.max(route_family(kind)[1]))
            c_p = probe_time_cost(vc, u_probe * veh.mu * G)
            h0 = float(pc.h(sigma, z))
            y = np.linspace(0.0, 20.0, 20001)
            gap = h0 - pc.h(s_of_d(y, sigma, 1.0, s_p), z)
            A = float(np.sum(0.5 * (gap[1:] + gap[:-1]) * np.diff(y)))
            l_min = c_p / A if A > 0 else np.inf
            l_lin = c_p / ((1 - np.log(2)) * pc.kappa * z * sigma)
            for ell in (1000.0, 5000.0, 25000.0, 0.8 * l_min, 1.25 * l_min):
                res = minimize_scalar(lambda D: J(D, c_p, pc, z, sigma, ell, s_p), bounds=(1.0, 20 * ell), method='bounded')
                worth = res.fun < h0 * (1 - 1e-06) and res.x < 19 * ell
                hit = worth == (ell > l_min)
                if ell in (1000.0, 5000.0, 25000.0):
                    total += 1
                    agree += int(hit)
                else:
                    sharp += int(hit)
                record['threshold'].append({'vehicle': veh.name, 'route': kind, 'ell': float(ell), 'l_min': float(l_min), 'A': A, 'l_lin': float(l_lin), 'worth': bool(worth), 'predicted': bool(ell > l_min)})
            print(f'    {veh.name[5:15]:<10s} {kind:<9s} l_min = {l_min / 1000:7.2f} km  (linear-h closed form {l_lin / 1000:7.2f} km)')
    print(f'    threshold predicts probe/no-probe correctly in {agree}/{total} cases, and in {sharp}/24 cases at 0.8 and 1.25 times l_min')
    record['sharp'] = sharp
    ok &= agree == total and sharp == 24
    print('\n[3] Scaling of the law: D* vs l at small probe cost (Civic, urban)')
    pc = pcs[VEHICLES[1].name, 'urban']
    base = None
    for ell in (2000.0, 16000.0, 128000.0):
        res = minimize_scalar(lambda D: J(D, 0.1, pc, z, sigma, ell, 0.0001), bounds=(1.0, 20 * ell), method='bounded')
        if base is None:
            base = (ell, res.x)
        else:
            slope = np.log(res.x / base[1]) / np.log(ell / base[0])
            ok &= abs(slope - 1 / 3) < 0.06
            record['scaling'].append({'ell0': base[0], 'ell': ell, 'D0': float(base[1]), 'D': float(res.x), 'slope': float(slope)})
            print(f'    l {base[0]:.0f} -> {ell:.0f}: log-slope of D* = {slope:.3f} (law: 0.333)')
    record['agree'], record['total'], record['n_checked'] = (agree, total, n_checked)
    from well_posedness import save_record
    save_record('u1d_spacing', record)
    print('=' * 100)
    print(f"U1d CHECK: {('all hold' if ok else 'A CHECK FAILED')}")
    print('=' * 100)
    return 0 if ok else 1
if __name__ == '__main__':
    sys.exit(main())
