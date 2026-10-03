from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pac_governed import PACVehicle
from validate_acceleration import MPH_TO_MS, VEHICLES, G
LBF_TO_N = 4.4482216152605

class Env:

    def __init__(self, veh, v, law, mu=1.0, m=1.0, crr=1.0, cda=1.0, rho=1.0, P=1.0, grade=0.0, pac=None):
        self.veh, self.v, self.law = (veh, v, law)
        self.s = dict(mu=mu, m=m, crr=crr, cda=cda, rho=rho, P=P)
        self.gam = grade
        self.pac = pac if law == 'pac2002' else None
        F_pt, gear = veh.F_PT(v)
        self.ice = len(veh.gear_ratios) > 1
        self.P = F_pt * P * (rho if self.ice else 1.0)
        self.mass = veh.mass * m
        self.J = veh.m_eff(gear) - veh.mass
        self.m_eff = self.mass + self.J
        self.hL = veh.cg_height_m / veh.wheelbase_m
        self.k = self.mass * self.hL
        W = self.mass * G
        self.c_f = veh.front_weight_frac * W * np.cos(grade) - self.hL * W * np.sin(grade)
        self.c_r = (1 - veh.front_weight_frac) * W * np.cos(grade) + self.hL * W * np.sin(grade)
        A = veh.A_lbf * LBF_TO_N
        B = veh.B_lbfmph * LBF_TO_N / MPH_TO_MS
        C = veh.C_lbfmph2 * LBF_TO_N / MPH_TO_MS ** 2
        self.F_roll = (A + B * v) * crr * m * np.cos(grade)
        self.F_aero = C * v * v * cda * rho
        self.R = self.F_roll + self.F_aero + W * np.sin(grade)
        self.a_lo, self.a_hi = (-self.c_r / self.k, self.c_f / self.k)
        self.driven = {'FWD': ['f'], 'RWD': ['r']}.get(veh.drive, ['f', 'r'])

    def N(self, a):
        return {'f': self.c_f - self.k * a, 'r': self.c_r + self.k * a}

    def phi(self, Nax):
        Nax = max(Nax, 0.0)
        if self.law == 'ellipse':
            return self.veh.mu * Nax
        return 2.0 * self.pac.peak_wheel_force(max(Nax / 2.0, 1.0))

    def T(self, a):
        n = self.N(a)
        return self.s['mu'] * sum((self.phi(n[j]) for j in self.driven))

    def Phi(self, a):
        return min(self.P, self.T(a)) - self.R - self.m_eff * a

    def a_max(self):
        lo, hi = (self.a_lo * 0.999, self.a_hi * 0.999)
        if self.Phi(hi) >= 0:
            return hi
        return brentq(self.Phi, lo, hi, xtol=1e-12)

    def regime(self, a):
        return 'traction' if self.T(a) < self.P else 'power'

    def Lambda(self, a, h=1.0):
        n = self.N(a)
        out = 0.0
        for j in self.driven:
            Nj = n[j]
            dphi = (self.phi(Nj + h) - self.phi(Nj - h)) / (2 * h)
            out += self.s['mu'] * (self.phi(Nj) - dphi * Nj)
        return out

    def grade_gain(self, a, h=1.0):
        if 'r' not in self.driven:
            return -np.inf
        Nr = self.N(a)['r']
        dphi = (self.phi(Nr + h) - self.phi(Nr - h)) / (2 * h)
        return self.s['mu'] * dphi * self.hL
_PAC_CACHE = {}

def make(veh, v, law, **kw):
    pac = None
    if law == 'pac2002':
        key = (veh.name, veh.mu, veh.mass_curb_kg, veh.driver_kg)
        pac = _PAC_CACHE.get(key)
        if pac is None:
            pac = _PAC_CACHE[key] = PACVehicle(veh)
    return Env(veh, v, law, pac=pac, **kw)

def monotone_checks():
    tests = [('mu', +1, np.linspace(0.5, 1.5, 11)), ('crr', -1, np.linspace(0.5, 2.0, 11)), ('cda', -1, np.linspace(0.5, 2.0, 11)), ('P', +1, np.linspace(0.5, 1.5, 11))]
    rows, total_bad = ([], 0)
    for name, sign, vals in tests:
        bad = n = 0
        min_step = np.inf
        for veh in VEHICLES:
            for law in ('ellipse', 'pac2002'):
                for vm in (10, 40, 80):
                    prev = None
                    for x in vals:
                        a = make(veh, vm * MPH_TO_MS, law, **{name: x}).a_max()
                        if prev is not None:
                            n += 1
                            min_step = min(min_step, sign * (a - prev))
                            if sign * (a - prev) < -1e-09:
                                bad += 1
                        prev = a
        rows.append((name, 'non-decreasing' if sign > 0 else 'non-increasing', n, bad, float(vals[0]), float(vals[-1]), float(min_step)))
        total_bad += bad
    return (rows, total_bad)

def grade_check():
    bad = n = 0
    worst_gain = -np.inf
    min_step = np.inf
    for veh in VEHICLES:
        for law in ('ellipse', 'pac2002'):
            for vm in (10, 40, 80):
                prev = None
                for gdeg in np.linspace(0, 8, 9):
                    e = make(veh, vm * MPH_TO_MS, law, grade=np.radians(gdeg))
                    a = e.a_max()
                    worst_gain = max(worst_gain, e.grade_gain(a))
                    if prev is not None:
                        n += 1
                        min_step = min(min_step, prev - a)
                        if a - prev > 1e-09:
                            bad += 1
                    prev = a
    grade_check.min_step = float(min_step)
    return (n, bad, worst_gain)

def sign_checks(eps=0.0001):
    out = []
    for veh in VEHICLES:
        for law in ('ellipse', 'pac2002'):
            for vm in (10, 40, 80):
                v = vm * MPH_TO_MS
                e0 = make(veh, v, law)
                a0 = e0.a_max()
                reg = e0.regime(a0)
                am = make(veh, v, law, m=1 + eps).a_max()
                dm = (am - a0) / eps
                if reg == 'traction':
                    pred_m = e0.J * a0 + e0.F_aero - e0.Lambda(a0)
                else:
                    pred_m = -1.0
                ar = make(veh, v, law, rho=1 + eps).a_max()
                dr = (ar - a0) / eps
                if reg == 'traction':
                    pred_r = -e0.F_aero
                else:
                    pred_r = e0.P - e0.F_aero if e0.ice else -e0.F_aero
                out.append(dict(veh=veh.name, law=law, vm=vm, reg=reg, a=a0, dm=dm, pred_m=pred_m, J_a=e0.J * a0, Faero=e0.F_aero, Lam=e0.Lambda(a0) if reg == 'traction' else float('nan'), dr=dr, pred_r=pred_r))
    return out

def main() -> int:
    print('=' * 100)
    print('T2 -- MONOTONICITY AND SIGN TABLE (straight line; 3 vehicles x 2 tyre laws x 3 speeds)')
    print('=' * 100)
    ok = True
    rows, bad = monotone_checks()
    record = {'monotone': [], 'grade': {}, 'signs': []}
    print('\n[1] Monotone parameters (comparison principle)')
    for name, direction, n, b, lo, hi, ms in rows:
        print(f'    {name:4s} predicted {direction:15s}  comparisons {n:4d}  violations {b}  multiplier {lo:.1f}-{hi:.1f}  smallest step {ms:.2e}')
        record['monotone'].append(dict(param=name, direction=direction, n=n, violations=b, lo=lo, hi=hi, min_step=ms))
    ok &= bad == 0
    n, b, wg = grade_check()
    print("\n[2] Grade (uphill), predicted non-increasing while mu*phi'*h/L < 1 on the rear axle")
    print(f"    comparisons {n}  violations {b}  worst mu*phi'*h/L = {wg:.3f}")
    record['grade'] = dict(n=n, violations=b, worst_gain=float(wg), min_step=grade_check.min_step)
    ok &= b == 0 and wg < 1
    rows = sign_checks()
    for r in rows:
        record['signs'].append({k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in r.items()})
    print('\n[3] Sign formulas at the root (finite difference vs prediction)')
    print(f"    {'vehicle':<30s} {'law':<8s} {'mph':>3s} {'regime':<9s} {'da/dlnm':>9s} {'J*a':>7s} {'F_aero':>7s} {'Lambda':>8s} {'m ok':>5s} {'da/drho':>9s} {'rho ok':>6s}")
    for r in rows:
        mok = np.sign(r['dm']) == np.sign(r['pred_m']) or abs(r['dm']) < 1e-06
        rok = np.sign(r['dr']) == np.sign(r['pred_r']) or abs(r['dr']) < 1e-06
        ok &= bool(mok and rok)
        lam = '' if np.isnan(r['Lam']) else f"{r['Lam']:8.0f}"
        print(f"    {r['veh'][:30]:<30s} {r['law']:<8s} {r['vm']:3d} {r['reg']:<9s} {r['dm']:9.3f} {r['J_a']:7.0f} {r['Faero']:7.0f} {lam:>8s} {str(bool(mok)):>5s} {r['dr']:9.4f} {str(bool(rok)):>6s}")
    print('    (da/dlnm = d a_max / d(ln m) in m/s^2, i.e. per unit relative change in mass;')
    print('     da/drho likewise per unit relative change in air density; J*a, F_aero, Lambda in N)')
    print('=' * 100)
    print(f"T2 NUMERICAL CHECK: {('all predictions hold' if ok else 'A PREDICTION FAILED')}")
    print('=' * 100)
    from well_posedness import save_record
    save_record('t2_monotonicity', record)
    return 0 if ok else 1
if __name__ == '__main__':
    sys.exit(main())
