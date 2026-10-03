from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pac_governed import PACVehicle
from t2_monotonicity import Env
from t4_uncertainty import fleet_mu_ratios
from validate_acceleration import VEHICLES, G
DS = 2.0

def route():
    s = np.arange(0.0, 6000.0 + DS, DS)
    vlim = np.full_like(s, 90 / 3.6)
    vlim[(s > 2500) & (s < 3300)] = 50 / 3.6
    kappa = np.zeros_like(s)
    kappa[(s > 1200) & (s < 1450)] = 1 / 120.0
    kappa[(s > 4200) & (s < 4500)] = 1 / 80.0
    grade = np.zeros_like(s)
    grade[(s > 1800) & (s < 2400)] = 0.04
    grade[(s > 3600) & (s < 4100)] = -0.03
    return (s, vlim, kappa, grade)
F_NODES = np.linspace(0.0, 1.0, 21)

class Tables:

    def __init__(self, veh, law, mu, grades, pac):
        self.veh, self.law, self.mu, self.pac = (veh, law, mu, pac)
        self.v = np.linspace(0.5, 45.0, 180)
        self.tab = {}
        self.ftab = {}
        for g in np.unique(grades):
            self.tab[g] = np.array([Env(veh, vv, law, mu=mu, grade=np.arctan(g), pac=pac).a_max() for vv in self.v])

    def a(self, v, g):
        return float(np.interp(v, self.v, self.tab[g]))

    def a_share(self, v, g, f):
        if f >= 1.0:
            return self.a(v, g)
        if g not in self.ftab:
            self.ftab[g] = np.array([[Env(self.veh, vv, self.law, mu=self.mu * fn, grade=np.arctan(g), pac=self.pac).a_max() for vv in self.v] for fn in F_NODES])
        col = np.array([np.interp(v, self.v, row) for row in self.ftab[g]])
        return float(np.interp(f, F_NODES, col))

def ellipse_share(v, kappa, mu_abs):
    ay = v * v * kappa / (mu_abs * G)
    return float(np.sqrt(max(0.0, 1.0 - ay * ay)))

def brake_decel(veh, v, share, mu_abs, grade):
    m = veh.mass
    meff = veh.m_eff(veh.F_PT(max(v, 0.5))[1])
    th = np.arctan(grade)
    return (share * mu_abs * m * G * np.cos(th) + veh.road_load_N(v) + m * G * np.sin(th)) / meff

def T_star(veh, law, mu, pac=None, R=None):
    s, vlim, kappa, grade = R if R is not None else route()
    mu_abs = veh.mu * mu
    with np.errstate(divide='ignore'):
        vk = np.where(kappa > 0, np.sqrt(mu_abs * G / np.maximum(kappa, 1e-12)), np.inf)
    cap = np.minimum(vlim, vk)
    tb = Tables(veh, law, mu, grade, pac)
    n = len(s)
    fwd = np.empty(n)
    fwd[0] = 0.0
    for i in range(n - 1):
        a = tb.a_share(fwd[i], grade[i], ellipse_share(fwd[i], kappa[i], mu_abs))
        fwd[i + 1] = min(cap[i + 1], np.sqrt(max(fwd[i] ** 2 + 2 * a * DS, 0.0)))
    bwd = np.empty(n)
    bwd[-1] = 0.0
    m = veh.mass
    for i in range(n - 1, 0, -1):
        v = bwd[i]
        a_brake = brake_decel(veh, v, ellipse_share(v, kappa[i], mu_abs), mu_abs, grade[i])
        bwd[i - 1] = min(cap[i - 1], np.sqrt(v * v + 2 * max(a_brake, 0.0) * DS))
    v = np.minimum(fwd, bwd)
    vm = 0.5 * (v[1:] + v[:-1])
    return (float(np.sum(DS / np.maximum(vm, 0.001))), v)

def main() -> int:
    print('=' * 96)
    print('T5 -- MINIMUM-TIME TRANSIT AS A PRODUCT OF THE ENVELOPE')
    print('=' * 96)
    R = route()
    ok = True
    record = {'grid': [float(x) for x in np.linspace(0.5, 1.5, 11)], 'monotone': [], 'quantiles': [], 'at_limit': []}
    print('\n[1] Monotonicity: T*(mu) over mu x[0.5, 1.5]')
    grid = np.linspace(0.5, 1.5, 11)
    for veh in VEHICLES:
        for law in ('ellipse', 'pac2002'):
            pac = PACVehicle(veh) if law == 'pac2002' else None
            Ts = [T_star(veh, law, mu, pac, R)[0] for mu in grid]
            mono = all((Ts[i + 1] <= Ts[i] + 1e-09 for i in range(len(Ts) - 1)))
            ok &= mono
            record['monotone'].append(dict(vehicle=veh.name, law=law, T=[float(x) for x in Ts], mono=bool(mono)))
            print(f'    {veh.name[:30]:<30s} {law:<8s} T*: {Ts[0]:7.2f} s (mu x0.5) -> {Ts[5]:7.2f} s (x1.0) -> {Ts[-1]:7.2f} s (x1.5)   non-increasing: {mono}')
    print('\n[2] Exact trip-time quantiles q_T(p) = T*(q_mu(1-p)) vs Monte Carlo')
    ratios = fleet_mu_ratios()
    ps = np.array([0.05, 0.5, 0.95])
    srt = np.sort(ratios)
    n = len(srt)
    q_upper = np.array([srt[n - int(np.ceil(n * p))] for p in ps])
    for veh in VEHICLES:
        mc = np.array([T_star(veh, 'ellipse', r, None, R)[0] for r in ratios])
        exact = np.array([T_star(veh, 'ellipse', q, None, R)[0] for q in q_upper])
        mcq = np.quantile(mc, ps, method='inverted_cdf')
        d = float(np.max(np.abs(exact - mcq)))
        ok &= d < 1e-09
        record['quantiles'].append(dict(vehicle=veh.name, ps=[float(x) for x in ps], exact=[float(x) for x in exact], mc=[float(x) for x in mcq], maxdiff=d))
        print(f'    {veh.name[:30]:<30s} q05/q50/q95 exact {exact[0]:7.2f}/{exact[1]:7.2f}/{exact[2]:7.2f} s   MC {mcq[0]:7.2f}/{mcq[1]:7.2f}/{mcq[2]:7.2f}   max|diff| {d:.1e}')
        spread = exact[2] - exact[0]
        print(f'      trip-time 90% band: {spread:.2f} s ({100 * spread / exact[1]:.1f}% of median); {len(ratios)} solves (MC) vs 3 (exact)')
    print('\n[3] Where the route is traction- vs cornering- vs limit-bound (nominal mu, ellipse)')
    for veh in VEHICLES:
        _, v = T_star(veh, 'ellipse', 1.0, None, R)
        s, vlim, kappa, grade = R
        at_lim = np.mean(np.isclose(v, vlim, atol=0.05))
        record['at_limit'].append(dict(vehicle=veh.name, share=float(at_lim)))
        print(f'    {veh.name[:30]:<30s} share of route at the speed limit: {100 * at_lim:5.1f}%')
    print('=' * 96)
    print(f"T5 NUMERICAL CHECK: {('all hold' if ok else 'A CHECK FAILED')}")
    print('=' * 96)
    from well_posedness import save_record
    save_record('t5_mintime', record)
    return 0 if ok else 1
if __name__ == '__main__':
    sys.exit(main())
