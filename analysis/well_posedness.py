from __future__ import annotations
import sys
from copy import deepcopy
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pac_governed import PACVehicle
from validate_acceleration import MPH_TO_MS, VEHICLES, G

class Envelope:

    def __init__(self, veh, v, law):
        self.veh, self.v, self.law = (veh, v, law)
        self.F_pt, self.gear = veh.F_PT(v)
        self.m_eff = veh.m_eff(self.gear)
        self.R = veh.road_load_N(v)
        W = veh.mass * G
        self.k = veh.mass * veh.cg_height_m / veh.wheelbase_m
        self.c_f = veh.front_weight_frac * W
        self.c_r = (1 - veh.front_weight_frac) * W
        self.a_lo, self.a_hi = (-self.c_r / self.k, self.c_f / self.k)
        self.pac = PACVehicle(veh) if law == 'pac2002' else None

    def T(self, a):
        if self.law == 'ellipse':
            return self.veh.traction_limit_N(a)
        N = self.pac.normal_loads(a)
        return sum((self.pac.peak_wheel_force(N[w]) for w in self.pac.driven_wheels()))

    def Phi(self, a):
        return min(self.F_pt, self.T(a)) - self.R - self.m_eff * a

    def grid(self, n=4001):
        return np.linspace(self.a_lo * 0.999, self.a_hi * 0.999, n)

def check(env):
    a = env.grid()
    Phi = np.array([env.Phi(x) for x in a])
    Tv = np.array([env.T(x) for x in a])
    da = a[1] - a[0]
    t_sup = float(np.max(np.diff(Tv) / da))
    g = max(t_sup, 0.0)
    strictly_dec = bool(np.all(np.diff(Phi) < 0))
    feas = a[Phi >= 0]
    a_grid_max = float(feas.max()) if feas.size else float('nan')
    interval = bool(feas.size and np.all(Phi[a <= a_grid_max] >= 0))
    max_slope = float(np.max(np.diff(Phi) / da))
    return dict(t_sup=t_sup, g=g, margin=env.m_eff - g, strictly_dec=strictly_dec, interval=interval, a_grid_max=a_grid_max, max_slope=max_slope)

def a_max_root(env):
    from scipy.optimize import brentq
    lo, hi = (env.a_lo * 0.999, env.a_hi * 0.999)
    if env.Phi(hi) >= 0:
        return hi
    return brentq(env.Phi, lo, hi, xtol=1e-12)

def lipschitz_check(veh, v, law, rel=0.0001):
    e0 = Envelope(veh, v, law)
    c0 = check(e0)
    a0 = a_max_root(e0)
    v2 = deepcopy(veh)
    v2.mu *= 1 + rel
    e1 = Envelope(v2, v, law)
    a1 = a_max_root(e1)
    measured = abs(a1 - a0) / (veh.mu * rel)
    binds = e0.T(a0) < e0.F_pt
    dPhi_dmu = e0.T(a0) / veh.mu if binds else 0.0
    bound = dPhi_dmu / c0['margin'] if c0['margin'] > 0 else float('inf')
    h = 0.0001
    t_root = (e0.T(a0 + h) - e0.T(a0 - h)) / (2 * h)
    local = dPhi_dmu / (e0.m_eff - t_root) if binds else 0.0
    return (measured, bound, binds, local, t_root)

def main() -> int:
    print('=' * 96)
    print('T0 -- WELL-POSEDNESS OF THE LONGITUDINAL ENVELOPE')
    print('=' * 96)
    print(f"{'vehicle':<32s} {'law':<8s} {'v[mph]':>6s} {'m_eff':>7s} {'gain g':>8s} {'margin':>8s} {'Phi dec':>8s} {'interval':>9s}")
    ok = True
    record = {'cases': [], 'lipschitz': []}
    for veh in VEHICLES:
        for law in ('ellipse', 'pac2002'):
            for vm in (5, 30, 60, 100):
                env = Envelope(veh, vm * MPH_TO_MS, law)
                c = check(env)
                good = c['strictly_dec'] and c['interval'] and (c['margin'] > 0)
                ok &= good
                record['cases'].append(dict(vehicle=veh.name, law=law, mph=vm, m_eff=env.m_eff, t_sup=c['t_sup'], g=c['g'], margin=c['margin'], max_slope=c['max_slope'], a_max=c['a_grid_max'], strictly_dec=c['strictly_dec'], interval=c['interval']))
                print(f"{veh.name[:32]:<32s} {law:<8s} {vm:6d} {env.m_eff:7.0f} {c['g']:8.1f} {c['margin']:8.1f} {str(c['strictly_dec']):>8s} {str(c['interval']):>9s}")
    print('\nLipschitz bound (iii): |d a_max / d mu| <= |dPhi/dmu| / (m_eff - g)')
    print(f"{'vehicle':<32s} {'law':<8s} {'v':>4s} {'measured':>10s} {'bound':>10s} {'binds':>6s}")
    for veh in VEHICLES:
        for law in ('ellipse', 'pac2002'):
            vm = 10
            meas, bound, binds, local, t_root = lipschitz_check(veh, vm * MPH_TO_MS, law)
            holds = meas <= bound * 1.02 + 1e-06 and abs(meas - local) <= 0.005 * max(local, 1.0)
            ok &= holds
            record['lipschitz'].append(dict(vehicle=veh.name, law=law, mph=vm, measured=meas, bound=bound, local=local, t_root=t_root, traction_binds=bool(binds)))
            print(f"{veh.name[:32]:<32s} {law:<8s} {vm:4d} {meas:10.4f} {bound:10.4f} {str(binds):>6s}  {('ok' if holds else 'VIOLATED')}")
    print('\nReading: g is the gain of traction force w.r.t. acceleration through load')
    print("transfer. FWD has T' < 0 and so g = 0. RWD/AWD have g = sup T' > 0, and the")
    print('envelope is well-posed only while g < m_eff. The local sensitivity factor at the root is')
    print("m_eff/(m_eff - T'(a_max)), which is below 1 for FWD and above 1 for RWD.")
    print('=' * 96)
    print(f"T0 NUMERICAL CHECK: {('all conditions hold' if ok else 'A CONDITION FAILED')}")
    print('=' * 96)
    save_record('well_posedness', record)
    return 0 if ok else 1

def save_record(name, record):
    import json
    out = Path(__file__).resolve().parent.parent / 'data'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{name}.json').write_text(json.dumps(record, indent=1, default=float))
if __name__ == '__main__':
    sys.exit(main())
