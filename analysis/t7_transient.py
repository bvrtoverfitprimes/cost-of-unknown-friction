from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_acceleration import MPH_TO_MS, VEHICLES
V60 = 60 * MPH_TO_MS

def envelope(veh):
    vg = np.linspace(0.0, V60 * 1.05, 400)
    ag = np.array([veh.a_max_fixedpoint(max(v, 0.05))[0] for v in vg])
    return lambda v: float(np.interp(v, vg, ag))

def run(a_phys, taus, sigma=None, v0=None, dt=0.0002):
    v, t = (0.0, 0.0)
    ys = [0.0] * (len(taus) + (1 if sigma else 0))
    while v < V60 and t < 60:
        u = a_phys(v)
        stages = list(taus) + ([sigma / max(v, v0)] if sigma else [])
        x = u
        for k, tau in enumerate(stages):
            ys[k] += dt * (x - ys[k]) / tau
            x = ys[k]
        x = min(x, u)
        v += dt * x
        t += dt
    return t

def run_phys(a_phys, dt=0.0002):
    v, t = (0.0, 0.0)
    while v < V60:
        v += dt * a_phys(v)
        t += dt
    return t

def main() -> int:
    print('=' * 90)
    print('T7 -- TRANSIENT DEFICIT')
    print('=' * 90)
    ok = True
    record = {'lags': [], 'relax': [], 'vplus': []}
    for veh in VEHICLES:
        a = envelope(veh)
        w = np.linspace(0.0, V60, 4001)
        la = np.log([a(x) for x in w])
        record['vplus'].append({'vehicle': veh.name, 'vplus': float(np.sum(np.maximum(np.diff(la), 0.0))), 'a0': float(a(0.0)), 'a60': float(a(V60))})
    print('\n[1] Constant lags: 0-60 time penalty vs the predicted sum of time constants')
    print(f"    {'vehicle':<32s} {'lags [s]':<16s} {'T_phys':>7s} {'T_real':>7s} {'penalty':>8s} {'sum tau':>8s} {'pen/sum':>7s}")
    for veh in VEHICLES:
        a = envelope(veh)
        Tp = run_phys(a)
        for taus in ([0.1], [0.15, 0.05], [0.3, 0.1, 0.05]):
            Tr = run(a, taus)
            pen, pred = (Tr - Tp, sum(taus))
            err = 100 * (pen - pred) / pred
            ok &= pen >= -0.001 and pen <= pred * 1.02
            record['lags'].append({'vehicle': veh.name, 'taus': list(taus), 'T_phys': Tp, 'T_real': Tr, 'penalty': pen, 'ratio': pen / pred})
            print(f'    {veh.name[:32]:<32s} {str(taus):<16s} {Tp:7.3f} {Tr:7.3f} {pen:8.3f} {pred:8.3f} {100 * pen / pred:6.1f}%')
    print('\n[2] Speed-dependent tyre relaxation tau = sigma/max(v, v0), sigma = 0.5 m, plus tau_p = 0.10 s')
    print(f"    {'vehicle':<32s} " + ''.join((f"{'v0=' + str(v0):>12s}" for v0 in (2.0, 1.0, 0.5, 0.25))))
    for veh in VEHICLES:
        a = envelope(veh)
        Tp = run_phys(a)
        row = f'    {veh.name[:32]:<32s} '
        pens = []
        for v0 in (2.0, 1.0, 0.5, 0.25):
            pens.append(run(a, [0.1], sigma=0.5, v0=v0) - Tp)
            row += f'{pens[-1]:12.3f}'
        record['relax'].append({'vehicle': veh.name, 'v0': [2.0, 1.0, 0.5, 0.25], 'penalty': pens})
        print(row)
    print('    (penalty in seconds; it grows as the floor speed v0 falls -- the tau = sigma/v')
    print('     singularity at standstill -- so a launch value depends on the low-speed tyre model)')
    from well_posedness import save_record
    save_record('t7_transient', record)
    print('    positive variation of ln a_max over 0-60 mph: ' + ', '.join((f"{r['vehicle'][5:15]} {r['vplus']:.2e}" for r in record['vplus'])))
    print('=' * 90)
    print(f"T7 CHECK: {('0 <= penalty <= sum of time constants in every case' if ok else 'FAILED')}")
    print('=' * 90)
    return 0 if ok else 1
if __name__ == '__main__':
    sys.exit(main())
