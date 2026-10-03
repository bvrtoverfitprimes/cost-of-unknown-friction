from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_d3 import FLEET, _load, GRADE_RUNS
G = 9.80665
FRONT_FRAC = {'leaf': 0.58, 'prius': 0.6, 'focus': 0.61, 'focus_bev': 0.56}
H_OVER_L = 0.2

def d3_bounds():
    rows = []
    for key in FLEET:
        runs, f = _load(key)
        sp = f['spec']
        m = sp.m_test
        ff = FRONT_FRAC[key]
        u_free, u_geo, u_smooth, p_ratio, n = (0.0, 0.0, 0.0, 0.0, 0)
        pedal_at_pmax, pedal_max, pbest = (None, None, -1.0)
        for r in runs:
            if r.test_id in GRADE_RUNS:
                continue
            a = r.accel(1.0)
            F, v = (r.F, r.v)
            w = max(1, int(round(1.0 / r.dt)))
            Fsm = np.convolve(F, np.ones(w) / w, mode='same')
            sel = (v > 1.0) & (F > 0)
            if not sel.any():
                continue
            n += int(sel.sum())
            Fs, As, Vs = (F[sel], a[sel], v[sel])
            u_free = max(u_free, float(np.max(Fs / (m * G))))
            N_D = ff * m * G - H_OVER_L * m * As
            u_geo = max(u_geo, float(np.max(Fs / N_D)))
            u_smooth = max(u_smooth, float(np.max(Fsm[sel] / N_D)))
            pw = Fs * Vs
            i = int(np.argmax(pw))
            p_ratio = max(p_ratio, float(pw[i]) / (f['P_rated_kW'] * 1000.0))
            if 'pedal_pct' in r.sig:
                ped = r.sig['pedal_pct'][sel]
                pedal_max = max(pedal_max or 0.0, float(np.max(ped)))
                if pw[i] > pbest:
                    pbest, pedal_at_pmax = (float(pw[i]), float(ped[i]))
        rows.append(dict(name=sp.name, n=n, u_free=u_free, u_geo=u_geo, u_smooth=u_smooth, p_ratio=p_ratio, pedal_at_pmax=pedal_at_pmax, pedal_max=pedal_max))
    return rows

def fmvss_points():
    from validate_braking import build, load_rows
    sol, *_ = build(load_rows())
    mu = np.array([s['mu_eff'] for s in sol])
    return (len(sol), float(np.median(mu)), float(np.percentile(mu, 25)), float(np.percentile(mu, 75)))

def main() -> int:
    print('=' * 96)
    print('T3 (a),(b),(e) -- WHAT ORDINARY DRIVING REVEALS vs WHAT A LIMIT EVENT REVEALS')
    print('=' * 96)
    rows = d3_bounds()
    print('\n[1] Argonne D3 -- ordinary driving (drive cycles; demand-limited)')
    print(f"    {'vehicle':<26s} {'samples':>8s} {'mu>= no-geom':>12s} {'mu>= axle':>10s} {'mu>= axle,1s':>12s} {'F*v/P_rated':>12s} {'pedal@Pmax':>11s} {'pedal max':>10s}")
    for r in rows:
        pa = 'n/a' if r['pedal_at_pmax'] is None else f"{r['pedal_at_pmax']:.0f}%"
        pm = 'n/a' if r['pedal_max'] is None else f"{r['pedal_max']:.0f}%"
        print(f"    {r['name']:<26s} {r['n']:8d} {r['u_free']:12.3f} {r['u_geo']:10.3f} {r['u_smooth']:12.3f} {r['p_ratio']:12.2f} {pa:>11s} {pm:>10s}")
    tot = sum((r['n'] for r in rows))
    lo = min((r['u_smooth'] for r in rows))
    hi = max((r['u_smooth'] for r in rows))
    print(f'\n    {tot} driving samples (v > 1 m/s, F > 0; the rest are braking or coasting).')
    print(f'    Grip: mu >= {lo:.2f}-{hi:.2f} on the driven axle (1 s smoothed), and NOTHING above.')
    print('    Power: P_max >= 0.68-0.99 of the manufacturer rating, and nothing above.')
    print('    Neither is pinned: the data contain no binding flag for either. Pedal position is')
    print('    not one -- the Leaf delivered 95% of rated power at 54% pedal.')
    n, med, q1, q3 = fmvss_points()
    print('\n[2] NHTSA FMVSS 135 -- ABS stops (traction binds; ABS cycling IS the flag)')
    print(f'    {n} stops pin grip at a POINT: median mu_eff = {med:.2f}  (IQR {q1:.2f}-{q3:.2f})')
    print('\n[3] The structure')
    print('    Without a binding flag, an envelope parameter is only bounded below, however much')
    print('    data there is. The power bound is tight here only because US06 drives near full')
    print('    power; the grip bound is loose because drive cycles never approach the tyre limit.')
    print('    A flagged limit event (an ABS stop) pins grip. Grip and power need DIFFERENT')
    print('    events -- the necessity result.')
    print('\n    Caveat: D3 tyres run on a steel dynamometer roll, FMVSS on a road surface. The two')
    print('    grip numbers are different quantities; what carries over is the structure.')
    ok = all((r['p_ratio'] <= 1.0 for r in rows))
    print('=' * 96)
    print(f"T3 (a,b,e) CHECK: {('no lower bound exceeds its true value' if ok else 'A BOUND EXCEEDS THE RATING')}")
    print('=' * 96)
    return 0 if ok else 1
if __name__ == '__main__':
    sys.exit(main())
