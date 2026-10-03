from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t2_monotonicity import make
from validate_acceleration import MPH_TO_MS, VEHICLES

def fleet_mu_ratios():
    from validate_braking import build, load_rows
    sol, *_ = build(load_rows())
    mu = np.array([s['mu_eff'] for s in sol])
    return mu / np.median(mu)

def main() -> int:
    print('=' * 96)
    print('T4 -- UNCERTAINTY PROPAGATION')
    print('=' * 96)
    ok = True
    ratios = fleet_mu_ratios()
    print(f'friction spread (FMVSS, n={len(ratios)}): relative sd {ratios.std():.3f}, 5-95% {np.percentile(ratios, 5):.3f}-{np.percentile(ratios, 95):.3f}')
    print('\n[1] Exact quantiles q_a(p) = a_max(q_mu(p)) vs Monte Carlo (same samples)')
    print(f"    {'vehicle':<30s} {'law':<8s} {'q05 exact':>10s} {'q05 MC':>8s} {'q50':>7s} {'q95 exact':>10s} {'q95 MC':>8s} {'max|diff|':>10s}")
    ps = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    record = {'ratios': [float(x) for x in ratios], 'ps': [float(x) for x in ps], 'quantiles': [], 'box': [], 'first_order': []}
    for veh in VEHICLES:
        for law in ('ellipse', 'pac2002'):
            v = 10 * MPH_TO_MS
            mc = np.array([make(veh, v, law, mu=r).a_max() for r in ratios])
            exact = np.array([make(veh, v, law, mu=q).a_max() for q in np.quantile(ratios, ps, method='inverted_cdf')])
            mcq = np.quantile(mc, ps, method='inverted_cdf')
            d = float(np.max(np.abs(exact - mcq)))
            ok &= d < 1e-09
            record['quantiles'].append(dict(vehicle=veh.name, law=law, exact=[float(x) for x in exact], mc=[float(x) for x in mcq], maxdiff=d))
            print(f'    {veh.name[:30]:<30s} {law:<8s} {exact[0]:10.4f} {mcq[0]:8.4f} {exact[2]:7.4f} {exact[4]:10.4f} {mcq[4]:8.4f} {d:10.2e}')
    print('\n[2] Interval over a box: vertices vs 4096-point dense sweep')
    print('    box: mu x[0.85,1.15], C_rr x[0.8,1.3], C_dA x[0.9,1.1], P x[0.9,1.05]')
    rng = np.random.default_rng(0)
    box = dict(mu=(0.85, 1.15), crr=(0.8, 1.3), cda=(0.9, 1.1), P=(0.9, 1.05))
    best = dict(mu=1.15, crr=0.8, cda=0.9, P=1.05)
    worst = dict(mu=0.85, crr=1.3, cda=1.1, P=0.9)
    print(f"    {'vehicle':<30s} {'law':<8s} {'mph':>4s} {'vertex lo':>10s} {'sweep lo':>9s} {'sweep hi':>9s} {'vertex hi':>10s}")
    for veh in VEHICLES:
        for law in ('ellipse', 'pac2002'):
            for vm in (10, 60):
                v = vm * MPH_TO_MS
                lo = make(veh, v, law, **worst).a_max()
                hi = make(veh, v, law, **best).a_max()
                pts = [make(veh, v, law, **{k: rng.uniform(*box[k]) for k in box}).a_max() for _ in range(4096 if law == 'ellipse' else 512)]
                slo, shi = (min(pts), max(pts))
                ok &= lo <= slo + 1e-09 and shi <= hi + 1e-09
                record['box'].append(dict(vehicle=veh.name, law=law, mph=vm, n=len(pts), vertex_lo=lo, sweep_lo=slo, sweep_hi=shi, vertex_hi=hi))
                print(f'    {veh.name[:30]:<30s} {law:<8s} {vm:4d} {lo:10.4f} {slo:9.4f} {shi:9.4f} {hi:10.4f}')
    print('\n[3] First-order sd(a_max) and the T0 amplification factor (10 mph)')
    print(f"    {'vehicle':<30s} {'law':<8s} {'sd(a) MC':>9s} {'sd(a) 1st-order':>16s} {'amplification':>14s}")
    for veh in VEHICLES:
        for law in ('ellipse', 'pac2002'):
            v = 10 * MPH_TO_MS
            e = make(veh, v, law)
            a0 = e.a_max()
            h = 0.0001
            dadmu = (make(veh, v, law, mu=1 + h).a_max() - make(veh, v, law, mu=1 - h).a_max()) / (2 * h)
            mc = np.array([make(veh, v, law, mu=r).a_max() for r in ratios])
            first = abs(dadmu) * ratios.std()
            dT = (e.T(a0 + 0.001) - e.T(a0 - 0.001)) / 0.002
            amp = e.m_eff / (e.m_eff - dT) if e.T(a0) < e.P else float('nan')
            record['first_order'].append(dict(vehicle=veh.name, law=law, sd_mc=float(mc.std()), sd_first=float(first), amplification=float(amp)))
            print(f'    {veh.name[:30]:<30s} {law:<8s} {mc.std():9.4f} {first:16.4f} {amp:14.3f}')
    print('=' * 96)
    print(f"T4 NUMERICAL CHECK: {('all hold' if ok else 'A CHECK FAILED')}")
    print('=' * 96)
    from well_posedness import save_record
    save_record('t4_uncertainty', record)
    return 0 if ok else 1
if __name__ == '__main__':
    sys.exit(main())
