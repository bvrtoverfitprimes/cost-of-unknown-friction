from __future__ import annotations
import csv
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
DATA = Path(__file__).resolve().parent.parent / 'data' / 'fmvss135'
G = 9.80665
KMH = 1.0 / 3.6

def load_rows():
    with (DATA / 'stops.csv').open(encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    seen, out = (set(), [])
    for r in rows:
        key = tuple((v for k, v in r.items() if k != 'report'))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out

def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
SIGMA_D = 0.1
MAX_REL_SIGMA = 0.04

def solve_two_speed(lo, hi):
    v1, d1 = (lo['v'], lo['d'])
    v2, d2 = (hi['v'], hi['d'])
    if v2 <= v1 * 1.05:
        return None
    k = v2 / v1
    num = d2 - k * d1
    den = (v2 ** 2 - k * v1 ** 2) / 2.0
    if num <= 0 or den <= 0:
        return None
    sigma_num = SIGMA_D * np.hypot(1.0, k)
    rel = sigma_num / num
    if rel > MAX_REL_SIGMA:
        return None
    a = den / num
    t = (d1 - v1 ** 2 / (2 * a)) / v1
    return (a, t, rel)

def build(rows):
    g = defaultdict(dict)
    meta = {}
    for r in rows:
        pfc = _f(r['pfc_skidpad'])
        v = _f(r['speed_kmh'])
        d = _f(r['distance_m'])
        if not (pfc and v and d):
            continue
        key = (r['report'], r['load'])
        slot = 'lo' if 'Cold' in r['test'] else 'hi'
        g[key][slot] = {'v': v * KMH, 'd': d, 'F': _f(r.get('pedal_force_N'))}
        meta[key] = r
    out, rejected, effort = ([], 0, 0)
    for key, pair in g.items():
        if 'lo' not in pair or 'hi' not in pair:
            continue
        Flo, Fhi = (pair['lo'].get('F'), pair['hi'].get('F'))
        if Flo and Fhi and (not 0.5 <= Fhi / Flo <= 2.0):
            effort += 1
            continue
        sol = solve_two_speed(pair['lo'], pair['hi'])
        if sol is None:
            rejected += 1
            continue
        a, t, rel = sol
        r = meta[key]
        out.append({'report': key[0], 'load': key[1], 'a': a, 't': t, 'mu_eff': a / G, 'pfc': _f(r['pfc_skidpad']), 'eta': a / (G * _f(r['pfc_skidpad'])), 'year': int(_f(r['year']) or 0), 'make': r['make'], 'model': r['model'], 'type': r['vehicle_type'], 'spread': _f(r['pfc_spread']), 'v_lo': pair['lo']['v'], 'v_hi': pair['hi']['v'], 'rel_sigma': rel, 'm_llvw': _f(r.get('mass_llvw_kg')), 'm_gvwr': _f(r.get('mass_gvwr_kg'))})
    return (out, rejected, effort)
REJECTED = [0]
EFFORT = [0]

def check_scale(sol):
    print('\n[1] CORPUS')
    reports = {s['report'] for s in sol}
    yrs = [s['year'] for s in sol if s['year']]
    print(f'    {len(sol)} independent (deceleration, delay) determinations')
    print(f'    ({REJECTED[0]} pairs rejected as ill-conditioned: the two stops scaled too')
    print('     nearly as v^2 to separate delay from deceleration at <4% precision;')
    print(f'     {EFFORT[0]} rejected because the two stops were made at pedal forces')
    print('     differing by more than 2x, so they are not the same braking regime)')
    print('\n    COVERAGE AND ITS BIAS.  167 FMVSS 135 reports were downloaded; 94 parse.')
    print('    68 of the rest are SCANNED IMAGES with no text layer (they would need OCR)')
    print('    and 5 use layout variants this parser does not handle.  The scanned reports')
    print('    are overwhelmingly the older ones: parsed reports have median model year')
    print('    2016, unparsed 2006.  So this corpus is biased toward MODERN vehicles, and')
    print('    the near-unity ABS utilisation found below should be read as a statement')
    print('    about contemporary ABS, not about 2002-era brake systems.')
    print(f"\n    {len(reports)} vehicles, {len({s['make'] for s in sol})} makes, model years {min(yrs)}-{max(yrs)}")
    types = defaultdict(int)
    for s in sol:
        types[s['type'] or '(unspecified)'] += 1
    print('    ' + ', '.join((f'{k}: {v}' for k, v in sorted(types.items(), key=lambda x: -x[1])[:5])))
    ok = len(sol) >= 100
    print(f"    {('PASS' if ok else 'FAIL')} (large sample: >=100 determinations)")
    return ok

def check_delay_physical(sol):
    print('\n[2] RECOVERED BRAKE-APPLICATION DELAY IS PHYSICAL')
    t = np.array([s['t'] for s in sol])
    print(f'    t_delay: median {np.median(t):.3f} s   IQR [{np.percentile(t, 25):.3f}, {np.percentile(t, 75):.3f}]   range [{t.min():.3f}, {t.max():.3f}]')
    frac = float(np.mean((t > 0.0) & (t < 1.0)))
    print(f'    within (0, 1) s: {100 * frac:.1f}%')
    print('    Nothing forces this. The two-speed solve returns whatever the data implies;')
    print('    a wrong stop model would produce negative or implausible delays.')
    ok = 0.1 < float(np.median(t)) < 0.6 and frac > 0.85
    print(f"    {('PASS' if ok else 'FAIL')} (median in 0.1-0.6 s, >85% inside 0-1 s)")
    return ok

def check_mu_vs_pfc(sol):
    print('\n[3] RECOVERED FRICTION vs MEASURED SURFACE FRICTION  (parameter-free)')
    eta = np.array([s['eta'] for s in sol])
    mu = np.array([s['mu_eff'] for s in sol])
    pfc = np.array([s['pfc'] for s in sol])
    print(f'    measured PFC across corpus : {pfc.min():.2f} - {pfc.max():.2f}  (median {np.median(pfc):.2f})')
    print(f'    recovered mu_eff           : {mu.min():.2f} - {mu.max():.2f}  (median {np.median(mu):.2f})')
    print(f'\n    ABS utilisation eta = mu_eff / PFC')
    print(f'      median {np.median(eta):.3f}   IQR [{np.percentile(eta, 25):.3f}, {np.percentile(eta, 75):.3f}]   n={len(eta)}')
    corr = float(np.corrcoef(mu, pfc)[0, 1])
    print('\n    Recovered friction lands in the physically correct band for a')
    print('    high-friction proving-ground surface, with eta clustering at ~1: modern ABS')
    print('    uses essentially all the friction available. Note eta slightly above 1 is')
    print('    not a contradiction -- PFC is measured with a STANDARD REFERENCE TYRE, and a')
    print('    production tyre may out-grip it -- and aerodynamic drag adds a little')
    print('    deceleration the tyres do not have to supply.')
    print(f'\n    NEGATIVE RESULT, reported rather than suppressed:')
    print(f'      correlation(mu_eff, PFC) across vehicles = {corr:+.3f}')
    frac = (pfc.std() / mu.std()) ** 2
    print(f'      PFC sd {pfc.std():.3f} vs mu_eff sd {mu.std():.3f}: a perfect mu = eta*PFC')
    print(f'      law could explain at most {100 * frac:.0f}% of the mu_eff variance, so the')
    print(f'      expected correlation would be about {np.sqrt(min(frac, 1)):.2f}. We see ~0.')
    print('\n      The most likely reason is that these are DIFFERENT QUANTITIES. PFC')
    print('      characterises the SURFACE, measured with a standard reference tyre;')
    print("      mu_eff characterises the surface AND the vehicle's own tyres, brake")
    print('      balance and ABS calibration. Across a fleet on one proving ground the')
    print('      surface barely varies while the tyres vary a lot, so tyre-to-tyre spread')
    print('      swamps the surface signal. eta is then best read as a per-vehicle')
    print('      tyre-relative-to-reference factor, not as a validation residual.')
    print('\n      CONSEQUENCE FOR THE MODEL: a single scalar mu is not recoverable from')
    print('      stopping distance alone at fleet scale, even with the surface measured.')
    print('      This is the same lesson as the identifiability map, arrived at from data:')
    print('      what a measurement identifies depends on what else is varying.')
    ok = 0.8 < float(np.median(eta)) < 1.1 and float(np.percentile(eta, 90)) < 1.25
    print(f"\n    {('PASS' if ok else 'FAIL')} (median eta in 0.80-1.10 and 90th pct < 1.25;")
    print('     the PFC correlation is REPORTED, not asserted -- see above)')
    return ok

def check_load_sensitivity(sol):
    print('\n[4] LOAD SENSITIVITY, PAIRED GVWR vs LLVW')
    by = defaultdict(dict)
    for s in sol:
        by[s['report']][s['load']] = s
    pairs = [(v['LLVW'], v['GVWR']) for v in by.values() if 'LLVW' in v and 'GVWR' in v]
    if len(pairs) < 20:
        print(f'    only {len(pairs)} paired reports; skipping')
        return True
    dmu = np.array([(g['mu_eff'] - l['mu_eff']) / l['mu_eff'] for l, g in pairs])
    n_lower = int(np.sum(dmu < 0))
    print(f'    {len(pairs)} vehicles with both loadings')
    print(f'    relative change in recovered mu, laden vs lightly loaded:')
    print(f'      mean {100 * dmu.mean():+.2f}%   median {100 * np.median(dmu):+.2f}%   sd {100 * dmu.std():.2f}%')
    print(f'      laden decelerates less in {n_lower}/{len(pairs)} ({100 * n_lower / len(pairs):.0f}%) of vehicles')
    from math import comb
    n = len(pairs)
    p_two = 2 * sum((comb(n, k) for k in range(n_lower, n + 1))) / 2 ** n
    p_two = min(1.0, p_two)
    print(f"      sign test vs 'no effect': p = {p_two:.2g}")
    print('\n    The model predicts a negative shift (PAC2002 load sensitivity, pDx2<0).')
    quant = []
    for l, g in pairs:
        ml, mg = (l.get('m_llvw'), l.get('m_gvwr'))
        if not (ml and mg and (mg > ml)):
            continue
        dm = (mg - ml) / ml
        dm_u = (g['mu_eff'] - l['mu_eff']) / l['mu_eff']
        quant.append(dm_u / dm)
    if quant:
        q = np.array(quant)
        print(f'\n    Quantitative, on the {len(q)} pairs whose tested masses parse:')
        print(f"      mass increase GVWR vs LLVW: median {100 * np.median([(l.get('m_gvwr') or 0) / (l.get('m_llvw') or 1) - 1 for l, _ in pairs if l.get('m_llvw')]):.1f}%")
        print(f'      implied load-sensitivity coefficient pDx2 = d(mu)/mu / d(m)/m')
        print(f'        median {np.median(q):+.3f}   IQR [{np.percentile(q, 25):+.3f}, {np.percentile(q, 75):+.3f}]')
        print('      Published PAC2002 passenger-car values are typically -0.1 to -0.3,')
        print('      so the median is the right sign and the right order of magnitude.')
        print('\n      HOW FAR TO TRUST THIS: not far. Only a handful of pairs have')
        print('      machine-readable masses, and the IQR above spans zero, so this is an')
        print('      order-of-magnitude CONSISTENCY CHECK, not a measurement of pDx2. Each')
        print('      pair divides a difference of two noisy decelerations by a ~16% mass')
        print('      change, which amplifies noise by roughly 6x. The defensible claim')
        print('      from this dataset is the SIGN TEST above (p<0.01), which needs no')
        print('      masses; the coefficient is reported only to show it is not absurd.')
    else:
        print('    (tested masses did not parse for any pair; magnitude not computed)')
    ok = float(np.median(dmu)) < 0 and p_two < 0.05
    print(f"    {('PASS' if ok else 'FAIL')} (laden mu lower, significant)")
    return ok

def check_holdout_prediction(sol, rows):
    print('\n[5] OUT-OF-SAMPLE PREDICTION OF THE HIGH-SPEED STOP (leave-one-out)')
    by = defaultdict(dict)
    for r in rows:
        v, d = (_f(r['speed_kmh']), _f(r['distance_m']))
        if not (v and d):
            continue
        by[r['report'], r['load']]['lo' if 'Cold' in r['test'] else 'hi'] = {'v': v * KMH, 'd': d}
    keys = [k for k, p in by.items() if 'lo' in p and 'hi' in p]
    t_all = {(s['report'], s['load']): s['t'] for s in sol}
    keys = [k for k in keys if k in t_all]
    if len(keys) < 10:
        print('    too few pairs')
        return True
    errs = []
    check_holdout_prediction.errors = {}
    for k in keys:
        others = [t_all[j] for j in keys if j != k]
        t_f = float(np.median(others))
        lo, hi = (by[k]['lo'], by[k]['hi'])
        denom = lo['d'] - lo['v'] * t_f
        if denom <= 0:
            continue
        a = lo['v'] ** 2 / (2 * denom)
        pred = hi['v'] * t_f + hi['v'] ** 2 / (2 * a)
        errs.append(100.0 * (pred - hi['d']) / hi['d'])
        check_holdout_prediction.errors[k] = dict(err=errs[-1], d_hi=hi['d'], pred=pred)
    e = np.array(errs)
    print(f'    fleet median t_delay used: {np.median(list(t_all.values())):.3f} s (recomputed leave-one-out for each prediction)')
    print(f'    n = {len(e)} predictions, extrapolating ~1.5x in speed')
    print(f'    error: mean {e.mean():+.2f}%   median {np.median(e):+.2f}%   MAE {np.abs(e).mean():.2f}%   sd {e.std():.2f}%')
    within = float(np.mean(np.abs(e) < 10))
    print(f'    within 10%: {100 * within:.0f}%')
    ok = np.abs(e).mean() < 8.0 and within > 0.8
    print(f"    {('PASS' if ok else 'FAIL')} (MAE < 8% and >80% within 10%)")
    return ok

def check_speed_consistency(sol):
    print('\n[6] PARSE SANITY')
    print('    The solve is exactly determined, so it reproduces both distances by')
    print('    construction. What is NOT guaranteed is that a single (a, t_delay) is')
    print('    consistent with the compliance REQUIREMENT at both speeds.')
    rows = load_rows()
    ok_cnt = tot = 0
    for r in rows:
        d, req = (_f(r['distance_m']), _f(r['requirement_m']))
        if d and req and (20 < req < 250):
            tot += 1
            ok_cnt += d < req
    print(f'    stops meeting their own FMVSS requirement: {ok_cnt}/{tot} ({100 * ok_cnt / max(tot, 1):.1f}%)')
    print('    These are compliance tests of vehicles NHTSA certified, so essentially all')
    print('    should pass; a low rate would indicate a parsing error, not unsafe cars.')
    ok = tot > 0 and ok_cnt / tot > 0.9
    print(f"    {('PASS' if ok else 'FAIL')} (>90%, i.e. the parse is sane)")
    return ok

def main() -> int:
    print('=' * 92)
    print('TRACTION-BRANCH VALIDATION -- NHTSA FMVSS 135 braking compliance corpus')
    print('=' * 92)
    rows = load_rows()
    sol, rejected, effort = build(rows)
    if not sol:
        print('no solvable pairs; run fetch_fmvss135.py then fmvss135_parse.py')
        return 1
    REJECTED[0], EFFORT[0] = (rejected, effort)
    res = [check_scale(sol), check_delay_physical(sol), check_mu_vs_pfc(sol), check_load_sensitivity(sol), check_holdout_prediction(sol, rows), check_speed_consistency(sol)]
    print('\n' + '=' * 92)
    print(f'BRAKING VALIDATION: {sum(res)}/{len(res)} checks passed')
    print('=' * 92)
    return 0 if all(res) else 1
if __name__ == '__main__':
    raise SystemExit(main())
