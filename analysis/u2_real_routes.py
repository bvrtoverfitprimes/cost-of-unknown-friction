from __future__ import annotations
import json
import re
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t4_uncertainty import fleet_mu_ratios
from t5_mintime import DS, T_star
from u1_value_of_information import calibrated_prior, probe_time_cost, ratchet
from validate_acceleration import G, VEHICLES
ROUTE_DIR = Path(__file__).resolve().parent.parent / 'data' / 'routes'
NAMES = ('argonne_naperville', 'boulder_canyon')

def parse_speed(tag):
    if tag is None:
        return None
    m = re.match('\\s*(\\d+(?:\\.\\d+)?)\\s*(mph|km/h|kmh)?', str(tag))
    if not m:
        return None
    v = float(m.group(1))
    return v * 0.44704 if m.group(2) == 'mph' or (m.group(2) is None and 'mph' in str(tag)) else v / 3.6 if m.group(2) else v * 0.44704

def load_route(name, signals_stop):
    d = json.load(open(ROUTE_DIR / f'{name}.json'))
    lon = np.array([c[0] for c in d['coords']])
    lat = np.array([c[1] for c in d['coords']])
    R_E = 6371000.0
    x = np.radians(lon - lon[0]) * R_E * np.cos(np.radians(lat.mean()))
    y = np.radians(lat - lat[0]) * R_E
    seg = np.hypot(np.diff(x), np.diff(y))
    keep = np.concatenate([[True], seg > 0.05])
    x, y = (x[keep], y[keep])
    elev = np.array(d['elevation'])[keep]
    ctrl = [c for c, k in zip(d['controls'], keep) if k]
    spd = [parse_speed(s['maxspeed']) for s in d['segments']]
    spd = [spd[i - 1] if i > 0 else spd[0] for i in np.nonzero(keep)[0]]
    s_v = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    tagged = np.array([v is not None for v in spd])
    vals = np.array([v if v is not None else np.nan for v in spd])
    idx = np.nonzero(tagged)[0]
    for i in np.nonzero(~tagged)[0]:
        vals[i] = vals[idx[np.argmin(np.abs(idx - i))]]
    s = np.arange(0.0, s_v[-1], DS)
    vlim = vals[np.clip(np.searchsorted(s_v, s, side='right') - 1, 0, len(vals) - 1)]
    xs, ys = (np.interp(s, s_v, x), np.interp(s, s_v, y))
    heading = np.unwrap(np.arctan2(np.gradient(ys), np.gradient(xs)))
    kappa = np.abs(np.gradient(gaussian_filter1d(heading, 15.0 / DS), DS))
    kappa[kappa < 1 / 2000.0] = 0.0
    z = gaussian_filter1d(np.interp(s, s_v, elev), 100.0 / DS)
    grade = np.clip(np.gradient(z, DS), -0.12, 0.12)
    grade = np.round(grade / 0.005) * 0.005
    vlim = vlim.copy()
    vlim[0] = vlim[-1] = 0.0
    n_stop, n_sig = (0, 0)
    for c, sv in zip(ctrl, s_v):
        if c == 'stop' or (c == 'traffic_signals' and signals_stop):
            vlim[min(int(round(sv / DS)), len(vlim) - 1)] = 0.0
        n_stop += c == 'stop'
        n_sig += c == 'traffic_signals'
    meta = dict(L=s[-1], n_stop=n_stop, n_sig=n_sig, climb=float(np.sum(np.maximum(np.diff(z), 0))), tagged=float(tagged.mean()), max_kappa=float(kappa.max()))
    return ((s, vlim, kappa, grade), meta)

def main() -> int:
    print('=' * 104)
    print('U2 -- REAL ROUTES: PRICE OF UNIDENTIFIABILITY AND VALUE OF PROBING')
    print('=' * 104)
    ok = True
    ratios = fleet_mu_ratios()
    s0, _ = calibrated_prior(ratios)
    zd = norm.ppf(0.95)
    q05 = float(np.quantile(ratios, 0.05, method='inverted_cdf'))
    import json
    u1 = json.loads((Path(__file__).resolve().parent.parent / 'data' / 'u1_value.json').read_text())
    u_ord, u_us06 = (u1['u_ord'], u1['u_us06'])
    sds, us = ratchet(s0, 3)
    record = {'s0': s0, 'q05': q05, 'q_lognormal': float(np.exp(-zd * s0)), 'q_one_probe': float(np.exp(-zd * sds[1])), 'u_probe': float(us[0]), 'u_ord': u_ord, 'u_us06': u_us06, 'cases': []}
    for name in NAMES:
        for sig in (False, True):
            R, meta = load_route(name, sig)
            print(f"\n{name}  (signals {('all stop' if sig else 'all green')}): {meta['L'] / 1000:.1f} km, {meta['n_sig']} signals, {meta['n_stop']} stop signs, climb {meta['climb']:.0f} m, limits tagged on {100 * meta['tagged']:.0f}% of vertices, tightest radius {1 / max(meta['max_kappa'], 1e-09):.0f} m")
            print(f"    {'vehicle':<22s} {'T*(1)':>8s} {'ordinary':>15s} {'incl.US06':>15s} {'fleet q05':>15s} {'1-probe net':>11s}")
            case = {'route': name, 'signals_stop': sig, **meta, 'rows': []}
            for veh in VEHICLES:
                T1 = T_star(veh, 'ellipse', 1.0, None, R)[0]
                cells = []
                prices = []
                for r in (u_ord / veh.mu, u_us06 / veh.mu, q05):
                    Tr = T_star(veh, 'ellipse', r, None, R)[0]
                    prices.append(float(Tr - T1))
                    cells.append(f'{Tr - T1:7.1f} ({100 * (Tr - T1) / T1:4.1f}%)')
                    ok &= Tr >= T1 - 1e-09
                Tq = T_star(veh, 'ellipse', np.exp(-zd * s0), None, R)[0]
                T1p = T_star(veh, 'ellipse', np.exp(-zd * sds[1]), None, R)[0]
                cost = probe_time_cost(float(np.median(R[1][R[1] > 0])), us[0] * veh.mu * G)
                net = Tq - T1p - cost
                case['rows'].append({'vehicle': veh.name, 'T1': float(T1), 'prices': prices, 'Tq': float(Tq), 'T1p': float(T1p), 'cost': float(cost), 'net': float(net)})
                print(f'    {veh.name[:22]:<22s} {T1:8.1f} ' + ' '.join(cells) + f' {net:11.2f}')
            Ts = [T_star(VEHICLES[1], 'ellipse', m, None, R)[0] for m in np.linspace(0.4, 1.4, 11)]
            mono = all((Ts[i + 1] <= Ts[i] + 1e-09 for i in range(10)))
            ok &= mono
            case['mono_mu'] = [float(m) for m in np.linspace(0.4, 1.4, 11)]
            case['mono_T'] = [float(t) for t in Ts]
            case['mono'] = bool(mono)
            record['cases'].append(case)
            print(f'    T*(mu) non-increasing over mu x[0.4, 1.4] (Civic): {mono}')
    from well_posedness import save_record
    save_record('u2_routes', record)
    print('=' * 104)
    print(f"U2 CHECK: {('all hold' if ok else 'A CHECK FAILED')}")
    print('=' * 104)
    return 0 if ok else 1
if __name__ == '__main__':
    sys.exit(main())
