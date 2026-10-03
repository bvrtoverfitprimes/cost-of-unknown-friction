from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.stats import norm
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t4_uncertainty import fleet_mu_ratios
from t5_mintime import DS, T_star, route
from validate_acceleration import G, VEHICLES

def calibrated_prior(ratios, delta=0.05):
    q = float(np.quantile(ratios, delta, method='inverted_cdf'))
    return (float(-np.log(q) / norm.ppf(1 - delta)), q)
KMH = 1 / 3.6

def route_family(kind):
    if kind == 'rural':
        return route()
    if kind == 'highway':
        s = np.arange(0.0, 20000.0 + DS, DS)
        vlim = np.full_like(s, 110 * KMH)
        kappa = np.zeros_like(s)
        for a in (5000, 12000):
            kappa[(s > a) & (s < a + 800)] = 1 / 600.0
        grade = np.zeros_like(s)
        grade[(s > 8000) & (s < 9500)] = 0.03
        return (s, vlim, kappa, grade)
    if kind == 'urban':
        s = np.arange(0.0, 5000.0 + DS, DS)
        vlim = np.full_like(s, 50 * KMH)
        for x in np.arange(400.0, 5000.0, 400.0):
            vlim[np.abs(s - x) < DS / 2] = 0.0
        kappa = np.zeros_like(s)
        grade = np.zeros_like(s)
        return (s, vlim, kappa, grade)
    if kind == 'mountain':
        s = np.arange(0.0, 8000.0 + DS, DS)
        vlim = np.full_like(s, 70 * KMH)
        kappa = np.zeros_like(s)
        for i, a in enumerate(np.arange(400.0, 8000.0, 500.0)):
            R = (35.0, 60.0, 90.0)[i % 3]
            kappa[(s > a) & (s < a + 80)] = 1 / R
        grade = np.zeros_like(s)
        grade[(s > 0) & (s < 4000)] = 0.07
        grade[s >= 4000] = -0.07
        return (s, vlim, kappa, grade)
    raise ValueError(kind)
ROUTES = ('highway', 'rural', 'mountain', 'urban')

def n_transitions(R):
    s, vlim, kappa, grade = R
    stops = int(np.sum(vlim == 0)) + 2
    curves = int(np.sum(np.diff((kappa > 0).astype(int)) == 1))
    drops = int(np.sum(np.diff(vlim[vlim > 0]) < 0))
    return (stops, curves, drops)

def profile_feasible(veh, v, mu_ratio, R, tol=1e-06):
    from t5_mintime import Tables, brake_decel, ellipse_share
    s, vlim, kappa, grade = R
    mu_abs = veh.mu * mu_ratio
    if np.any(kappa > 0):
        if np.any(v[kappa > 0] ** 2 * kappa[kappa > 0] > mu_abs * G * (1 + 1e-09) + tol):
            return False
    tb = Tables(veh, 'ellipse', mu_ratio, grade, None)
    m = veh.mass
    dv2 = np.diff(v ** 2) / (2 * DS)
    for i in np.nonzero(dv2 > 0)[0]:
        if dv2[i] > tb.a_share(v[i], grade[i], ellipse_share(v[i], kappa[i], mu_abs)) + 1e-06:
            return False
    for i in np.nonzero(dv2 < 0)[0]:
        b = brake_decel(veh, v[i + 1], ellipse_share(v[i + 1], kappa[i + 1], mu_abs), mu_abs, grade[i + 1])
        if -dv2[i] > max(b, 0.0) + 1e-06:
            return False
    return True

def d3_cycle_bounds():
    from t3_identifiability import FRONT_FRAC, H_OVER_L
    from validate_d3 import FLEET, GRADE_RUNS, _load
    out = {}
    for key in FLEET:
        runs, f = _load(key)
        m = f['spec'].m_test
        for r in runs:
            if r.test_id in GRADE_RUNS:
                continue
            c = r.cycle.lower()
            kind = 'us06' if 'us06' in c else 'highway' if 'highway' in c else 'udds' if 'udds' in c else None
            if kind is None:
                continue
            w = max(1, int(round(1.0 / r.dt)))
            Fsm = np.convolve(r.F, np.ones(w) / w, mode='same')
            a = r.accel(1.0)
            sel = r.v > 1.0
            N_D = FRONT_FRAC[key] * m * G - H_OVER_L * m * a
            ut = float(np.max(np.where(Fsm[sel] > 0, Fsm[sel] / N_D[sel], 0.0)))
            ub = float(np.max(np.where(Fsm[sel] < 0, -Fsm[sel] / (m * G), 0.0)))
            d = out.setdefault((f['spec'].name, kind), [0.0, 0.0])
            d[0], d[1] = (max(d[0], ut), max(d[1], ub))
    return out

def braking_excess(v, b):
    t_brake = v / b
    dist = v * v / (2 * b)
    return t_brake - dist / v

def transition_excess(V, a, n=4000):
    w = np.linspace(0.0, V, n + 1)
    f = (1 - w / V) / np.array([a(x) for x in w])
    return float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(w)))

def probe_info(u_rel, n=50, eta=0.02):
    x_max = 1.0 - (1.0 - min(u_rel, 0.999999)) ** (1.0 / 3.0)
    x = np.linspace(0.0, x_max, 2001)[1:]
    s_mu = x * x * (3 - 2 * x)
    s_c = 3 * x * (1 - x) ** 2
    I = n / eta ** 2 * np.array([[np.mean(s_mu * s_mu), np.mean(s_mu * s_c)], [np.mean(s_mu * s_c), np.mean(s_c * s_c)]])
    return 1.0 / np.linalg.inv(I)[0, 0]

def probe_time_cost(v, u_p_g, tau=0.5, a_re=1.5):
    dv = u_p_g * tau
    return (dv * tau / 2 + dv * dv / (2 * a_re)) / v

def ratchet(s0, k, eps=0.05, n=50, eta=0.02):
    info = 1.0 / s0 ** 2
    sds, us = ([s0], [])
    z = norm.ppf(1 - eps)
    for _ in range(k):
        s = 1.0 / np.sqrt(info)
        u = np.exp(-z * s)
        us.append(u)
        info += probe_info(u, n, eta)
        sds.append(1.0 / np.sqrt(info))
    return (sds, us)

def main() -> int:
    print('=' * 100)
    print('U1 -- PRICE OF UNIDENTIFIABILITY AND VALUE OF PROBING')
    print('=' * 100)
    ok = True
    ratios = fleet_mu_ratios()
    lr = np.log(ratios)
    s0_sample = float(np.std(lr))
    s0, q_emp = calibrated_prior(ratios)
    print(f'friction belief from the FMVSS fleet: sd(ln mu) = {s0_sample:.3f}, calibrated s0 = {s0:.4f} (q05 = {q_emp:.3f}, n = {len(ratios)})')
    record = {'s0': s0, 's0_sample': s0_sample, 'q05': q_emp, 'chance': [], 'cycles': [], 'price': [], 'law': [], 'ratchet': [], 'gain': [], 'kstar': []}
    print(f'\n[a] Chance-constrained min time = T*(q_mu(delta)): execute each plan at all {len(ratios)} mu')
    R = route_family('rural')
    srt = np.sort(ratios)
    for veh in VEHICLES:
        for delta in (0.05, 0.25):
            mu_plan = srt[int(np.floor(delta * len(srt)))]
            T, v = T_star(veh, 'ellipse', mu_plan, None, R)
            feas = np.array([profile_feasible(veh, v, r, R) for r in ratios])
            should = ratios >= mu_plan - 1e-12
            match = bool(np.all(feas == should))
            p_fail = 1 - feas.mean()
            ok &= match and p_fail <= delta + 1e-12
            record['chance'].append(dict(vehicle=veh.name, delta=delta, mu_plan=float(mu_plan), T=float(T), n_infeasible=int((~feas).sum()), n_below=int((~should).sum()), n=len(ratios), match=match))
            print(f'    {veh.name[:28]:<28s} delta={delta:.2f}  mu_plan={mu_plan:.3f}  T={T:7.2f} s  infeasible share {p_fail:.3f}  feasible set is the up-set: {match}')
    print('\n[b] Identified lower bounds from ordinary driving (Argonne D3) vs pinned road friction')
    b = d3_cycle_bounds()
    from validate_braking import build, load_rows
    sol, *_ = build(load_rows())
    mu_road = float(np.median([s['mu_eff'] for s in sol]))
    record['mu_road'] = mu_road
    rows = {}
    for (name, kind), (ut, ub) in sorted(b.items()):
        rows.setdefault(kind, []).append(max(ut, ub))
        record['cycles'].append(dict(vehicle=name, cycle=kind, traction=float(ut), braking=float(ub)))
        print(f'    {name[:28]:<28s} {kind:<8s} traction u={ut:.2f}  braking u={ub:.2f}  -> mu >= {max(ut, ub):.2f}  ({max(ut, ub) / mu_road:.2f} of road mu)')
    r_cycle = {k: float(np.median(v)) / mu_road for k, v in rows.items()}
    r_ord = round(max(r_cycle.get('udds', 0), r_cycle.get('highway', 0)), 2)
    r_all = round(max(r_cycle.values()), 2)
    record['r_ord'], record['r_us06'] = (r_ord, r_all)
    u_cycle = {k: float(np.median(v)) for k, v in rows.items()}
    u_ord = round(max(u_cycle.get('udds', 0), u_cycle.get('highway', 0)), 2)
    u_all = round(max(u_cycle.values()), 2)
    record['u_ord'], record['u_us06'] = (u_ord, u_all)
    print(f'    absolute observed utilisation used for planning: ordinary {u_ord:.2f}; incl. US06 {u_all:.2f}')
    print(f"    median identified fraction: UDDS {r_cycle.get('udds', 0):.2f}, Highway {r_cycle.get('highway', 0):.2f}, US06 {r_cycle.get('us06', 0):.2f}")
    print(f'    planning fractions used: ordinary (UDDS+Highway) {r_ord:.2f}; incl. US06 {r_all:.2f}')
    print('\n[b/c] Price of unidentifiability T*(r) - T*(1) by route family [s and % of T*(1)]')
    hdr = f"    {'vehicle':<22s} {'route':<9s} {'stops':>5s} {'curves':>6s} {'T*(1)':>8s}"
    hdr += f" {'ordinary':>16s} {'incl. US06':>16s} {'fleet q05':>16s}"
    print(hdr)
    q05 = float(np.quantile(ratios, 0.05, method='inverted_cdf'))
    price = {}
    for veh in VEHICLES:
        for kind in ROUTES:
            Rk = route_family(kind)
            st, cu, dr = n_transitions(Rk)
            T1 = T_star(veh, 'ellipse', 1.0, None, Rk)[0]
            out = []
            for r in (u_ord / veh.mu, u_all / veh.mu, q05):
                Tr = T_star(veh, 'ellipse', r, None, Rk)[0]
                out.append((Tr - T1, 100 * (Tr - T1) / T1))
            price[veh.name, kind] = (T1, out)
            ok &= all((o[0] >= -1e-09 for o in out))
            record['price'].append(dict(vehicle=veh.name, route=kind, stops=st, curves=cu, T1=float(T1), r=[u_ord / veh.mu, u_all / veh.mu, q05], price=[float(o[0]) for o in out]))
            print(f'    {veh.name[:22]:<22s} {kind:<9s} {st:5d} {cu:6d} {T1:8.1f}' + ''.join((f' {o[0]:7.2f} ({o[1]:5.2f}%)' for o in out)))
    print('\n[c] Exact per-stop braking excess v/(2b) and its friction price')
    for vk in (50, 90):
        v = vk * KMH
        e1, e2 = (braking_excess(v, G * mu_road), braking_excess(v, G * u_ord))
        ok &= abs(e1 - v / (2 * G * mu_road)) < 1e-12
        print(f'    from {vk} km/h: excess {e1:.3f} s at mu={mu_road:.3f}, {e2:.3f} s at mu={u_ord:.3f}; price per stop {e2 - e1:.3f} s')
    veh = VEHICLES[1]
    T1, out = price[veh.name, 'urban']
    st = n_transitions(route_family('urban'))[0]
    v = 50 * KMH
    brake_only = st * (braking_excess(v, G * u_ord) - braking_excess(v, G * veh.mu))
    print(f'    urban, {veh.name[:20]}: solver price {out[0][0]:.2f} s vs braking-only law {brake_only:.2f} s ({st} stops); remainder = traction-limited launches')
    from t5_mintime import Tables, brake_decel
    print('    transition-excess law  E(V) = int_0^V (1 - w/V)/a(w) dw  vs the solver (urban route):')
    for veh in VEHICLES:
        Ru = route_family('urban')
        n_cyc = n_transitions(Ru)[0] - 1
        T1 = price[veh.name, 'urban'][0]
        for r in (u_ord / veh.mu, u_all / veh.mu):

            def law(rr):
                tb = Tables(veh, 'ellipse', rr, np.zeros(1), None)
                mu_abs = veh.mu * rr
                Ea = transition_excess(v, lambda w: tb.a(max(w, 0.5), 0.0))
                Eb = transition_excess(v, lambda w: brake_decel(veh, w, 1.0, mu_abs, 0.0))
                return n_cyc * (Ea + Eb)
            pred = law(r) - law(1.0)
            sol = T_star(veh, 'ellipse', r, None, Ru)[0] - T1
            ok &= abs(pred - sol) / sol < 0.05
            brk = n_cyc * (transition_excess(v, lambda w: brake_decel(veh, w, 1.0, veh.mu * r, 0.0)) - transition_excess(v, lambda w: brake_decel(veh, w, 1.0, veh.mu, 0.0)))
            record['law'].append(dict(vehicle=veh.name, r=float(r), cycles=n_cyc, law=float(pred), solver=float(sol), braking=float(brk)))
            print(f'      {veh.name[:26]:<26s} r={r:.2f}: law {pred:7.2f} s  solver {sol:7.2f} s  ({100 * (pred - sol) / sol:+.1f}%)')
    print('\n[d] Sequential braking probes, each limited to slip risk eps = 5%')
    print('    probe: 0.5 s ramp, n = 50 samples, force noise 2%, brush model, stiffness unknown')
    sds, us = ratchet(s0, 6)
    for i, (sd, u) in enumerate(zip(sds[1:], us)):
        print(f'    probe {i + 1}: utilisation {u:.3f} of mu   ->  sd(ln mu) {sds[i]:.4f} -> {sd:.4f}')
    ok &= all((sds[i + 1] < sds[i] for i in range(len(sds) - 1)))
    ok &= all((us[i + 1] > us[i] for i in range(len(us) - 1)))
    record['ratchet'] = [dict(k=i + 1, u=float(u), sd_before=float(sds[i]), sd_after=float(sds[i + 1])) for i, u in enumerate(us)]
    zd = norm.ppf(0.95)
    print('\n    Trip-time gain from probing (planner uses the 5% quantile), Civic and ZR1:')
    print(f"    {'vehicle':<22s} {'route':<9s} {'prior q05':>10s} {'1 probe':>9s} {'3 probes':>9s} {'probe cost':>10s} {'net (3)':>8s}")
    for veh in VEHICLES[1:]:
        for kind in ROUTES:
            Rk = route_family(kind)
            T_prior = T_star(veh, 'ellipse', np.exp(-zd * s0), None, Rk)[0]
            T_1 = T_star(veh, 'ellipse', np.exp(-zd * sds[1]), None, Rk)[0]
            T_3 = T_star(veh, 'ellipse', np.exp(-zd * sds[3]), None, Rk)[0]
            vc = float(np.max(Rk[1]))
            cost = sum((probe_time_cost(vc, us[i] * veh.mu * G) for i in range(3)))
            T1 = price[veh.name, kind][0]
            ok &= T_3 <= T_1 <= T_prior
            print(f'    {veh.name[:22]:<22s} {kind:<9s} {T_prior - T1:10.2f} {T_1 - T1:9.2f} {T_3 - T1:9.2f} {cost:10.3f} {T_prior - T_3 - cost:8.2f}')
    print('    (columns 3-5: excess over the full-information time T*(1); net = gain - probe cost)')
    print('\n[d2] Optimal number of probes k* = argmax_k [T*(prior q05) - T*(q05 after k) - cost(k)]')
    for veh in VEHICLES:
        for kind in ROUTES:
            Rk = route_family(kind)
            vc = float(np.max(Rk[1]))
            T_prior = T_star(veh, 'ellipse', np.exp(-zd * s0), None, Rk)[0]
            nets = [0.0]
            for k in range(1, 6):
                Tk = T_star(veh, 'ellipse', np.exp(-zd * sds[k]), None, Rk)[0]
                cost = sum((probe_time_cost(vc, us[i] * veh.mu * G) for i in range(k)))
                nets.append(T_prior - Tk - cost)
            k_star = int(np.argmax(nets))
            record['kstar'].append(dict(vehicle=veh.name, route=kind, k=k_star, prior_price=float(T_prior - price[veh.name, kind][0]), nets=[float(x) for x in nets]))
            print(f'    {veh.name[:22]:<22s} {kind:<9s} k* = {k_star}  net gain {nets[k_star]:6.2f} s  (nets k=0..5: ' + ', '.join((f'{x:.2f}' for x in nets)) + ')')
    print('=' * 100)
    print(f"U1 CHECK: {('all hold' if ok else 'A CHECK FAILED')}")
    print('=' * 100)
    from well_posedness import save_record
    save_record('u1_value', record)
    return 0 if ok else 1
if __name__ == '__main__':
    sys.exit(main())
