from __future__ import annotations
import sys
import numpy as np

def simulate(a, taus, V, dt=0.0001, t_max=200.0):
    v, t = (0.0, 0.0)
    y = np.zeros(len(taus))
    while v < V and t < t_max:
        u = a(v)
        x = u
        for j, tau in enumerate(taus):
            y[j] += dt * (x - y[j]) / tau
            x = y[j]
        d = min(x, u)
        v += dt * d
        t += dt
    return t

def t_phys(a, V, n=200000):
    w = np.linspace(0.0, V, n + 1)
    f = 1.0 / np.array([a(x) for x in w])
    return float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(w)))

def main() -> int:
    print('=' * 96)
    print('T7b -- TRANSIENT BOUND')
    print('=' * 96)
    ok = True
    rng = np.random.default_rng(1)
    record = {'monotone': [], 'staircase': [], 'general': []}
    print('\n[1] Random non-increasing envelopes: penalty / sum(tau) must be <= 1')
    worst = 0.0
    for trial in range(60):
        k = rng.integers(1, 4)
        taus = list(rng.uniform(0.02, 0.4, k))
        V = rng.uniform(5, 40)
        knots = np.sort(rng.uniform(0, V, 6))
        vals = np.sort(rng.uniform(0.3, 12.0, 7))[::-1]
        xs = np.concatenate([[0], knots, [V * 1.01]])
        ys = np.concatenate([vals, [vals[-1]]])
        a = lambda w, xs=xs, ys=ys: float(np.interp(w, xs, ys))
        pen = simulate(a, taus, V, dt=0.0002) - t_phys(a, V)
        ratio = pen / sum(taus)
        worst = max(worst, ratio)
        record['monotone'].append({'k': int(k), 'sum_tau': float(sum(taus)), 'V': float(V), 'penalty': float(pen), 'ratio': float(ratio)})
        ok &= ratio <= 1.0 + 0.02 and pen >= -0.001
    print(f'    60 random envelopes, 1-3 lags: worst penalty / sum(tau) = {worst:.3f}')
    print('\n[2] Increasing staircase, each level held ~5 tau: penalty grows without bound')
    print('    single lag; variation bound tau (1 + V+[ln a]); long-step prediction')
    print('    tau (1 + sum (1 - a_{i-1}/a_i))')
    tau, q, a0 = (0.2, 2.0, 1.0)
    prev = 0.0
    for steps in (1, 2, 4, 6, 8):
        levels = a0 * q ** np.arange(steps)
        widths = 5 * tau * levels
        edges = np.concatenate([[0], np.cumsum(widths)])

        def a(w, levels=levels, edges=edges):
            i = min(int(np.searchsorted(edges, w, side='right')) - 1, len(levels) - 1)
            return float(levels[max(i, 0)])
        V = float(edges[-1])
        pen = simulate(a, [tau], V, dt=0.0001) - t_phys(a, V, n=400000)
        vplus = float(np.sum(np.log(levels[1:] / levels[:-1])))
        bound = tau * (1 + vplus)
        pred = tau * (1 + np.sum(1 - levels[:-1] / levels[1:]))
        print(f'    {steps} levels: penalty {pen:.3f} s = {pen / tau:.2f} tau; long-step prediction {pred / tau:.2f} tau; variation bound {bound / tau:.2f} tau')
        record['staircase'].append({'levels': int(steps), 'tau': tau, 'penalty': float(pen), 'prediction': float(pred), 'bound': float(bound)})
        ok &= pen <= bound * 1.02
        if steps >= 2:
            ok &= pen > tau * 1.2
        ok &= pen > prev
        prev = pen
    print('\n[3] Variation bound, single lag, random general (non-monotone) envelopes')
    worst = 0.0
    for trial in range(40):
        tau = rng.uniform(0.05, 0.4)
        V = rng.uniform(5, 40)
        xs = np.concatenate([[0], np.sort(rng.uniform(0, V, 6)), [V * 1.01]])
        ys = rng.uniform(0.5, 10.0, 8)
        a = lambda w, xs=xs, ys=ys: float(np.interp(w, xs, ys))
        w = np.linspace(0, V, 20001)
        la = np.log([a(x) for x in w])
        vplus = float(np.sum(np.maximum(np.diff(la), 0)))
        pen = simulate(a, [tau], V, dt=0.0002) - t_phys(a, V)
        r = pen / (tau * (1 + vplus))
        worst = max(worst, r)
        record['general'].append({'tau': float(tau), 'V': float(V), 'vplus': vplus, 'penalty': float(pen), 'ratio': float(r), 'ratio_plain': float(pen / tau)})
        ok &= r <= 1.02
    print(f'    40 envelopes: worst penalty / [tau (1 + V+(ln a))] = {worst:.3f}')
    from well_posedness import save_record
    save_record('t7b_transient', record)
    print('=' * 96)
    print(f"T7b CHECK: {('all hold' if ok else 'A CHECK FAILED')}")
    print('=' * 96)
    return 0 if ok else 1
if __name__ == '__main__':
    sys.exit(main())
