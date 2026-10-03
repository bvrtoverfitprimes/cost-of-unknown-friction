from __future__ import annotations
import sys
import numpy as np
G, RHO = (9.80665, 1.2)
NAMES = ['mu', 'm', 'C_rr', 'C_dA', 'grade', 'wind']
TH0 = np.array([0.85, 1400.0, 0.01, 0.7, 0.02, 3.0])
WF, HL, J = (0.61, 0.2, 60.0)

def balance(th, v, a):
    mu, m, crr, cda, g, w = th
    return (m + J) * a + crr * m * G * np.cos(g) + 0.5 * RHO * cda * (v + w) ** 2 + m * G * np.sin(g)

def binding_drive(th, v, a):
    mu, m, crr, cda, g, w = th
    N = m * G * (WF * np.cos(g) - HL * np.sin(g)) - m * HL * a
    return mu * N

def binding_brake(th, v, a):
    mu, m, crr, cda, g, w = th
    return mu * m * G * np.cos(g)

def jac(fn, th, v, a, h=1e-06):
    row = np.zeros(len(th))
    for k in range(len(th)):
        d = np.zeros(len(th))
        d[k] = h * TH0[k]
        row[k] = (fn(th + d, v, a) - fn(th - d, v, a)) / (2 * h)
    return row

def a_limit(th, v):
    mu, m, crr, cda, g, w = th
    R = crr * m * G * np.cos(g) + 0.5 * RHO * cda * (v + w) ** 2 + m * G * np.sin(g)
    N0 = m * G * (WF * np.cos(g) - HL * np.sin(g))
    return (mu * N0 - R) / (m + J + mu * m * HL)

def limit_events(speeds, k=5):
    return [(binding_drive, v, a_limit(TH0, v)) for v in speeds for _ in range(k)]

def design(extra):
    rows = []
    rng = np.random.default_rng(0)
    for v, a in zip(rng.uniform(5, 30, 60), rng.uniform(-1.5, 2.0, 60)):
        rows.append(jac(balance, TH0, v, a))
    for fn, v, a in extra:
        rows.append(jac(balance, TH0, v, a))
        rows.append(jac(fn, TH0, v, a))
    return np.array(rows)

def report(label, Jm):
    s = np.linalg.svd(Jm, compute_uv=False)
    _, _, Vt = np.linalg.svd(Jm)
    tol = s[0] * 1e-09
    rank = int(np.sum(s > tol))
    print(f'\n  {label}')
    print(f'    singular values: ' + ' '.join((f'{x:.2e}' for x in s)))
    print(f'    structural rank {rank} of {Jm.shape[1]}')
    nulls = []
    for k in range(rank, Jm.shape[1]):
        vec = Vt[k]
        vec = vec / vec[np.argmax(np.abs(vec))]
        terms = ' '.join((f'{NAMES[i]}:{vec[i]:+.2f}' for i in range(len(vec)) if abs(vec[i]) > 0.001))
        print(f'    null direction: {terms}')
        nulls.append(terms)
    report.records.append(dict(label=label, singular=[float(x) for x in s], rank=rank, rows=int(Jm.shape[0]), null=nulls))
    return rank
report.records = []

def main() -> int:
    print('=' * 96)
    print('T3(c) -- JOINT IDENTIFIABILITY: Fisher rank of the six-parameter vector')
    print('=' * 96)
    a8, a20, a5, a30 = (a_limit(TH0, v) for v in (8.0, 20.0, 5.0, 30.0))
    print(f'  limit accelerations at the nominal parameters: a(8) = {a8:.3f}, a(20) = {a20:.3f}, a(5) = {a5:.3f}, a(30) = {a30:.3f} m/s^2')
    r1 = report('D1 ordinary driving (balance only)', design([]))
    r2 = report(f'D2 + drive-axle limit events at ONE speed (v = 8, a = {a8:.2f})', design(limit_events([8.0])))
    r3 = report(f'D3 + drive-axle limit events at TWO speeds (v = 8 and 20, a = {a8:.2f} and {a20:.2f})', design(limit_events([8.0, 20.0])))
    r4 = report('D4 + four-wheel BRAKING limit events at two decelerations (control)', design([(binding_brake, 20.0, -7.0)] * 5 + [(binding_brake, 25.0, -5.0)] * 5))
    ok = (r1, r2, r3, r4) == (4, 5, 6, 5)
    print('\n  Predicted ranks 4, 5, 6, 5.')
    sigma = 50.0
    crbs = []
    for speeds in ([8.0, 20.0], [5.0, 30.0]):
        Jm = design(limit_events(speeds))
        s = np.linalg.svd(Jm, compute_uv=False)
        cov = np.linalg.inv(Jm.T @ Jm / sigma ** 2)
        se_rel = np.sqrt(np.diag(cov))
        cond = float(s[0] / s[-1])
        print(f'\n  Cramer-Rao standard errors with limit events at v = {speeds} (sigma_F = 50 N, 80 samples), condition number {cond:.2e}:')
        for nm, sv, t in zip(NAMES, se_rel, TH0):
            print(f'    {nm:<6s} +/- {100 * sv:10.1f}%   (absolute +/- {sv * t:.4g})')
        crbs.append(dict(speeds=speeds, accel=[float(a_limit(TH0, v)) for v in speeds], cond=cond, rows=[dict(name=nm, rel=float(sv), abs=float(sv * t)) for nm, sv, t in zip(NAMES, se_rel, TH0)]))
    print('=' * 96)
    print(f"T3(c) CHECK: {('prediction confirmed' if ok else f'ranks were {(r1, r2, r3, r4)}')}")
    print('=' * 96)
    import json
    from pathlib import Path
    out = Path(__file__).resolve().parent.parent / 'data'
    out.mkdir(parents=True, exist_ok=True)
    (out / 't3c_fisher.json').write_text(json.dumps(dict(designs=report.records, names=NAMES, nominal=[float(x) for x in TH0], crb=crbs, sigma=sigma, wf=WF, hl=HL, J=J), indent=1))
    return 0 if ok else 1
if __name__ == '__main__':
    sys.exit(main())
