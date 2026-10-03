from __future__ import annotations
import sys
import numpy as np

def g(x):
    return np.stack([x ** 2 * (3 - 2 * x), 3 * x * (1 - x) ** 2], axis=-1)

def x_of_u(u):
    return 1 - (1 - u) ** (1 / 3)

def sd_joint(u_max, n, design):
    xm = x_of_u(u_max)
    if design == 'uniform in slip':
        x = np.linspace(xm / n, xm, n)
    else:
        x = x_of_u(np.linspace(u_max / n, u_max, n))
    G = g(x)
    info = G.T @ G
    return float(np.sqrt(np.linalg.inv(info)[0, 0]))

def main() -> int:
    print('=' * 90)
    print('T3(d) -- INFORMATION ABOUT PEAK FRICTION vs UTILISATION (brush model)')
    print('=' * 90)
    print('\n[1] Exact per-sample sensitivity mu dF/dmu = N x^2 (3-2x) vs the small-u law N u^2/3')
    for u in (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 0.95):
        x = x_of_u(u)
        print(f'    u = {u:4.2f}   exact {x ** 2 * (3 - 2 * x):.5f}   u^2/3 {u * u / 3:.5f}   ratio {x ** 2 * (3 - 2 * x) / (u * u / 3):.3f}')
    print('\n[2] Small-slip joint-variance constant: n x_max^4 Var(ln mu_hat) -> 240/27 = 8.889')
    for xm in (0.02, 0.05, 0.1):
        x = np.linspace(xm / 4000, xm, 4000)
        G = g(x)
        v = np.linalg.inv(G.T @ G)[0, 0] * 4000 * xm ** 4
        vk = 1 / (G[:, 0] @ G[:, 0]) * 4000 * xm ** 4
        print(f'    x_max {xm:.2f}: joint {v:.3f} (theory 8.889)   stiffness known {vk:.3f} (theory 0.556)   ratio {v / vk:.2f} (theory 16)')
    print("\n[3] Against the Volvo patent's published CRLB ratios")
    target1, target2 = (0.7787 / 0.0471, 0.0593 / 0.0471)
    ok = False
    for design in ('uniform in slip', 'uniform in utilisation'):
        r1 = sd_joint(0.3, 10, design) / sd_joint(0.8, 10, design)
        r2 = sd_joint(0.5, 160, design) / sd_joint(0.8, 10, design)
        e1, e2 = (100 * (r1 / target1 - 1), 100 * (r2 / target2 - 1))
        ok |= abs(e1) < 15 and abs(e2) < 15
        print(f'    {design:<24s} sd(0.3)/sd(0.8) = {r1:6.2f} (patent {target1:.2f}, {e1:+.0f}%)   sd(0.5,n160)/sd(0.8,n10) = {r2:5.3f} (patent {target2:.3f}, {e2:+.0f}%)')
    print('=' * 90)
    print(f"T3(d) CHECK: {('published ratios reproduced within 15%' if ok else 'NOT reproduced')}")
    print('=' * 90)
    return 0 if ok else 1
if __name__ == '__main__':
    sys.exit(main())
