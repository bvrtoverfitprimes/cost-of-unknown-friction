from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import environment as envmod
from pac_governed import PACVehicle
from t2_monotonicity import Env
from validate_acceleration import MPH_TO_MS, VEHICLES
NAMES = ['mu', 'wetness', 'wet/dry', 'T', 'altitude', 'C_rr', 'C_dA', 'headwind', 'payload', 'grade']
LO = np.array([0.81, 0.0, 0.71, -10.0, 0.0, 0.8, 0.9, -5.0, 0.0, 0.0])
HI = np.array([1.24, 1.0, 0.87, 40.0, 3000.0, 1.3, 1.1, 15.0, 300.0, 6.0])
RHO_REF = envmod.air_density(293.15, envmod.P0_ISA, 0.5)

def a_max(veh, v, law, pac, x):
    mu, wet, ceil, T, alt, crr, cda, hw, payload, grade = x
    if wet >= 0.5:
        envmod.WET_CEILING = ceil
        mu = mu * envmod.mu_wet(1.0, max(v, 1.0), (wet - 0.5) * 0.006)
    T_K = T + 273.15
    p = envmod.pressure_at_altitude(alt, T0=T_K + 0.0065 * alt)
    rho = envmod.air_density(T_K, p, 0.5) / RHO_REF
    ice = len(veh.gear_ratios) > 1
    P = envmod.power_multiplier(T_K, p, 0.5) / rho if ice else 1.0
    cda_eff = cda * ((v + hw) / v) ** 2
    m = (veh.mass + payload) / veh.mass
    e = Env(veh, v, law, mu=mu, m=m, crr=crr, cda=cda_eff, rho=rho, P=P, grade=np.arctan(grade / 100.0), pac=pac)
    return (e.a_max(), e.regime(e.a_max()))

def sobol(f, N, rng, lo=LO, hi=HI):
    d = len(lo)
    A = lo + (hi - lo) * rng.random((N, d))
    B = lo + (hi - lo) * rng.random((N, d))
    fA = np.array([f(r)[0] for r in A])
    fB = np.array([f(r)[0] for r in B])
    var = np.var(np.concatenate([fA, fB]))
    S, ST = (np.zeros(d), np.zeros(d))
    fABs = []
    for i in range(d):
        ABi = A.copy()
        ABi[:, i] = B[:, i]
        fAB = np.array([f(r)[0] for r in ABi])
        fABs.append(fAB)
        S[i] = np.mean(fB * (fAB - fA)) / var
        ST[i] = 0.5 * np.mean((fA - fAB) ** 2) / var
    boot = np.random.default_rng(12345)
    Sb, STb = (np.zeros((200, d)), np.zeros((200, d)))
    for b in range(200):
        k = boot.integers(0, N, N)
        vb = np.var(np.concatenate([fA[k], fB[k]]))
        for i in range(d):
            Sb[b, i] = np.mean(fB[k] * (fABs[i][k] - fA[k])) / vb
            STb[b, i] = 0.5 * np.mean((fA[k] - fABs[i][k]) ** 2) / vb
    regimes = [f(r)[1] for r in A[:min(N, 200)]]
    frac_tr = np.mean([r == 'traction' for r in regimes])
    return (S, ST, var, frac_tr, Sb.std(axis=0), STb.std(axis=0))

def main() -> int:
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    rng = np.random.default_rng(1)
    ceil0 = envmod.WET_CEILING
    print('=' * 100)
    print(f'T6 -- REGIME-RESOLVED SOBOL INDICES OF a_max  (N = {N} per matrix)')
    print('=' * 100)
    summary = {}
    for veh in VEHICLES:
        for law in ('ellipse', 'pac2002'):
            pac = PACVehicle(veh) if law == 'pac2002' else None
            res = {}
            for label, vm in (('10 mph', 10), ('60 mph', 60)):
                v = vm * MPH_TO_MS
                res[label] = sobol(lambda x: a_max(veh, v, law, pac, x), N, rng)

            def pooled(x):
                vm = 10 if x[-1] < 0.5 else 60
                return a_max(veh, vm * MPH_TO_MS, law, pac, x[:-1])
            r = sobol(pooled, N, rng, lo=np.append(LO, 0.0), hi=np.append(HI, 1.0))
            res['pooled'] = (r[0][:-1], r[1][:-1], r[2], r[3], r[4][:-1], r[5][:-1], r[1][-1], r[5][-1])
            envmod.WET_CEILING = ceil0
            summary[veh.name, law] = res
            print(f'\n{veh.name}   [{law}]')
            print(f"  {'input':<10s}" + ''.join((f'{lab:>20s}' for lab in res)))
            print(f"  {'':<10s}" + ''.join((f"{'S':>9s}{'ST':>11s}" for _ in res)))
            for i, nm in enumerate(NAMES):
                row = f'  {nm:<10s}'
                for lab in res:
                    S, ST = (res[lab][0], res[lab][1])
                    flag = '*' if ST[i] < 0.01 else ' '
                    row += f'{S[i]:9.3f}{ST[i]:10.3f}{flag}'
                print(row)
            print('  ' + ''.join((f"{'traction share':>12s} {res[l][3]:5.2f}  " for l in res)))
            print(f"  pooled: total index of SPEED itself = {res['pooled'][6]:.3f}")
    print('\n  * = total index below 0.01: the term can be dropped in that regime')
    print('=' * 100)
    record = {'N': N, 'names': NAMES, 'lo': LO.tolist(), 'hi': HI.tolist(), 'evaluations': int(len(summary) * (2 * (len(NAMES) + 2) + len(NAMES) + 3) * N), 'cases': [{'vehicle': k[0], 'law': k[1], **{lab: {'S': np.asarray(v[0]).tolist(), 'ST': np.asarray(v[1]).tolist(), 'var': float(v[2]), 'traction_share': float(v[3]), 'S_se': np.asarray(v[4]).tolist(), 'ST_se': np.asarray(v[5]).tolist(), **({'ST_speed': float(v[6]), 'ST_speed_se': float(v[7])} if lab == 'pooled' else {})} for lab, v in r.items()}} for k, r in summary.items()]}
    from well_posedness import save_record
    save_record('t6_sobol', record)
    np.save(Path(__file__).with_name('t6_sobol_results.npy'), {str(k): {l: (np.asarray(v[0]).tolist(), np.asarray(v[1]).tolist(), float(v[2]), float(v[3])) for l, v in r.items()} for k, r in summary.items()}, allow_pickle=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
