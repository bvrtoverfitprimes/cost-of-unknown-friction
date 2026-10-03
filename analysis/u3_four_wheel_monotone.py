from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from four_wheel_equilibrium import FourWheelVehicle
from magic_formula import MFTire
from validate_acceleration import VEHICLES
NB = 72

def radial_boundary(tire, Fz):
    ks = np.linspace(0.0, 0.8, 161)
    als = np.radians(np.linspace(0.0, 30.0, 121))
    FX, FY = ([], [])
    for a in als:
        for k in ks:
            FX.append(tire.Fx_combined(k, a, Fz))
            FY.append(abs(tire.Fy_combined(k, a, Fz)))
    FX, FY = (np.array(FX), np.array(FY))
    th = np.arctan2(FY, FX)
    r = np.hypot(FX, FY)
    sel = (th >= 0) & (th <= np.pi / 2)
    b = np.minimum((th[sel] / (np.pi / 2) * NB).astype(int), NB - 1)
    rb = np.zeros(NB)
    np.maximum.at(rb, b, r[sel])
    return rb

def main() -> int:
    print('=' * 96)
    print('U3 -- FOUR-WHEEL COMBINED-SLIP ENVELOPE: MONOTONICITY IN FRICTION')
    print('=' * 96)
    ok = True
    record = {'nest': [], 'envelope': [], 'scales': [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]}
    print('\n[1] Nestedness of the PAC2002 combined-slip achievable set under lambda_mu')
    lams = (0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 1.8)
    worst = 0.0
    for fz_rel in (0.5, 1.0, 1.5):
        rbs = []
        for lam in lams:
            t = MFTire()
            t.lambda_mux = t.lambda_muy = lam
            rbs.append(radial_boundary(t, fz_rel * t.Fz0))
        t = MFTire()
        t.lambda_mux = t.lambda_muy = 0.25
        low = radial_boundary(t, fz_rel * t.Fz0)
        m = (low > 0) & (rbs[0] > 0)
        below = float(np.max((low[m] - rbs[0][m]) / rbs[0][m]))
        viol = 0.0
        for i in range(len(lams) - 1):
            lo, hi = (rbs[i], rbs[i + 1])
            m = (lo > 0) & (hi > 0)
            viol = max(viol, float(np.max((lo[m] - hi[m]) / hi[m])))
        worst = max(worst, viol)
        growth = rbs[-1][rbs[0] > 0] / rbs[0][rbs[0] > 0]
        record['nest'].append(dict(fz_rel=fz_rel, worst_violation=float(viol), below=below, growth_min=float(growth.min()), growth_max=float(growth.max())))
        print(f'    Fz = {fz_rel:.1f} Fz0: largest outward violation {100 * max(viol, 0):.2f}% (radius ratio lambda 1.8 / 0.3 over directions: {growth.min():.2f}-{growth.max():.2f}); lambda 0.25 vs 0.3, outside the range used: {100 * below:+.2f}%')
    ok &= worst < 0.01
    print(f'    nested to within {100 * max(worst, 0):.2f}% (angular binning tolerance 1%)')
    print('\n[2] Four-wheel equilibrium envelope a_x vs friction scale, ZR1 at 20 m/s')
    base = VEHICLES[2]
    scales = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1)
    print(f"    {'R [m]':>7s} " + ''.join((f"{'x' + str(s):>9s}" for s in scales)) + '   monotone')
    for R in (1000000000.0, 300.0, 150.0, 100.0, 70.0):
        row, z = ([], None)
        for s in scales:
            fw = FourWheelVehicle(base)
            fw.tire.lambda_mux *= s
            fw.tire.lambda_muy *= s
            sol = fw.solve_equilibrium(20.0, 1.0 / R, a_x_guess=3.0, z_init=z)
            if sol['converged']:
                z = sol['z']
                row.append(sol['a_x'])
            else:
                row.append(np.nan)
        arr = np.array(row)
        good = arr[~np.isnan(arr)]
        mono = bool(np.all(np.diff(good) >= -0.001))
        ok &= mono
        record['envelope'].append(dict(R=float(R), a=[None if np.isnan(x) else float(x) for x in row], mono=mono))
        print(f'    {(R if R < 100000000.0 else np.inf):7.0f} ' + ''.join((f'{x:9.3f}' if not np.isnan(x) else f"{'n/c':>9s}" for x in row)) + f'   {mono}')
    print('    (n/c: lateral demand exceeds the scaled friction -- no equilibrium exists)')
    print('=' * 96)
    print(f"U3 CHECK: {('all hold' if ok else 'A CHECK FAILED')}")
    print('=' * 96)
    from well_posedness import save_record
    save_record('u3_four_wheel', record)
    return 0 if ok else 1
if __name__ == '__main__':
    sys.exit(main())
