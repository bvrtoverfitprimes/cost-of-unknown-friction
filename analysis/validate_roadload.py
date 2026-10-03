import numpy as np
import pandas as pd
from pathlib import Path
LBF_TO_N = 4.4482216152605
MPH_TO_MS = 0.44704
LB_TO_KG = 0.45359237
G = 9.80665
R_AIR = 287.058
T_REF = 293.15
P_REF = 98210.0
RHO_REF = P_REF / (R_AIR * T_REF)
ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / 'data' / 'epa_testcar' / '13tstcar.csv'

def load_unique_vehicles() -> pd.DataFrame:
    df = pd.read_csv(CSV, low_memory=False)
    cols = {'Represented Test Veh Make': 'make', 'Represented Test Veh Model': 'model', 'Test Vehicle ID': 'vid', 'Test Veh Configuration #': 'cfg', 'Vehicle Type': 'vtype', 'Equivalent Test Weight (lbs.)': 'etw_lb', 'Rated Horsepower': 'hp', 'Drive System Description': 'drive', 'Test Veh Displacement (L)': 'disp_L', '# of Gears': 'gears', 'Axle Ratio': 'axle', 'N/V Ratio': 'nv', 'Target Coef A (lbf)': 'A_lbf', 'Target Coef B (lbf/mph)': 'B_lbfmph', 'Target Coef C (lbf/mph**2)': 'C_lbfmph2'}
    df = df[list(cols)].rename(columns=cols)
    df = df.dropna(subset=['A_lbf', 'B_lbfmph', 'C_lbfmph2', 'etw_lb'])
    df = df.drop_duplicates(subset=['vid', 'cfg', 'A_lbf', 'B_lbfmph', 'C_lbfmph2'])
    return df.reset_index(drop=True)

def to_si(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['A_N'] = out.A_lbf * LBF_TO_N
    out['B_Nspm'] = out.B_lbfmph * LBF_TO_N / MPH_TO_MS
    out['C_Ns2pm2'] = out.C_lbfmph2 * LBF_TO_N / MPH_TO_MS ** 2
    out['mass_kg'] = out.etw_lb * LB_TO_KG
    out['weight_N'] = out.mass_kg * G
    out['CdA_m2'] = 2.0 * out.C_Ns2pm2 / RHO_REF
    out['Crr_implied'] = out.A_N / out.weight_N
    return out

def report_range(name, series, lo, hi, unit=''):
    s = series.dropna()
    inside = ((s >= lo) & (s <= hi)).mean() * 100
    print(f'  {name}')
    print(f'    n={len(s)}  median={s.median():.4f}{unit}  IQR=[{s.quantile(0.25):.4f}, {s.quantile(0.75):.4f}]  p5={s.quantile(0.05):.4f}  p95={s.quantile(0.95):.4f}')
    print(f'    inside physically-expected band [{lo}, {hi}]{unit}: {inside:.1f}% of vehicles')
    return inside

def main():
    df = to_si(load_unique_vehicles())
    print('=' * 78)
    print('VALIDATION TEST 1 -- derived road-load model vs. EPA measured coastdown data')
    print('=' * 78)
    print(f'source: {CSV.relative_to(ROOT)}')
    print(f'unique tested vehicle configurations: {len(df)}')
    print(f'reference air density used (20 C, 98.21 kPa): rho = {RHO_REF:.5f} kg/m^3')
    print()
    print('-' * 78)
    print('TEST 1A: implied drag area C_D*A from the measured quadratic coefficient C')
    print('  model: C_SI = 0.5 * rho * C_D * A')
    print('  physically expected for road vehicles: ~0.4 m^2 (slippery compact)')
    print('                                          to ~2.2 m^2 (large pickup/van)')
    pass_a = report_range('implied C_D*A', df.CdA_m2, 0.4, 2.2, ' m^2')
    print()
    for vt, grp in df.groupby('vtype'):
        if len(grp) < 25:
            continue
        print(f'    by type: {vt:<28s} n={len(grp):4d}  median C_D*A = {grp.CdA_m2.median():.3f} m^2')
    print()
    print('-' * 78)
    print('TEST 1B: implied rolling resistance coefficient from the constant coefficient A')
    print('  model: A_SI = C_rr * m * g')
    print('  physically expected for passenger radial tires: 0.006 - 0.015')
    pass_b = report_range('implied C_rr', df.Crr_implied, 0.006, 0.015)
    print()
    print('-' * 78)
    print('TEST 1C  [STRUCTURAL -- CAN FAIL]: does A scale linearly with vehicle weight?')
    print('  model predicts:  A_SI = C_rr * (m*g), i.e. a straight line through the origin')
    print('                   whose slope IS the fleet-average rolling resistance coefficient.')
    x = df.weight_N.values
    y = df.A_N.values
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = (x[ok], y[ok])
    slope0 = float(x @ y / (x @ x))
    resid0 = y - slope0 * x
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2_through_origin = 1.0 - float((resid0 ** 2).sum()) / ss_tot
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    r2_free = 1.0 - float(((y - pred) ** 2).sum()) / ss_tot
    corr = float(np.corrcoef(x, y)[0, 1])
    print(f'    through-origin fit : C_rr = {slope0:.5f}          R^2 = {r2_through_origin:.4f}')
    print(f'    unconstrained fit  : slope = {slope:.5f}, intercept = {intercept:.1f} N, R^2 = {r2_free:.4f}')
    print(f'    Pearson correlation between A and vehicle weight: r = {corr:.4f}')
    print(f"    => implied fleet-average C_rr = {slope0:.5f}  ({('PLAUSIBLE' if 0.006 <= slope0 <= 0.015 else 'OUT OF EXPECTED RANGE')})")
    print()
    print('-' * 78)
    print('TEST 1D: is the linear-in-v term B small, as the model implicitly assumes?')
    print('  the model lumps all speed-linear loss into C_rr(v); if B were dominant,')
    print('  the two-term (constant + quadratic) decomposition would be inadequate.')
    v_ms = np.linspace(5, 45, 9)
    A_ = df.A_N.median()
    B_ = df.B_Nspm.median()
    C_ = df.C_Ns2pm2.median()
    print(f'    fleet-median coefficients: A={A_:.1f} N, B={B_:.3f} N/(m/s), C={C_:.4f} N/(m/s)^2')
    print(f"    {'v [m/s]':>8} {'v [mph]':>8} {'A [N]':>9} {'B*v [N]':>9} {'C*v^2 [N]':>10} {'B share':>8} {'total [N]':>10}")
    for v in v_ms:
        tot = A_ + B_ * v + C_ * v * v
        print(f'    {v:8.1f} {v / MPH_TO_MS:8.1f} {A_:9.1f} {B_ * v:9.1f} {C_ * v * v:10.1f} {100 * B_ * v / tot:7.1f}% {tot:10.1f}')
    print()
    print('-' * 78)
    print('TEST 1E: regime crossover -- at what speed does aero overtake rolling resistance?')
    print('  model: 0.5*rho*C_D*A*v^2 = C_rr*m*g  =>  v_cross = sqrt(A_SI / C_SI)')
    vx = np.sqrt(df.A_N / df.C_Ns2pm2)
    print(f'    median crossover speed: {vx.median():.1f} m/s = {vx.median() / MPH_TO_MS:.1f} mph')
    print(f'    IQR: [{vx.quantile(0.25) / MPH_TO_MS:.1f}, {vx.quantile(0.75) / MPH_TO_MS:.1f}] mph')
    print("    (this is a falsifiable prediction of the model's regime structure:")
    print('     below this speed a vehicle is rolling-resistance dominated, above it aero dominated)')
    print()
    df.to_csv(ROOT / 'analysis' / 'epa_roadload_derived.csv', index=False)
    print('-' * 78)
    print('wrote analysis/epa_roadload_derived.csv')
    print()
    print('VERDICT')
    print(f'  implied C_D*A physically plausible for {pass_a:.1f}% of vehicles')
    print(f'  implied C_rr  physically plausible for {pass_b:.1f}% of vehicles')
    print(f'  A-vs-weight linearity R^2 (through origin) = {r2_through_origin:.4f}')
if __name__ == '__main__':
    main()
