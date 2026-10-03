import numpy as np
from dataclasses import dataclass, field
from typing import List
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
LBF_TO_N = 4.4482216152605
MPH_TO_MS = 0.44704
LB_TO_KG = 0.45359237
G = 9.80665
RHO_REF = 98210.0 / (287.058 * 293.15)
SIM_WINDOW_S = 600.0

def tire_radius(section_mm: float, aspect_pct: float, rim_in: float, loaded_deflection: float=0.97) -> float:
    r_unloaded = (rim_in * 25.4 / 2.0 + section_mm * aspect_pct / 100.0) / 1000.0
    return r_unloaded * loaded_deflection

@dataclass
class Vehicle:
    name: str
    A_lbf: float
    B_lbfmph: float
    C_lbfmph2: float
    axle_ratio: float
    nv_rpm_per_mph: float
    mass_curb_kg: float
    P_max_W: float
    T_max_Nm: float
    r_wheel: float
    gear_ratios: List[float]
    redline_rpm: float
    wheelbase_m: float
    cg_height_m: float
    front_weight_frac: float
    drive: str
    mu: float
    eta_d: float = 0.9
    I_wheels: float = 4.0
    I_rotor: float = 0.1
    shift_time_s: float = 0.0
    driver_kg: float = 75.0
    target_0_60_s: float = None
    target_source: str = ''

    @property
    def mass(self):
        return self.mass_curb_kg + self.driver_kg

    def total_ratio(self, g_idx):
        return self.gear_ratios[g_idx] * self.axle_ratio

    def implied_nv_top(self):
        G_top = self.total_ratio(len(self.gear_ratios) - 1)
        return G_top * MPH_TO_MS * 60.0 / (2.0 * np.pi * self.r_wheel)

    def m_eff(self, g_idx):
        r = self.r_wheel
        return self.mass + self.I_wheels / r ** 2 + self.I_rotor * self.total_ratio(g_idx) ** 2 / r ** 2

    def road_load_N(self, v):
        A = self.A_lbf * LBF_TO_N
        B = self.B_lbfmph * LBF_TO_N / MPH_TO_MS
        C = self.C_lbfmph2 * LBF_TO_N / MPH_TO_MS ** 2
        return A + B * v + C * v * v

    def engine_torque(self, omega_rad_s):
        if omega_rad_s <= 1e-06:
            return self.T_max_Nm
        return min(self.T_max_Nm, self.P_max_W / omega_rad_s)

    def F_PT_gear(self, v, g_idx):
        G_tot = self.total_ratio(g_idx)
        omega = G_tot * v / self.r_wheel
        if omega > self.redline_rpm * 2 * np.pi / 60.0:
            return None
        return self.eta_d * G_tot * self.engine_torque(omega) / self.r_wheel

    def F_PT(self, v):
        best, best_g = (-np.inf, len(self.gear_ratios) - 1)
        for g in range(len(self.gear_ratios)):
            f = self.F_PT_gear(v, g)
            if f is not None and f > best:
                best, best_g = (f, g)
        if not np.isfinite(best):
            return (0.0, len(self.gear_ratios) - 1)
        return (best, best_g)

    def traction_limit_N(self, a_x):
        W = self.mass * G
        transfer = self.mass * a_x * self.cg_height_m / self.wheelbase_m
        if self.drive == 'RWD':
            N_driven = (1.0 - self.front_weight_frac) * W + transfer
        elif self.drive == 'FWD':
            N_driven = self.front_weight_frac * W - transfer
        else:
            N_driven = W
        return self.mu * max(N_driven, 0.0)

    def a_max_fixedpoint(self, v):
        F_res = self.road_load_N(v)
        F_pt, gear = self.F_PT(v)
        m_eff = self.m_eff(gear)
        a_pt = (F_pt - F_res) / m_eff
        if self.traction_limit_N(a_pt) >= F_pt:
            return (a_pt, 'powertrain', gear)
        W = self.mass * G
        k = self.mass * self.cg_height_m / self.wheelbase_m
        if self.drive == 'RWD':
            num = self.mu * (1.0 - self.front_weight_frac) * W - F_res
            den = m_eff - self.mu * k
        elif self.drive == 'FWD':
            num = self.mu * self.front_weight_frac * W - F_res
            den = m_eff + self.mu * k
        else:
            num = self.mu * W - F_res
            den = m_eff
        if den <= 0:
            raise ValueError(f'{self.name}: non-physical denominator (mu*h/L too large)')
        a_tr = num / den

        def Psi(a):
            return m_eff * a - min(F_pt, self.traction_limit_N(a)) + F_res
        try:
            a_num = brentq(Psi, -40.0, 40.0, xtol=1e-11)
            if abs(a_num - a_tr) > 1e-06:
                raise AssertionError(f'{self.name} v={v:.2f}: closed form {a_tr:.8f} != numerical {a_num:.8f}')
        except ValueError:
            pass
        return (a_tr, 'traction', gear)

    def a_max_in_gear(self, v, gear):
        F_res = self.road_load_N(v)
        F_pt = self.F_PT_gear(v, gear)
        if F_pt is None:
            F_pt = 0.0
        m_eff = self.m_eff(gear)
        a_pt = (F_pt - F_res) / m_eff
        if self.traction_limit_N(a_pt) >= F_pt:
            return (a_pt, 'powertrain')
        W = self.mass * G
        k = self.mass * self.cg_height_m / self.wheelbase_m
        if self.drive == 'RWD':
            num, den = (self.mu * (1 - self.front_weight_frac) * W - F_res, m_eff - self.mu * k)
        elif self.drive == 'FWD':
            num, den = (self.mu * self.front_weight_frac * W - F_res, m_eff + self.mu * k)
        else:
            num, den = (self.mu * W - F_res, m_eff)
        return (num / den, 'traction')

    def simulate_0_60(self):
        v_target = 60.0 * MPH_TO_MS
        t, v, gear = (0.0, 0.0, 0)
        dt = 0.0005
        w_redline = self.redline_rpm * 2 * np.pi / 60.0
        trace, shifts = ([], 0)
        dist = 0.0
        t_rollout = None
        while v < v_target and t < SIM_WINDOW_S:
            omega = self.total_ratio(gear) * v / self.r_wheel
            if omega >= w_redline and gear < len(self.gear_ratios) - 1:
                gear += 1
                shifts += 1
                for _ in range(int(self.shift_time_s / dt)):
                    v = max(v - self.road_load_N(v) / self.m_eff(gear) * dt, 0.0)
                    t += dt
                continue
            a, regime = self.a_max_in_gear(v, gear)
            v += max(a, 1e-06) * dt
            dist += v * dt
            t += dt
            if t_rollout is None and dist >= 0.3048:
                t_rollout = t
            trace.append((t, v, a, regime, gear))
        if v < v_target:
            return (None, None, trace, shifts)
        return (t, t - (t_rollout or 0.0), trace, shifts)

    def regime_map(self, speeds_mph):
        rows = []
        for vm in speeds_mph:
            v = vm * MPH_TO_MS
            a, regime, gear = self.a_max_fixedpoint(v)
            fpt, _ = self.F_PT(v)
            rows.append((vm, a, regime, gear + 1, fpt, self.traction_limit_N(a), self.road_load_N(v)))
        return rows
VEHICLES = [Vehicle(name='2013 Nissan Leaf (BEV, single-speed)', A_lbf=29.97, B_lbfmph=0.0713, C_lbfmph2=0.02206, axle_ratio=7.9377, nv_rpm_per_mph=114.7, mass_curb_kg=1477.0, P_max_W=80000.0, T_max_Nm=254.0, r_wheel=tire_radius(205, 55, 16), gear_ratios=[1.0], redline_rpm=10390.0, wheelbase_m=2.7, cg_height_m=0.52, front_weight_frac=0.57, drive='FWD', mu=0.9, I_wheels=3.6, I_rotor=0.05, shift_time_s=0.0, target_0_60_s=9.9, target_source='EV Specifications (method not stated)'), Vehicle(name='2013 Honda Civic 1.8 (FWD, 5MT)', A_lbf=20.06, B_lbfmph=0.1822, C_lbfmph2=0.01736, axle_ratio=4.29, nv_rpm_per_mph=43.7, mass_curb_kg=1243.0, P_max_W=104400.0, T_max_Nm=174.0, r_wheel=tire_radius(195, 65, 15), gear_ratios=[3.143, 1.87, 1.235, 0.949, 0.727], redline_rpm=6500.0, wheelbase_m=2.67, cg_height_m=0.55, front_weight_frac=0.61, drive='FWD', mu=0.85, I_wheels=3.2, I_rotor=0.14, shift_time_s=0.45, target_0_60_s=9.5, target_source='MotorWeek road test (EX, 5-speed automatic)'), Vehicle(name='2013 Corvette ZR1 (RWD, 638 hp)', A_lbf=35.12, B_lbfmph=0.4778, C_lbfmph2=0.0167, axle_ratio=3.42, nv_rpm_per_mph=22.3, mass_curb_kg=1521.0, P_max_W=475800.0, T_max_Nm=819.0, r_wheel=tire_radius(335, 25, 20), gear_ratios=[2.29, 1.61, 1.21, 1.0, 0.82, 0.68], redline_rpm=6600.0, wheelbase_m=2.685, cg_height_m=0.45, front_weight_frac=0.51, drive='RWD', mu=1.35, I_wheels=5.0, I_rotor=0.25, shift_time_s=0.35, target_0_60_s=3.3, target_source='Motor Trend instrumented test (one-foot rollout)')]

def main():
    print('=' * 88)
    print('VALIDATION TEST 2 -- full acceleration envelope vs. published 0-60 mph times')
    print('=' * 88)
    print('road load A,B,C = EPA J2263 MEASURED values for each vehicle (not fitted by us)')
    print()
    print('-' * 88)
    print("CONSISTENCY CHECK: predict EPA's measured N/V (rpm per mph) from tire size + top gear")
    print('  this is a genuine out-of-sample check on r_wheel and the gear ratios we supplied')
    print(f"  {'vehicle':<40} {'N/V pred':>9} {'N/V EPA':>9} {'err':>8}")
    for v in VEHICLES:
        pred = v.implied_nv_top()
        err = 100 * (pred - v.nv_rpm_per_mph) / v.nv_rpm_per_mph
        print(f'  {v.name:<40} {pred:9.1f} {v.nv_rpm_per_mph:9.1f} {err:+7.1f}%')
    print()
    results = []
    for veh in VEHICLES:
        print('-' * 88)
        print(f'{veh.name}')
        print(f'  r_wheel = {veh.r_wheel:.4f} m (from tire size)')
        print(f'  mass (curb+driver) = {veh.mass:.0f} kg')
        print(f'  m_eff: 1st gear {veh.m_eff(0):.0f} kg (+{100 * (veh.m_eff(0) / veh.mass - 1):.1f}%), top gear {veh.m_eff(len(veh.gear_ratios) - 1):.0f} kg (+{100 * (veh.m_eff(len(veh.gear_ratios) - 1) / veh.mass - 1):.1f}%)')
        print(f'  drive={veh.drive}  mu={veh.mu}  h/L={veh.cg_height_m / veh.wheelbase_m:.3f}  shift={veh.shift_time_s}s')
        t60, t60_ro, trace, shifts = veh.simulate_0_60()
        if t60 is None:
            print('  !! never reached 60 mph')
            continue
        err = 100.0 * (t60_ro - veh.target_0_60_s) / veh.target_0_60_s
        err_raw = 100.0 * (t60 - veh.target_0_60_s) / veh.target_0_60_s
        print()
        print(f'  PREDICTED 0-60 (standing start) : {t60:.2f} s   ({err_raw:+.1f}%)')
        print(f'  PREDICTED 0-60 (1-ft rollout)   : {t60_ro:.2f} s   <-- US test convention')
        print(f'  PUBLISHED 0-60                  : {veh.target_0_60_s:.2f} s   [{veh.target_source}]')
        print(f'  ERROR (vs rollout convention)   : {err:+.1f} %')
        print()
        print(f"  {'mph':>5} {'a [m/s2]':>9} {'a [g]':>6} {'regime':>11} {'gear':>5} {'F_PT [N]':>9} {'F_TR [N]':>9} {'F_res [N]':>9}")
        for vm, a, regime, gear, fpt, ftr, fres in veh.regime_map([0, 10, 20, 30, 40, 50, 60]):
            print(f'  {vm:5.0f} {a:9.3f} {a / G:6.3f} {regime:>11} {gear:5d} {fpt:9.0f} {ftr:9.0f} {fres:9.0f}')
        frac_tr = np.mean([1.0 if r[3] == 'traction' else 0.0 for r in trace])
        print(f'  fraction of the 0-60 run spent TRACTION-limited: {100 * frac_tr:.0f}%')
        print(f'  upshifts during the run: {shifts}')
        results.append((veh.name, t60_ro, veh.target_0_60_s, err))
        print()
    print('=' * 88)
    print('SUMMARY')
    print(f"  {'vehicle':<40} {'pred':>7} {'pub':>7} {'err':>8}")
    for name, t60, tgt, err in results:
        print(f'  {name:<40} {t60:7.2f} {tgt:7.2f} {err:+7.1f}%')
    errs = np.array([abs(r[3]) for r in results])
    print()
    print(f'  mean |error| = {errs.mean():.1f} %   max |error| = {errs.max():.1f} %')
if __name__ == '__main__':
    main()
