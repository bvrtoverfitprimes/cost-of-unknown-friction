import numpy as np
R_DRY = 287.058
R_VAPOR = 461.495
G0 = 9.80665
M_AIR = 0.0289644
R_UNIV = 8.31446
T0_ISA = 288.15
P0_ISA = 101325.0
LAPSE = 0.0065

def saturation_vapor_pressure(T_K):
    Tc = T_K - 273.15
    return 611.21 * np.exp((18.678 - Tc / 234.5) * (Tc / (257.14 + Tc)))

def air_density(T_K, p_Pa, RH=0.0):
    p_v = RH * saturation_vapor_pressure(T_K)
    p_d = p_Pa - p_v
    return p_d / (R_DRY * T_K) + p_v / (R_VAPOR * T_K)

def pressure_at_altitude(h_m, T0=T0_ISA, p0=P0_ISA):
    return p0 * (1.0 - LAPSE * h_m / T0) ** (G0 * M_AIR / (R_UNIV * LAPSE))

def temperature_at_altitude(h_m, T0=T0_ISA):
    return T0 - LAPSE * h_m

def sae_j1349_correction(T_K, p_Pa, RH=0.0):
    p_v = RH * saturation_vapor_pressure(T_K)
    p_d_mbar = (p_Pa - p_v) / 100.0
    cf = 1.18 * (990.0 / p_d_mbar) * np.sqrt(T_K / 298.0) - 0.18
    return cf

def power_multiplier(T_K, p_Pa, RH=0.0, forced_induction=False, fi_recovery=0.75):
    na_mult = 1.0 / sae_j1349_correction(T_K, p_Pa, RH)
    if not forced_induction:
        return na_mult
    return na_mult + fi_recovery * (1.0 - na_mult)

def hydroplaning_speed(p_tire_kPa):
    p_psi = p_tire_kPa * 0.1450377
    v_knots = 9.0 * np.sqrt(p_psi)
    return v_knots * 0.514444

def speed_constant_Sp(MPD_mm):
    return 14.2 + 89.7 * MPD_mm

def mu_ifi(F60, v, MPD_mm=0.8):
    S_kmh = v * 3.6
    Sp = speed_constant_Sp(MPD_mm)
    return F60 * np.exp((60.0 - S_kmh) / Sp)
PEAK_TO_SLIDE_RATIO = 1.25
KAPPA_PEAK_WET = 0.1

def mu_wet(mu_dry, v, water_film_m, kappa_peak=KAPPA_PEAK_WET, **kw):
    return mu_wet_locked(mu_dry, v, water_film_m, slip_speed=kappa_peak * v, **kw)

def mu_wet_locked(mu_dry, v, water_film_m, p_tire_kPa=220.0, mu_hydroplane=0.08, MPD_mm=0.8, reference_film_m=0.0005, slip_speed=None):
    if water_film_m <= 0.0:
        return mu_dry
    S = v if slip_speed is None else slip_speed
    mu_speed = mu_ifi(mu_dry * _wet_dry_ratio_at_60(MPD_mm), S, MPD_mm)
    excess = max(0.0, water_film_m - reference_film_m)
    beta = 313.0
    mu = mu_speed * np.exp(-beta * excess)
    v_p = hydroplaning_speed(p_tire_kPa)
    if water_film_m >= 0.0025:
        if v >= v_p:
            return mu_hydroplane
        if v > 0.8 * v_p:
            w = (v - 0.8 * v_p) / (0.2 * v_p)
            mu = (1 - w) * mu + w * mu_hydroplane
    return float(max(min(mu, WET_CEILING * mu_dry), mu_hydroplane))
WET_DRY_BAND = (0.71, 0.87)
WET_CEILING = 0.5 * (WET_DRY_BAND[0] + WET_DRY_BAND[1])

def _wet_dry_ratio_at_60(MPD_mm):
    return float(np.clip(0.55 + 0.22 * MPD_mm, 0.45, 0.85))
TIRE_TEMP_WINDOWS = {'street_allseason': dict(T_opt_C=45.0, width_C=90.0, T_ref_C=25.0), 'street_summer': dict(T_opt_C=55.0, width_C=75.0, T_ref_C=30.0), 'extreme_summer': dict(T_opt_C=70.0, width_C=60.0, T_ref_C=45.0), 'racing_slick': dict(T_opt_C=90.0, width_C=35.0, T_ref_C=85.0)}

def mu_temperature(mu_ref, T_tire_C, tire_type='street_summer', T_ref_C=None, T_opt_C=None, width_C=None):
    w = TIRE_TEMP_WINDOWS.get(tire_type, TIRE_TEMP_WINDOWS['street_summer'])
    T_opt = w['T_opt_C'] if T_opt_C is None else T_opt_C
    width = w['width_C'] if width_C is None else width_C
    T_ref = w['T_ref_C'] if T_ref_C is None else T_ref_C

    def shape(T):
        return np.exp(-0.5 * ((T - T_opt) / width) ** 2)
    return mu_ref * shape(T_tire_C) / shape(T_ref)
SKID_MEASURED_SNOW_ICE = {'ice': (0.1, 0.2), 'new_snow': (0.2, 0.25), 'old_snow': (0.25, 0.3), 'refrozen_snow': (0.3, 0.4), 'chloride_treated_snow': (0.35, 0.45), 'sand_treated_snow': (0.3, 0.4), 'chloride_sand_mixture': (0.3, 0.5)}
SURFACE_STATES = {'dry_asphalt': (0.8, 1.2), 'dry_concrete': (0.8, 1.1), 'wet_asphalt': (0.5, 0.75), 'standing_water': (0.25, 0.45), 'gravel': (0.35, 0.55), 'packed_snow': tuple((PEAK_TO_SLIDE_RATIO * x for x in SKID_MEASURED_SNOW_ICE['old_snow'])), 'loose_snow': tuple((PEAK_TO_SLIDE_RATIO * x for x in SKID_MEASURED_SNOW_ICE['new_snow'])), 'refrozen_snow': tuple((PEAK_TO_SLIDE_RATIO * x for x in SKID_MEASURED_SNOW_ICE['refrozen_snow'])), 'ice_near_0C': (0.05, 0.13), 'ice_cold': tuple((PEAK_TO_SLIDE_RATIO * x for x in SKID_MEASURED_SNOW_ICE['ice']))}

def mu_surface(state, quantile=0.5):
    lo, hi = SURFACE_STATES[state]
    return lo + quantile * (hi - lo)

def wind_relative(v_vehicle, heading_rad, wind_speed, wind_from_rad):
    v_vec = np.array([v_vehicle * np.cos(heading_rad), v_vehicle * np.sin(heading_rad)])
    w_vec = np.array([wind_speed * np.cos(wind_from_rad + np.pi), wind_speed * np.sin(wind_from_rad + np.pi)])
    rel = v_vec - w_vec
    V_rel = float(np.linalg.norm(rel))
    if V_rel < 1e-09:
        return (0.0, 0.0)
    fwd = np.array([np.cos(heading_rad), np.sin(heading_rad)])
    cos_b = np.clip(rel @ fwd / V_rel, -1.0, 1.0)
    cross = fwd[0] * rel[1] - fwd[1] * rel[0]
    beta = float(np.sign(cross) * np.arccos(cos_b))
    return (V_rel, beta)

def _selftest():
    print('=' * 80)
    print('ENVIRONMENTAL PHYSICS -- self-tests against known reference values')
    print('=' * 80)
    ok = []
    rho = air_density(288.15, 101325.0, RH=0.0)
    print(f'\n[EXACT] ISA sea level, 15 C, dry: rho = {rho:.5f} kg/m^3  (reference 1.225)')
    ok.append(abs(rho - 1.225) < 0.001)
    print(f"        {('PASS' if ok[-1] else 'FAIL')}")
    rho_dry = air_density(303.15, 101325.0, RH=0.0)
    rho_wet = air_density(303.15, 101325.0, RH=1.0)
    print(f'\n[EXACT] 30 C, 1 atm:  dry rho = {rho_dry:.5f},  100% RH rho = {rho_wet:.5f}')
    print(f"        humid air is {100 * (1 - rho_wet / rho_dry):.2f}% LESS dense -- {('PASS' if rho_wet < rho_dry else 'FAIL')}")
    ok.append(rho_wet < rho_dry)
    rho_epa = air_density(293.15, 98210.0, RH=0.0)
    print(f'\n[EXACT] EPA/J2263 reference (20 C, 98.21 kPa, dry): rho = {rho_epa:.5f} kg/m^3')
    print(f'        (this is the value used in validate_roadload.py)')
    print('\n[STANDARD] altitude effects (ISA):')
    print(f"  {'site':<22} {'alt [m]':>8} {'p [kPa]':>9} {'T [C]':>7} {'rho':>8} {'NA power':>9} {'turbo':>7}")
    for name, h in [('sea level', 0), ('Denver CO', 1609), ('Leadville CO', 3094), ('Mexico City', 2240), ('Pikes Peak summit', 4302)]:
        p = pressure_at_altitude(h)
        T = temperature_at_altitude(h)
        r = air_density(T, p)
        na = power_multiplier(T, p, forced_induction=False)
        fi = power_multiplier(T, p, forced_induction=True)
        print(f'  {name:<22} {h:8.0f} {p / 1000:9.2f} {T - 273.15:7.1f} {r:8.4f} {100 * na:8.1f}% {100 * fi:6.1f}%')
    na_denver = power_multiplier(temperature_at_altitude(1609), pressure_at_altitude(1609))
    print(f'\n  Denver NA power retention = {100 * na_denver:.1f}% (published rule of thumb: ~3% loss per 1000 ft => ~{100 - 3 * 5.28:.0f}%)')
    ok.append(0.78 < na_denver < 0.9)
    print(f"  {('PASS' if ok[-1] else 'FAIL')}")
    print('\n[STANDARD] Horne/NASA hydroplaning speed:')
    for p_kpa in [200, 220, 250, 300]:
        vp = hydroplaning_speed(p_kpa)
        print(f'  tire {p_kpa:3d} kPa ({p_kpa * 0.145:4.1f} psi): v_hydroplane = {vp:5.1f} m/s = {vp * 2.23694:5.1f} mph')
    print('\n[STANDARD] ASTM E1960 / PIARC speed constant Sp = 14.2 + 89.7*MPD:')
    for mpd in [0.2, 0.5, 0.8, 1.5, 2.5]:
        sp = speed_constant_Sp(mpd)
        print(f'  MPD {mpd:4.1f} mm -> Sp = {sp:6.1f} km/h   (friction at 100 km/h is {100 * np.exp((60 - 100) / sp):5.1f}% of its 60 km/h value)')
    ok.append(abs(speed_constant_Sp(0.8) - 85.96) < 0.01)
    print(f"  formula check: {('PASS' if ok[-1] else 'FAIL')}")
    for label, fn in (('LOCKED-WHEEL (skid trailer)', mu_wet_locked), ('PEAK, rolling tyre (what the envelope uses)', mu_wet)):
        print(f'\n[STANDARD speed law + EMPIRICAL depth] {label}: mu_wet/mu_dry')
        print(f"  {'water film':>12} " + ''.join((f'{v * 3.6:>8.0f}' for v in [8.33, 13.89, 19.44, 27.78, 33.33])))
        print(f"  {'[mm]':>12} " + ''.join((f"{'km/h':>8}" for _ in range(5))))
        for h_mm in [0.1, 0.5, 1.0, 2.0, 5.0]:
            row = ''.join((f'{fn(1.0, v, h_mm / 1000):8.3f}' for v in [8.33, 13.89, 19.44, 27.78, 33.33]))
            print(f'  {h_mm:12.1f} {row}')
    cal = mu_wet_locked(1.0, 27.78, 0.001)
    print(f'\n  calibration check (locked wheel): 1mm @ 100 km/h -> {cal:.3f} (target 0.39, [WetFriction])')
    ok.append(abs(cal - 0.39) < 0.03)
    print(f"  {('PASS' if ok[-1] else 'FAIL')}")
    order_ok = all((mu_wet(1.0, v, h / 1000) >= mu_wet_locked(1.0, v, h / 1000) - 1e-12 for v in [8.33, 13.89, 19.44, 27.78, 33.33] for h in [0.1, 0.5, 1.0, 2.0, 5.0]))
    ok.append(order_ok)
    print(f"  peak >= locked everywhere: {('PASS' if order_ok else 'FAIL')}")
    print('\n[MEASURED] TRB SR-115 Table 1 skid coefficients (30-40 km/h) -> peak mu:')
    print(f"  {'condition':<26} {'skid (measured)':>18} {'peak mu (x1.25)':>17}")
    for k, (lo, hi) in SKID_MEASURED_SNOW_ICE.items():
        print(f"  {k:<26} {f'{lo:.2f} - {hi:.2f}':>18} {f'{PEAK_TO_SLIDE_RATIO * lo:.2f} - {PEAK_TO_SLIDE_RATIO * hi:.2f}':>17}")
    print('\n[INDEPENDENT CHECK] Edmonton winter field trials (CJCE 2016) report friction')
    print('  reductions vs bare dry asphalt: ice -55%, light snow -69%, moderate -75%,')
    print('  heavy snow -81%. Compare against our surface table:')
    dry = mu_surface('dry_asphalt', 0.5)
    checks = [('ice', 'ice_cold', 55), ('light snow', 'loose_snow', 69), ('moderate snow', 'packed_snow', 75)]
    devs = []
    for label, key, target_pct in checks:
        m = mu_surface(key, 0.5)
        red = 100 * (1 - m / dry)
        devs.append(abs(red - target_pct))
        print(f'  {label:<15} ours: mu={m:5.3f} -> -{red:4.1f}%   Edmonton: -{target_pct}%   delta {red - target_pct:+5.1f} pts')
    print(f'  mean deviation {np.mean(devs):.1f} percentage points (different tires/surfaces/methods, so exact agreement is not expected)')
    ok.append(np.mean(devs) < 20)
    print(f"  {('PASS (same order)' if ok[-1] else 'FAIL')}")
    print('\n[EXACT] relative wind (vehicle at 30 m/s heading east):')
    for nm, wd, ws in [('no wind', 0, 0), ('10 m/s headwind', 0, 10), ('10 m/s tailwind', np.pi, 10), ('10 m/s crosswind', np.pi / 2, 10)]:
        V, b = wind_relative(30.0, 0.0, ws, wd)
        print(f'  {nm:<20} V_rel = {V:5.1f} m/s   beta = {np.degrees(b):+6.1f} deg')
    print('\n' + '=' * 80)
    print(f'SELF-TESTS: {sum(ok)}/{len(ok)} passed')
    return all(ok)
if __name__ == '__main__':
    _selftest()
