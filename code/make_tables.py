import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
OUT = ROOT / 'tables'
OUT.mkdir(exist_ok=True)


def tex_escape(s: str) -> str:
    return str(s).replace('&', '\\&').replace('%', '\\%').replace('_', '\\_').replace('#', '\\#')


def sci(x: float, digits: int=2) -> str:
    m, e = f'{x:.{digits}e}'.split('e')
    return f'{m}\\times10^{{{int(e)}}}'


def small(x: float) -> str:
    return '$0$' if x == 0 else f'${sci(x, 1)}$'


def thousands(n: int) -> str:
    return f'{n:,}'.replace(',', '{,}')


def nice_name(s: str) -> str:
    out = []
    for w in str(s).split():
        out.append(w.title() if len(w) > 3 and w.isalpha() else w)
    return ' '.join(out)


def write(name: str, body: str) -> None:
    if '@NFLEET@' in body:
        body = body.replace('@NFLEET@', str(len(load('t4_uncertainty')['ratios'])))
    (OUT / f'{name}.tex').write_text(body, encoding='utf-8')
    print(f'wrote tables/{name}.tex')


def load(name):
    import json
    return json.loads((DATA / f'{name}.json').read_text())


SHORT = {'2013 Nissan Leaf (BEV, single-speed)': 'Leaf', '2013 Honda Civic 1.8 (FWD, 5MT)': 'Civic', '2013 Corvette ZR1 (RWD, 638 hp)': 'ZR1'}


LAW = {'ellipse': 'ellipse', 'pac2002': 'PAC2002'}


def wellposed() -> None:
    d = load('well_posedness')
    rows = []
    for c in d['cases']:
        rows.append(f"{SHORT[c['vehicle']]} & {LAW[c['law']]} & ${c['mph']}$ & ${c['m_eff']:.0f}$ & ${c['t_sup']:.1f}$ & ${c['g']:.1f}$ & ${c['margin']:.1f}$ & ${c['max_slope']:.0f}$ & ${c['a_max']:.3f}$\\\\")
    lip = []
    for c in d['lipschitz']:
        lip.append(f"{SHORT[c['vehicle']]} & {LAW[c['law']]} & {('traction' if c['traction_binds'] else 'power')} & ${c['measured']:.4f}$ & ${c['local']:.4f}$ & ${c['bound']:.4f}$\\\\")
    body = "\\begin{table}[htbp]\n\\centering\\small\n\\caption{Check of Theorem~\\ref{thm:wellposed}. For each vehicle, tyre law, and speed, the effective mass $\\meff$, the largest slope $\\sup T'$ of the traction capability, and the load-transfer gain $\\beta=\\max(\\sup T',0)$ (all in kg), the margin $\\meff-\\beta$, the largest slope of the force margin $\\Phi$ on a grid of $4001$ accelerations (in N per m/s$^2$, negative when $\\Phi$ is strictly decreasing), and the envelope $\\amax$ (m/s$^2$). A negative envelope means that the vehicle cannot sustain that speed.}\\label{tab:app-wellposed}\n\\begin{tabular}{llrrrrrrr}\n\\toprule\nVehicle & Tyre law & mph & $\\meff$ & $\\sup T'$ & $\\beta$ & $\\meff-\\beta$ & $\\max\\Phi'$ & $\\amax$\\\\\n\\midrule\n" + '\n'.join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\n\\begin{table}[htbp]\n\\centering\\small\n\\caption{Check of the bound of Theorem~\\ref{thm:wellposed}\\ref{wp:lip} at $10$\\,mph. The measured value is $|\\partial\\amax/\\partial\\mu|$ from a relative change of $10^{-4}$ in friction, the local value is $|\\partial_\\mu\\Phi|/(\\meff-T'(\\amax))$ of equation~\\eqref{eq:derivative}, and the bound is $|\\partial_\\mu\\Phi|/(\\meff-\\beta)$, all at $\\amax$ and in m/s$^2$. The bound equals the local value when $T'(\\amax)=\\beta$.}\\label{tab:app-lipschitz}\n\\begin{tabular}{lllrrr}\n\\toprule\nVehicle & Tyre law & Binding & Measured & Local & Bound\\\\\n\\midrule\n" + '\n'.join(lip) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    write('wellposed', body)


D3NAME = {'2012 Nissan Leaf': '2012 Nissan Leaf', '2010 Toyota Prius': '2010 Toyota Prius', '2012 Ford Focus': '2012 Ford Focus', '2013 Ford Focus Electric': '2013 Ford Focus Electric'}


FNAME = {'mu': '$\\mu$', 'm': '$m$', 'C_rr': '$C_{rr}$', 'C_dA': '$C_DA$', 'grade': '$\\gamma$', 'wind': '$w$'}


def fisher() -> None:
    d = load('t3c_fisher')
    labels = ['Ordinary driving', 'Limit events at one acceleration', 'Limit events at two accelerations', 'Braking-limit events']
    rows = []
    for lab, ds in zip(labels, d['designs']):
        sv = ', '.join((f'${sci(x, 1)}$' if x > 0 else '$0$' for x in ds['singular']))
        rows.append(f"{lab} & ${ds['rows']}$ & ${ds['rank']}$ & {sv}\\\\")
    crb = []
    for c in d['crb']:
        crb.append(f"{FNAME[c['name']]} & ${d['nominal'][d['names'].index(c['name'])]:.4g}$ & ${100 * c['rel']:.1f}\\%$ & ${c['abs']:.3g}$\\\\")
    body = '\\begin{table}[htbp]\n\\centering\\small\n\\caption{Check of Theorem~\\ref{thm:jointid}. Every design contains $60$ balance samples at speeds drawn uniformly from $5$ to $30$\\,m/s and accelerations from $-1.5$ to $2.0$\\,m/s$^2$. The limit events are five samples at $3.0$\\,m/s$^2$ and $8$\\,m/s, plus, for the two-acceleration design, five at $1.5$\\,m/s$^2$ and $20$\\,m/s. The braking events are five at $-7$ and five at $-5$\\,m/s$^2$. Each limit event contributes a balance row and a limit row. The singular values are those of the Jacobian with each parameter scaled to its nominal value.}\\label{tab:app-fisher}\n\\begin{tabular}{>{\\raggedright\\arraybackslash}p{3.6cm}cc>{\\raggedright\\arraybackslash}p{5.6cm}}\n\\toprule\nDesign & Rows & Rank & Singular values\\\\\n\\midrule\n' + '\n'.join(rows) + f"\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n\n\\begin{{table}}[htbp]\n\\centering\\small\n\\caption{{Cram\\'er--Rao standard errors for the design with limit events at two accelerations, with force noise of ${d['sigma']:.0f}$\\,N on each of its $80$ rows. The nominal values are those of a Civic-like car, with grade in radians and wind in m/s.}}\\label{{tab:app-crb}}\n\\begin{{tabular}}{{lrrr}}\n\\toprule\nParameter & Nominal & Relative error & Absolute error\\\\\n\\midrule\n" + '\n'.join(crb) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    write('fisher', body)


def epa() -> None:
    import csv
    rows = list(csv.DictReader(open(DATA / 'epa_roadload_derived.csv', encoding='utf-8')))
    cda = np.array([float(r['CdA_m2']) for r in rows])
    crr = np.array([float(r['Crr_implied']) for r in rows])
    car = np.array([r['vtype'] == 'Car' for r in rows])
    truck = np.array([r['vtype'] == 'Truck' for r in rows])
    both = np.array([r['vtype'] == 'Both' for r in rows])
    A = np.array([float(r['A_N']) for r in rows])
    W = np.array([float(r['weight_N']) for r in rows])
    slope = float(np.sum(A * W) / np.sum(W * W))
    r_corr = float(np.corrcoef(A, W)[0, 1])
    ss_res = float(np.sum((A - slope * W) ** 2))
    ss_tot = float(np.sum((A - A.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot

    def q(x, nd):
        return ' & '.join((f'${v:.{nd}f}$' for v in np.percentile(x, [5, 25, 50, 75, 95])))
    in_cda = np.mean((cda >= 0.4) & (cda <= 2.2))
    in_crr = np.mean((crr >= 0.006) & (crr <= 0.015))
    body = f'\\begin{{table}}[htbp]\n\\centering\\small\n\\caption{{Road-load coefficients of the ${len(rows)}$ vehicle configurations of model year 2013 in the EPA test car list, inverted with $C=\\tfrac12\\rho C_DA$ and $A=C_{{rr}}mg$. The drag area lies in $[0.4,2.2]$\\,m$^2$ for ${100 * in_cda:.1f}\\%$ of them and the rolling resistance coefficient in $[0.006,0.015]$ for ${100 * in_crr:.1f}\\%$. The median drag area is ${np.median(cda[car]):.3f}$\\,m$^2$ for the ${car.sum()}$ configurations listed as cars, ${np.median(cda[truck]):.3f}$\\,m$^2$ for the ${truck.sum()}$ listed as trucks, and ${np.median(cda[both]):.3f}$\\,m$^2$ for the ${both.sum()}$ listed as both. A line through the origin fitted to the constant term $A$ against the weight $mg$ has slope ${slope:.5f}$, correlation $r={r_corr:.2f}$, and $R^2={r2:.2f}$.}}\\label{{tab:app-epa}}\n\\begin{{tabular}}{{lccccc}}\n\\toprule\nQuantity & $5\\%$ & $25\\%$ & Median & $75\\%$ & $95\\%$\\\\\n\\midrule\nDrag area $C_DA$ (m$^2$) & {q(cda, 3)}\\\\\nRolling resistance $C_{{rr}}$ & {q(crr, 4)}\\\\\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n'
    write('epa', body)
    print(f'  in_cda {in_cda:.4f} in_crr {in_crr:.4f} car {np.median(cda[car]):.3f} truck {np.median(cda[truck]):.3f} ratio {np.median(cda[truck]) / np.median(cda[car]):.3f} R2 {r2:.3f}')


PNAME = {'mu': 'Friction $\\mu$', 'crr': 'Rolling resistance $C_{rr}$', 'cda': 'Drag area $C_DA$', 'P': 'Power $P$'}


def signs() -> None:
    d = load('t2_monotonicity')
    mono = []

    def step(x):
        return '$0$' if abs(x) < 1e-12 else f'${sci(x)}$'
    arrow = {'non-decreasing': 'up', 'non-increasing': 'down'}
    for r in d['monotone']:
        mono.append(f"{PNAME[r['param']]} & {arrow[r['direction']]} & ${r['lo']:.1f}$--${r['hi']:.1f}$ & ${r['n']}$ & ${r['violations']}$ & {step(r['min_step'])}\\\\")
    g = d['grade']
    mono.append(f"Uphill grade & down & $0^\\circ$--$8^\\circ$ & ${g['n']}$ & ${g['violations']}$ & {step(g['min_step'])}\\\\")
    srows = []
    for r in d['signs']:
        lam = 'n/a' if r['reg'] != 'traction' else f"${(abs(r['Lam']) if abs(r['Lam']) < 0.5 else r['Lam']):.0f}$"
        pm = '$+$' if r['pred_m'] > 0 else '$-$'
        pr = '$+$' if r['pred_r'] > 0 else '$-$'
        srows.append(f"{SHORT[r['veh']]} & {LAW[r['law']]} & ${r['vm']}$ & {r['reg']} & ${r['J_a']:.0f}$ & ${r['Faero']:.0f}$ & {lam} & {pm} & ${r['dm']:.3f}$ & {pr} & ${r['dr']:.4f}$\\\\")
    body = f"\\begin{{table}}[htbp]\n\\centering\\small\n\\setlength{{\\tabcolsep}}{{5pt}}\n\\caption{{Check of Theorem~\\ref{{thm:signtable}}\\ref{{sg:mono}} and~\\ref{{sg:grade}}. Each parameter was swept over the stated range of multipliers in $11$ steps, and grade in $9$ steps, for the three vehicles, both tyre laws, and speeds of $10$, $40$, and $80$\\,mph. The second column gives the predicted direction of $\\amax$ as the parameter increases, and a pair is two consecutive points of a sweep. The last column is the smallest change of $\\amax$ (m/s$^2$) in the predicted direction over all pairs, which is non-negative when no pair violates the prediction. It is zero, to rounding error below $10^{{-12}}$, for friction and for power, because the envelope does not respond to friction where power binds or to power where traction binds (Proposition~\\ref{{prop:regimes}}). The largest value of $\\mu_r\\varphi_r'h/L$ encountered in the grade sweeps was ${g['worst_gain']:.3f}$.}}\\label{{tab:app-mono}}\n\\begin{{tabular}}{{llcrrr}}\n\\toprule\nParameter & Predicted & Range & Pairs & Violations & Smallest step\\\\\n\\midrule\n" + '\n'.join(mono) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\n\\begin{table}[htbp]\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{3.5pt}\n\\caption{Check of Theorem~\\ref{thm:signtable}\\ref{sg:mass} and~\\ref{sg:rho}. For each case the table gives the binding constraint, the terms $J\\amax$, $F_{\\mathrm{aero}}$, and $\\Lambda$ (N) of the mass formula, the predicted sign and the finite-difference value of $\\partial\\amax/\\partial\\ln m$, and the predicted sign and finite-difference value of $\\partial\\amax/\\partial\\ln\\rho$, both in m/s$^2$ from relative changes of $10^{-4}$. Where power binds, the mass prediction is negative by part~\\ref{sg:mass}.}\\label{tab:app-signs}\n\\begin{tabular}{lllllrrrcrcr}\n\\toprule\nVehicle & Law & mph & Binding & $J\\amax$ & $F_{\\mathrm{aero}}$ & $\\Lambda$ & & $\\partial_{\\ln m}\\amax$ & & $\\partial_{\\ln\\rho}\\amax$\\\\\n\\midrule\n' + '\n'.join(srows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    write('signs', body)


def propagation() -> None:
    d = load('t4_uncertainty')
    r = np.array(d['ratios'])
    qrows = []
    for q in d['quantiles']:
        ex = ', '.join((f'{x:.4f}' for x in q['exact']))
        qrows.append(f"{SHORT[q['vehicle']]} & {LAW[q['law']]} & {ex} & {small(q['maxdiff'])}\\\\")
    brows = []
    for b in d['box']:
        brows.append(f"{SHORT[b['vehicle']]} & {LAW[b['law']]} & ${b['mph']}$ & ${b['n']}$ & ${b['vertex_lo']:.4f}$ & ${b['sweep_lo']:.4f}$ & ${b['sweep_hi']:.4f}$ & ${b['vertex_hi']:.4f}$\\\\")
    frows = []
    for f in d['first_order']:
        amp = 'n/a' if np.isnan(f['amplification']) else f"${f['amplification']:.3f}$"
        frows.append(f"{SHORT[f['vehicle']]} & {LAW[f['law']]} & ${f['sd_first']:.4f}$ & ${f['sd_mc']:.4f}$ & {amp}\\\\")
    body = f'\\begin{{table}}[htbp]\n\\centering\\small\n\\caption{{Check of Theorem~\\ref{{thm:propagation}}\\ref{{pr:quant}} at $10$\\,mph. The friction distribution is the $@NFLEET@$ values of Table~\\ref{{tab:fmvss}} divided by their median, with relative standard deviation ${r.std():.3f}$ and $5$th and $95$th percentiles ${np.percentile(r, 5):.3f}$ and ${np.percentile(r, 95):.3f}$. The table gives the exact quantiles $\\amax(q_\\mu(p))$ (m/s$^2$) at $p=0.05$, $0.25$, $0.5$, $0.75$, and $0.95$, and the largest difference from the Monte Carlo quantiles of the $@NFLEET@$ envelope values.}}\\label{{tab:app-quant}}\n\\begin{{tabular}}{{llp{{6.2cm}}c}}\n\\toprule\nVehicle & Law & Exact quantiles & Largest difference\\\\\n\\midrule\n' + '\n'.join(qrows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\n\\begin{table}[htbp]\n\\centering\\small\n\\caption{Check of Theorem~\\ref{thm:propagation}\\ref{pr:box}. The box multiplies friction by $[0.85,1.15]$, $C_{rr}$ by $[0.8,1.3]$, $C_DA$ by $[0.9,1.1]$, and $P$ by $[0.9,1.05]$. The columns give the envelope (m/s$^2$) at the worst vertex, the smallest and largest envelope over $n$ random points of the box, and the envelope at the best vertex.}\\label{tab:app-box}\n\\begin{tabular}{llrrcccc}\n\\toprule\nVehicle & Law & mph & $n$ & Worst & Min & Max & Best\\\\\n\\midrule\n' + '\n'.join(brows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\n\\begin{table}[htbp]\n\\centering\\small\n\\caption{Check of Theorem~\\ref{thm:propagation}\\ref{pr:first} at $10$\\,mph: the first-order standard deviation of $\\amax$, the Monte Carlo standard deviation over the $@NFLEET@$ friction values (both m/s$^2$), and the local amplification $\\meff/(\\meff-T')$ where traction binds.}\\label{tab:app-first}\n\\begin{tabular}{llccc}\n\\toprule\nVehicle & Law & First order & Monte Carlo & Amplification\\\\\n\\midrule\n" + '\n'.join(frows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    write('propagation', body)


def transit() -> None:
    d = load('t5_mintime')
    grid = d['grid']
    mrows = []
    for m in d['monotone']:
        Ts = m['T']
        sel = [0, 2, 5, 8, 10]
        mrows.append(f"{SHORT[m['vehicle']]} & {LAW[m['law']]} & " + ' & '.join((f'${Ts[i]:.2f}$' for i in sel)) + '\\\\')
    qrows = []
    for q in d['quantiles']:
        qrows.append(f"{SHORT[q['vehicle']]} & " + ' & '.join((f'${x:.3f}$' for x in q['exact'])) + f" & {small(q['maxdiff'])}\\\\")
    lim = ', '.join((f"{SHORT[a['vehicle']]} ${100 * a['share']:.1f}\\%$" for a in d['at_limit']))
    hdr = ' & '.join((f'$\\times{grid[i]:.1f}$' for i in [0, 2, 5, 8, 10]))
    body = f'\\begin{{table}}[htbp]\n\\centering\\small\n\\caption{{Check of Theorem~\\ref{{thm:transit}} on the $6$\\,km rural route: the minimum trip time $T^\\star$ (s) at selected multipliers on the nominal friction, from a sweep of $11$ multipliers between $0.5$ and $1.5$ that is non-increasing in every case. The share of the route driven at the speed limit, at nominal friction with the ellipse law, is {lim}.}}\\label{{tab:app-transit}}\n\\begin{{tabular}}{{llccccc}}\n\\toprule\nVehicle & Law & {hdr}\\\\\n\\midrule\n' + '\n'.join(mrows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\n\\begin{table}[htbp]\n\\centering\\small\n\\caption{Trip-time quantiles (s) on the rural route with the ellipse law, computed as $T^\\star(q^{+}_\\mu(1-p))$ from three solves at $p=0.05$, $0.5$, and $0.95$, and the largest difference from the Monte Carlo quantiles of the $@NFLEET@$ solves at the measured friction values.}\\label{tab:app-transitq}\n\\begin{tabular}{lcccc}\n\\toprule\nVehicle & $p=0.05$ & $p=0.5$ & $p=0.95$ & Largest difference\\\\\n\\midrule\n' + '\n'.join(qrows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    write('transit', body)


def fourwheel() -> None:
    d = load('u3_four_wheel')
    nrows = '\n'.join((f"${n['fz_rel']:.1f}$ & ${100 * max(n['worst_violation'], 0):.2f}\\%$ & ${n['growth_min']:.2f}$--${n['growth_max']:.2f}$ & ${100 * n['below']:+.2f}\\%$\\\\" for n in d['nest']))
    erows = []
    for e in d['envelope']:
        lab = 'straight' if e['R'] > 100000000.0 else f"${e['R']:.0f}$\\,m"
        erows.append(lab + ' & ' + ' & '.join(('n/a' if x is None else f'${x:.3f}$' for x in e['a'])) + '\\\\')
    hdr = ' & '.join((f'$\\times{s:.1f}$' for s in d['scales']))
    body = '\\begin{table}[htbp]\n\\centering\\small\n\\caption{Nestedness of the PAC2002 combined-slip force set under the friction scale $\\lambda_\\mu\\in\\{0.3,0.5,0.7,0.9,1.0,1.2,1.5,1.8\\}$, for three normal loads. This range contains every scale, from about $0.36$ to $1.73$, at which our PAC2002 computations rely on monotonicity in friction. The largest outward violation is the largest relative amount by which the boundary at a smaller scale lies outside the boundary at the next larger scale, over $72$ directions. The radius ratio is the range over directions of the ratio of the boundary radius at scale $1.8$ to that at scale $0.3$. The last column gives the same violation for the scale $0.25$ against $0.3$, outside the range that we use, where a positive value means that nestedness fails.}\\label{tab:app-nest}\n\\begin{tabular}{cccc}\n\\toprule\nLoad ($F_{z0}$) & Largest outward violation & Radius ratio & Scale $0.25$\\\\\n\\midrule\n' + nrows + f"\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n\n\\begin{{table}}[htbp]\n\\centering\\small\n\\caption{{Four-wheel equilibrium envelope $\\amax$ (m/s$^2$) of the Corvette ZR1 at $20$\\,m/s with PAC2002 combined slip, against a multiplier on friction.}}\\label{{tab:app-fourwheel}}\n\\begin{{tabular}}{{l{'c' * len(d['scales'])}}}\n\\toprule\nRadius & {hdr}\\\\\n\\midrule\n" + '\n'.join(erows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    write('fourwheel', body)


def safe() -> None:
    d = load('u1_value')
    mu_road = d['mu_road']
    order = ['2010 Toyota Prius', '2012 Ford Focus', '2012 Nissan Leaf', '2013 Ford Focus Electric']
    by = {}
    for c in d['cycles']:
        by.setdefault(c['vehicle'], {})[c['cycle']] = max(c['traction'], c['braking'])
    crow = []
    for v in order:
        cells = []
        for cyc in ('udds', 'highway', 'us06'):
            x = by.get(v, {}).get(cyc)
            cells.append('n/a' if x is None else f'${x:.2f}$ (${x / mu_road:.2f}$)')
        crow.append(f'{v} & ' + ' & '.join(cells) + '\\\\')
    med = []
    for cyc in ('udds', 'highway', 'us06'):
        vals = [by[v][cyc] for v in order if cyc in by.get(v, {})]
        med.append(f'${np.median(vals):.2f}$ (${np.median(vals) / mu_road:.2f}$)')
    crow.append('\\midrule\nMedian & ' + ' & '.join(med) + '\\\\')
    chrows = []
    for c in d['chance']:
        chrows.append(f"{SHORT[c['vehicle']]} & ${c['delta']:.2f}$ & ${c['mu_plan']:.3f}$ & ${c['T']:.2f}$ & ${c['n_below']}$ & ${c['n_infeasible']}$ & ${c['n_infeasible'] / c['n']:.3f}$\\\\")
    lrows = []
    for r in d['law']:
        diff = 100 * (r['law'] - r['solver']) / r['solver']
        lrows.append(f"{SHORT[r['vehicle']]} & ${r['r']:.2f}$ & ${r['cycles']}$ & ${r['law']:.2f}$ & ${r['solver']:.2f}$ & ${diff:+.1f}\\%$ & ${100 * r['braking'] / r['law']:.0f}\\%$\\\\")
    body = '\\begin{table}[htbp]\n\\centering\\small\n\\caption{Lower bound on friction identified from each Argonne D3 drive cycle, taking the larger of the driven-axle traction utilisation and the all-wheel braking utilisation, and in brackets its fraction of the median friction $' + f'{mu_road:.2f}' + '$ of the FMVSS~135 stops. The Focus Electric has no highway run. In the second urban run of each of its sets the acceleration reaches about $3$\\,m/s$^2$, twice the $1.48$\\,m/s$^2$ maximum of the schedule, so those runs did not follow the urban trace. Without them its urban value is $0.44$, and the median is unchanged.}\\label{tab:d3cycles}\n\\begin{tabular}{lccc}\n\\toprule\nVehicle & UDDS (urban) & Highway & US06 (aggressive)\\\\\n\\midrule\n' + '\n'.join(crow) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\n\\begin{table}[htbp]\n\\centering\\small\n\\caption{Check of Theorem~\\ref{thm:chance} on the rural route with the friction ellipse. For each vehicle and safety level $\\delta$, the plan is the minimum-time profile at the friction multiplier $q(\\delta)$ of the $@NFLEET@$ measured values, with trip time $T^\\star$ (s). The plan was executed against all $@NFLEET@$ values. The table gives the number of values below $q(\\delta)$, the number at which the plan was not executable, and the infeasible share, which is at most $\\delta$ in every case.}\\label{tab:app-chance}\n\\begin{tabular}{lcccccc}\n\\toprule\nVehicle & $\\delta$ & $q(\\delta)$ & $T^\\star$ & Below & Infeasible & Share\\\\\n\\midrule\n' + '\n'.join(chrows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\n\\begin{table}[htbp]\n\\centering\\small\n\\caption{Check of Proposition~\\ref{prop:excess} on the urban route. For each vehicle and planning fraction $r$, the price (s) predicted by the transition-excess law, summed over the stop--go cycles, against the price computed by the three-pass solver, with their relative difference and the share of the predicted price due to braking.}\\label{tab:app-excess}\n\\begin{tabular}{lcccccc}\n\\toprule\nVehicle & $r$ & Cycles & Law & Solver & Difference & Braking\\\\\n\\midrule\n' + '\n'.join(lrows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    write('safe', body)


def probes() -> None:
    d = load('u1_value')
    rrows = [f"${r['k']}$ & ${r['u']:.3f}$ & ${r['sd_before']:.4f}$ & ${r['sd_after']:.4f}$\\\\" for r in d['ratchet']]
    krows = []
    for k in d['kstar']:
        nets = ' & '.join((f'${x:.2f}$' for x in k['nets'][1:]))
        krows.append(f"{SHORT[k['vehicle']]} & {k['route']} & ${k['prior_price']:.2f}$ & {nets} & ${k['k']}$\\\\")
    body = f"\\begin{{table}}[htbp]\n\\centering\\small\n\\caption{{The sequence of slip-risk-limited probes of Proposition~\\ref{{prop:ratchet}}, with prior standard deviation $s_0={d['s0']:.3f}$ of $\\ln\\mu$. For each probe the table gives its utilisation as a fraction of the friction and the standard deviation of $\\ln\\mu$ before and after it.}}\\label{{tab:app-ratchet}}\n\\begin{{tabular}}{{cccc}}\n\\toprule\nProbe & Utilisation & Before & After\\\\\n\\midrule\n" + '\n'.join(rrows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\n\\begin{table}[htbp]\n\\centering\\small\n\\setlength{\\tabcolsep}{4.5pt}\n\\caption{Net trip-time gain (s) of $k$ probes relative to planning at the $5\\%$ quantile of the fleet prior, for $k=1,\\dots,5$, together with the price of planning at that prior, which is the largest gain possible. The optimal number $k^\\star$ maximises the net gain, with $k^\\star=0$ when every net gain is negative.}\\label{tab:app-kstar}\n\\begin{tabular}{llccccccc}\n\\toprule\nVehicle & Route & Prior price & $k=1$ & $k=2$ & $k=3$ & $k=4$ & $k=5$ & $k^\\star$\\\\\n\\midrule\n' + '\n'.join(krows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    write('probes', body)


def spacing() -> None:
    d = load('u1d_spacing')
    curves = {(c['vehicle'], c['route']): c for c in d['curves']}
    lmin, llin = ({}, {})
    for t in d['threshold']:
        lmin[t['vehicle'], t['route']] = t['l_min']
        llin[t['vehicle'], t['route']] = t['l_lin']
    opt = {}
    for o in d['optimum']:
        opt.setdefault((o['vehicle'], o['route']), []).append(o)
    rows = []
    for key, c in curves.items():
        cells = []
        for o in opt[key]:
            num = f"${o['D_num'] / 1000:.1f}$" if o['worth'] else 'none'
            cells.append(num + f" & ${o['D_law'] / 1000:.1f}$")
        rows.append(f"{SHORT[key[0]]} & {key[1]} & ${opt[key][0]['c_p']:.2f}$ & ${c['kappa']:.2f}$ & ${c['h0']:.3f}$ & ${lmin[key] / 1000:.1f}$ & ${llin[key] / 1000:.1f}$ & " + ' & '.join(cells) + '\\\\')
    srows = [f"${s['ell0'] / 1000:.0f}$ & ${s['ell'] / 1000:.0f}$ & ${s['D0'] / 1000:.2f}$ & ${s['D'] / 1000:.2f}$ & ${s['slope']:.3f}$\\\\" for s in d['scaling']]
    body = f"\\begin{{table}}[htbp]\n\\centering\\small\n\\setlength{{\\tabcolsep}}{{3pt}}\n\\caption{{Inputs and results of Theorem~\\ref{{thm:spacing}}, with field standard deviation $\\sigma={d['sigma']:.3f}$, $z={d['z']:.3f}$, and probe residual standard deviation $s_p={d['s_p']:.4f}$. The columns give the probe cost $c_p$ (s), the slope $\\eta$ and the no-probe rate $\\psi_0$ of the price curve (s/km), the exact threshold $\\ell_{{\\min}}=c_p/\\Gamma$ of part~\\ref{{sp:exact}} and its closed form for a linear price curve (km), and for each correlation length $\\ell$ the numerical minimiser of $\\mathcal J(D)$ and the asymptotic $D^\\star$ of part~\\ref{{sp:law}} (km). The entry none means that no spacing beats not probing. The exact threshold decides this correctly in all ${d['total']}$ cases, and in all ${d['sharp']}$ further cases at $0.8\\,\\ell_{{\\min}}$ and $1.25\\,\\ell_{{\\min}}$.}}\\label{{tab:app-spacing}}\n\\begin{{tabular}}{{llccccccccccc}}\n\\toprule\n& & & & & \\multicolumn{{2}}{{c}}{{$\\ell_{{\\min}}$}} & \\multicolumn{{2}}{{c}}{{$\\ell=1$\\,km}} & \\multicolumn{{2}}{{c}}{{$\\ell=5$\\,km}} & \\multicolumn{{2}}{{c}}{{$\\ell=25$\\,km}}\\\\\n\\cmidrule(lr){{6-7}}\\cmidrule(lr){{8-9}}\\cmidrule(lr){{10-11}}\\cmidrule(lr){{12-13}}\nVehicle & Route & $c_p$ & $\\eta$ & $\\psi_0$ & Exact & Linear & Num. & Law & Num. & Law & Num. & Law\\\\\n\\midrule\n" + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\n\\begin{table}[htbp]\n\\centering\\small\n\\caption{Scaling of the numerical optimum with the correlation length for the Civic on the urban route, with a probe cost of $0.1$\\,s and a residual standard deviation of $10^{-4}$, close to the conditions of Theorem~\\ref{thm:spacing}\\ref{sp:law}. The last column is the slope of $\\ln D^\\star$ against $\\ln\\ell$, which the law predicts to be $1/3$.}\\label{tab:app-scaling}\n\\begin{tabular}{ccccc}\n\\toprule\n$\\ell_0$ (km) & $\\ell$ (km) & $D^\\star(\\ell_0)$ (km) & $D^\\star(\\ell)$ (km) & Slope\\\\\n\\midrule\n' + '\n'.join(srows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    write('spacing', body)


ROUTE_NAME = {'argonne_naperville': 'Suburban', 'boulder_canyon': 'Canyon'}


def routes() -> None:
    d = load('u2_routes')
    pct = lambda x, t: f'${x:.1f}$ (${100 * x / t:.1f}\\%$)'
    body_rows, probe_rows = ([], [])
    for c in d['cases']:
        sig = 'red' if c['signals_stop'] else 'green'
        label = f"{ROUTE_NAME[c['route']]} route, signals {sig}"
        body_rows.append('\\midrule\n\\multicolumn{6}{l}{' + label + '}\\\\')
        for r in c['rows']:
            prior = r['Tq'] - r['T1']
            body_rows.append(f"{SHORT[r['vehicle']]} & ${r['T1']:.0f}$ & {pct(r['prices'][0], r['T1'])} & {pct(r['prices'][1], r['T1'])} & {pct(prior, r['T1'])} & ${r['net']:.1f}$\\\\")
            probe_rows.append(f"{ROUTE_NAME[c['route']]}, {sig} & {SHORT[r['vehicle']]} & ${prior:.2f}$ & ${r['Tq'] - r['T1p']:.2f}$ & ${r['cost']:.2f}$ & ${r['net']:.2f}$ & ${100 * r['net'] / prior:.0f}\\%$\\\\")
    meta_rows, mono_rows = ([], [])
    for c in d['cases']:
        if not c['signals_stop']:
            meta_rows.append(f"{ROUTE_NAME[c['route']]} & ${c['L'] / 1000:.1f}$ & ${c['n_sig']}$ & ${c['n_stop']}$ & ${c['climb']:.0f}$ & ${100 * c['tagged']:.0f}\\%$ & ${1 / c['max_kappa']:.0f}$\\\\")
        sig = 'red' if c['signals_stop'] else 'green'
        mono_rows.append(f"{ROUTE_NAME[c['route']]}, {sig} & " + ' & '.join((f'${t:.1f}$' for t in c['mono_T'][::2])) + '\\\\')
    mus = ' & '.join((f'$\\times{m:.1f}$' for m in d['cases'][0]['mono_mu'][::2]))
    body = f"\\begin{{table}}[htbp]\n\\centering\\small\n\\caption{{Price of unidentifiability on the two real routes, in seconds and as a percentage of $T^\\star(\\mu)$, when planning at the absolute utilisation ${d['u_ord']:.2f}$ identified by ordinary driving, at ${d['u_us06']:.2f}$ identified by the US06 cycle, and at the $5\\%$ quantile ${d['q_lognormal']:.3f}$ of the fleet prior of Section~\\ref{{sec:probe}}. The last column is the net gain (s) of one probe relative to planning at that prior.}}\\label{{tab:realroutes}}\n\\begin{{tabular}}{{lccccc}}\n\\toprule\nVehicle & $T^\\star(\\mu)$ (s) & Ordinary & US06 & Fleet prior & One probe\\\\\n" + '\n'.join(body_rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    write('routes', body)
    app = '\\begin{table}[htbp]\n\\centering\\small\n\\caption{The two real routes after conversion: length (km), traffic signals, stop signs, accumulated climb of the smoothed elevation profile (m), share of route vertices with a posted-limit tag, and the tightest radius of the smoothed path (m).}\\label{tab:app-routes}\n\\begin{tabular}{lcccccc}\n\\toprule\nRoute & Length & Signals & Stop signs & Climb & Tagged & Tightest radius\\\\\n\\midrule\n' + '\n'.join(meta_rows) + f"\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n\n\\begin{{table}}[htbp]\n\\centering\\small\n\\caption{{One probe on the real routes. The prior price is $T^\\star$ at the $5\\%$ quantile ${d['q_lognormal']:.3f}$ of the fleet prior less $T^\\star(\\mu)$. The gain is the reduction in $T^\\star$ when one probe at ${d['u_probe']:.3f}$ of the friction raises the planning fraction to ${d['q_one_probe']:.3f}$, and the cost is $c(u_p)$ at the median posted limit of the route. All times are in seconds. The last column is the share of the prior price that the probe recovers.}}\\label{{tab:app-routeprobe}}\n\\begin{{tabular}}{{llccccc}}\n\\toprule\nRoute & Vehicle & Prior price & Gain & Cost & Net & Recovered\\\\\n\\midrule\n" + '\n'.join(probe_rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\n\\begin{table}[htbp]\n\\centering\\small\n\\setlength{\\tabcolsep}{4pt}\n\\caption{Check of Theorem~\\ref{thm:transit} on the real routes: the minimum trip time $T^\\star$ (s) of the Civic at selected multipliers on the nominal friction, from a sweep of $11$ multipliers between $0.4$ and $1.4$ that is non-increasing in every case.}\\label{tab:app-routemono}\n\\begin{tabular}{lcccccc}\n\\toprule\nRoute & ' + mus + '\\\\\n\\midrule\n' + '\n'.join(mono_rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    write('routes_app', app)


SOBOL_INPUTS = ['Friction', 'Wetness', 'Wet-to-dry ratio', 'Temperature', 'Altitude', '$C_{rr}$', '$C_DA$', 'Headwind', 'Payload', 'Grade']


def _sobol_table(d, law, key, label, caption):
    cases = [c for c in d['cases'] if c['law'] == law]
    rows = []
    for i, nm in enumerate(SOBOL_INPUTS):
        cells = []
        for c in cases:
            for sp in ('10 mph', '60 mph', 'pooled'):
                cells.append(f'${max(c[sp][key][i], 0.0):.2f}$')
        rows.append(nm + ' & ' + ' & '.join(cells) + '\\\\')
    if key == 'ST':
        cells = []
        for c in cases:
            cells += ['', '', f"${c['pooled']['ST_speed']:.2f}$"]
        rows.append('\\midrule\nSpeed & ' + ' & '.join(cells) + '\\\\')
    return f'\\begin{{table}}[htbp]\n\\centering\\small\n\\setlength{{\\tabcolsep}}{{4.2pt}}\n\\caption{{{caption}}}\\label{{{label}}}\n\\begin{{tabular}}{{lccccccccc}}\n\\toprule\n& \\multicolumn{{3}}{{c}}{{Leaf}} & \\multicolumn{{3}}{{c}}{{Civic}} & \\multicolumn{{3}}{{c}}{{ZR1}}\\\\\n\\cmidrule(lr){{2-4}}\\cmidrule(lr){{5-7}}\\cmidrule(lr){{8-10}}\nInput & $10$ & $60$ & Pool & $10$ & $60$ & Pool & $10$ & $60$ & Pool\\\\\n\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'


def sobol() -> None:
    d = load('t6_sobol')
    body = _sobol_table(d, 'pac2002', 'ST', 'tab:sobol', 'Total Sobol indices $ST_i$ of $\\amax$ with the PAC2002 tyre law, at $10$\\,mph, at $60$\\,mph, and pooled over both speeds. In the pooled case speed is an eleventh input, and its total index is given in the last row.')
    write('sobol', body)
    se = max((max((max(c[sp]['ST_se']) for sp in ('10 mph', '60 mph', 'pooled'))) for c in d['cases']))
    se1 = max((max((max(c[sp]['S_se']) for sp in ('10 mph', '60 mph', 'pooled'))) for c in d['cases']))
    pac = {c['vehicle']: c for c in d['cases'] if c['law'] == 'pac2002'}
    diff = max((abs(round(max(c[sp]['ST'][i], 0.0), 2) - round(max(pac[c['vehicle']][sp]['ST'][i], 0.0), 2)) for c in d['cases'] if c['law'] == 'ellipse' for sp in ('10 mph', '60 mph', 'pooled') for i in range(len(SOBOL_INPUTS))))
    shares = [f"{SHORT[c['vehicle']]} & {LAW[c['law']]} & ${c['10 mph']['traction_share']:.2f}$ & ${c['60 mph']['traction_share']:.2f}$\\\\" for c in d['cases']]
    app = _sobol_table(d, 'pac2002', 'S', 'tab:app-sobol-first', f'First-order Sobol indices $S_i$ with the PAC2002 tyre law, in the layout of Table~\\ref{{tab:sobol}}. Negative estimates are shown as zero. The largest bootstrap standard error of any first-order index is ${se1:.3f}$.') + '\n' + _sobol_table(d, 'ellipse', 'ST', 'tab:app-sobol-ellipse', f'Total Sobol indices $ST_i$ with the friction ellipse, in the layout of Table~\\ref{{tab:sobol}}. No entry differs from the PAC2002 value by more than ${diff:.2f}$.') + f"\n\\begin{{table}}[htbp]\n\\centering\\small\n\\caption{{Design of the Sobol computation. The inputs are independent and uniform on the stated ranges. Each index uses $N={d['N']}$ rows per sample matrix, and the whole computation used ${thousands(d['evaluations'])}$ evaluations of the envelope. Standard errors are from $200$ bootstrap resamples of the rows, and the largest for any total index is ${se:.3f}$.}}\\label{{tab:app-sobol-design}}\n\\begin{{tabular}}{{lc}}\n\\toprule\nInput & Range\\\\\n\\midrule\nFriction multiplier & $[0.81,1.24]$\\\\\nWetness $w$, dry for $w<0.5$, film $(w-0.5)\\times6$\\,mm otherwise & $[0,1]$\\\\\nWet-to-dry peak friction ratio & $[0.71,0.87]$\\\\\nAmbient temperature ($^\\circ$C) & $[-10,40]$\\\\\nAltitude (m) & $[0,3000]$\\\\\n$C_{{rr}}$ multiplier & $[0.8,1.3]$\\\\\n$C_DA$ multiplier & $[0.9,1.1]$\\\\\nHeadwind (m/s) & $[-5,15]$\\\\\nPayload (kg) & $[0,300]$\\\\\nUphill grade (\\%) & $[0,6]$\\\\\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n\n\\begin{{table}}[htbp]\n\\centering\\small\n\\caption{{Share of the first $200$ samples of each Sobol design in which traction binds.}}\\label{{tab:app-sobol-regime}}\n\\begin{{tabular}}{{llcc}}\n\\toprule\nVehicle & Tyre law & $10$\\,mph & $60$\\,mph\\\\\n\\midrule\n" + '\n'.join(shares) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    write('sobol_app', app)


def transient() -> None:
    import statistics
    d = load('t7b_transient')
    rows = []
    for k in (1, 2, 3):
        rs = [m['ratio'] for m in d['monotone'] if m['k'] == k]
        rows.append(f'Non-increasing & ${k}$ & ${len(rs)}$ & $\\sum_j\\tau_j$ & ${statistics.median(rs):.3f}$ & ${max(rs):.3f}$\\\\')
    g = d['general']
    rows.append(f"General & $1$ & ${len(g)}$ & $\\tau(1+\\operatorname{{Var}}^+)$ & ${statistics.median((x['ratio'] for x in g)):.3f}$ & ${max((x['ratio'] for x in g)):.3f}$\\\\")
    rows.append(f"General & $1$ & ${len(g)}$ & $\\tau$ & ${statistics.median((x['ratio_plain'] for x in g)):.3f}$ & ${max((x['ratio_plain'] for x in g)):.3f}$\\\\")
    srows = [f"${s['levels']}$ & ${s['penalty'] / s['tau']:.2f}$ & ${s['prediction'] / s['tau']:.2f}$ & ${s['bound'] / s['tau']:.2f}$\\\\" for s in d['staircase']]
    body = '\\begin{table}[htbp]\n\\centering\\small\n\\caption{Checks of Theorems~\\ref{thm:transientbound} and~\\ref{thm:transientvar} on random piecewise-linear envelopes. The non-increasing envelopes have seven levels on random knots, lags drawn uniformly from $[0.02,0.4]$\\,s, and target speeds from $[5,40]$\\,m/s. The general envelopes have eight random levels and one lag drawn from $[0.05,0.4]$\\,s. The last two columns give the median and the largest ratio of the penalty to the stated bound. The last row compares the general envelopes with the bound $\\tau$ that holds for non-increasing envelopes.}\\label{tab:app-transient}\n\\begin{tabular}{lccccc}\n\\toprule\nEnvelope & Lags & Cases & Bound & Median ratio & Largest ratio\\\\\n\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\n\\begin{table}[htbp]\n\\centering\\small\n\\caption{Penalty of a single lag $\\tau=' + f"{d['staircase'][0]['tau']:.1f}" + '$\\,s on a doubling staircase envelope, in units of $\\tau$, with each level held for about $5\\tau$. The prediction is the limit of Proposition~\\ref{prop:transientfail}. The bound is the value $\\tau(1+\\operatorname{Var}^+)$ of Theorem~\\ref{thm:transientvar}, which is proved for Lipschitz envelopes and is shown here for comparison.}\\label{tab:app-staircase}\n\\begin{tabular}{cccc}\n\\toprule\nLevels & Penalty & Prediction & Bound\\\\\n\\midrule\n' + '\n'.join(srows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    v = load('t7_transient')
    vrows = []
    for r in v['lags']:
        lag = ', '.join((f'{t:.2f}' for t in r['taus']))
        vrows.append(f"{SHORT[r['vehicle']]} & ${lag}$ & ${r['T_phys']:.3f}$ & ${r['penalty']:.3f}$ & ${100 * r['ratio']:.1f}\\%$\\\\")
    rrows = [f"{SHORT[r['vehicle']]} & " + ' & '.join((f'${p:.3f}$' for p in r['penalty'])) + '\\\\' for r in v['relax']]
    vp = max((r['vplus'] for r in v['vplus']))
    body += f'\n\\begin{{table}}[htbp]\n\\centering\\small\n\\caption{{The $0$--$60$\\,mph penalty of the three vehicles under constant lags (s), with the time $T_{{\\mathrm{{phys}}}}$ along the envelope and the penalty as a share of $\\sum_j\\tau_j$. The positive variation of $\\ln\\amax$ from rest to $60$\\,mph, computed on $4001$ speeds, is ${vp:.0f}$ for every vehicle. These runs follow the envelope without the interruptions of gear shifts, which is why the time of the Civic is shorter than in Table~\\ref{{tab:app-060}}.}}\\label{{tab:app-transient-veh}}\n\\begin{{tabular}}{{lcccc}}\n\\toprule\nVehicle & Lags (s) & $T_{{\\mathrm{{phys}}}}$ (s) & Penalty (s) & Share\\\\\n\\midrule\n' + '\n'.join(vrows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\n\\begin{table}[htbp]\n\\centering\\small\n\\caption{The $0$--$60$\\,mph penalty (s) with tyre relaxation length $0.5$\\,m and a powertrain lag of $0.10$\\,s, against the floor speed $v_0$ of the relaxation model.}\\label{tab:app-relax}\n\\begin{tabular}{lcccc}\n\\toprule\nVehicle & $v_0=2$\\,m/s & $v_0=1$\\,m/s & $v_0=0.5$\\,m/s & $v_0=0.25$\\,m/s\\\\\n\\midrule\n' + '\n'.join(rrows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    write('transient', body)


def price() -> None:
    d = load('u1_value')
    route_name = {'highway': 'Highway', 'rural': 'Rural', 'mountain': 'Mountain', 'urban': 'Urban'}
    prior = {(k['vehicle'], k['route']): k['prior_price'] for k in d['kstar']}
    rows = []
    last = None
    for r in d['price']:
        T = r['T1']
        p = [r['price'][0], r['price'][1], prior[r['vehicle'], r['route']]]
        label = route_name[r['route']] if r['route'] != last else ''
        last = r['route']
        cells = ' & '.join((f'${x:.1f}$ (${100 * x / T:.1f}\\%$)' for x in p))
        rows.append((r['route'], SHORT[r['vehicle']], f"{label} & {SHORT[r['vehicle']]} & ${T:.1f}$ & {cells}\\\\"))
    order = ['highway', 'rural', 'mountain', 'urban']
    rows.sort(key=lambda x: (order.index(x[0]), ['Leaf', 'Civic', 'ZR1'].index(x[1])))
    out, prev = ([], None)
    for route, veh, line in rows:
        if route == prev:
            line = '& ' + line.split(' & ', 1)[1]
        prev = route
        out.append(line)
    body = f"\\begin{{table}}[htbp]\n\\centering\\small\n\\setlength{{\\tabcolsep}}{{4pt}}\n\\caption{{Price of unidentifiability $\\Pi$ in seconds, with its percentage of the full-information time $T^\\star(\\mu)$, for the friction ellipse. The first two price columns plan at the absolute utilisation $u_{{\\mathrm{{obs}}}}={d['u_ord']:.2f}$ identified by ordinary driving and $u_{{\\mathrm{{obs}}}}={d['u_us06']:.2f}$ identified by the US06 cycle (Corollary~\\ref{{cor:price}}). The last plans at the $5\\%$ quantile of the fleet prior of Section~\\ref{{sec:probe}}, as Theorem~\\ref{{thm:chance}} allows.}}\\label{{tab:price}}\n\\begin{{tabular}}{{llcccc}}\n\\toprule\nRoute & Vehicle & $T^\\star(\\mu)$ (s) & $\\Pi$, ordinary & $\\Pi$, US06 & Fleet prior\\\\\n\\midrule\n" + '\n'.join(out) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    write('price', body)


TABLES = {'wellposed': wellposed, 'fisher': fisher, 'epa': epa, 'signs': signs, 'propagation': propagation, 'transit': transit, 'fourwheel': fourwheel, 'safe': safe, 'probes': probes, 'spacing': spacing, 'routes': routes, 'sobol': sobol, 'transient': transient, 'price': price}

if __name__ == '__main__':
    for n in sys.argv[1:] or list(TABLES):
        TABLES[n]()
