from __future__ import annotations
import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'd3'
MPH = 0.44704
LBF = 4.4482216152605
LB = 0.45359237
CANON = {'timestamp': 't', 'time': 't', 'dynospeed': 'v_mph', 'dynotractiveeffort': 'F_N', 'celltemp': 'T_cell_C', 'testcelltemp': 'T_cell_C', 'cellrh': 'RH_pct', 'testcellrh': 'RH_pct', 'phase#': 'phase', 'enginespeed': 'rpm', 'batterysoc': 'soc', 'hvbatterysoc': 'soc', 'acceleratorpedalposition': 'pedal_pct', 'fuelflowfrombenchmodal': 'fuel_ccps', 'enginecoolanttemp': 'T_cool_C', 'engineoiltemp': 'T_oil_C'}

def _canon(header: str) -> str:
    h = header.strip()
    h = re.sub('\\[.*?\\]', '', h)
    h = h.replace('w/500A max clamp', '')
    h = h.replace('w/200A max clamp', '200')
    key = re.sub('[\\s_]+', '', h).lower()
    if key in CANON:
        return CANON[key]
    m = re.match('^(hv)?battery(voltage|current)(200)?$', key)
    if m:
        return ('V_batt' if m.group(2) == 'voltage' else 'I_batt') + ('_200' if m.group(3) else '')
    return key

@dataclass
class Run:
    test_id: str
    vehicle: str
    sig: dict[str, np.ndarray]
    cycle: str = '?'
    T_cell_F: float = float('nan')

    @property
    def t(self) -> np.ndarray:
        return self.sig['t']

    @property
    def v(self) -> np.ndarray:
        return self.sig['v_mph'] * MPH

    @property
    def F(self) -> np.ndarray:
        return self.sig['F_N']

    @property
    def dt(self) -> float:
        d = np.diff(self.t)
        return float(np.median(d))

    def has(self, *keys: str) -> bool:
        return all((k in self.sig for k in keys))

    def accel(self, smooth_s: float=0.5) -> np.ndarray:
        v, dt = (self.v, self.dt)
        half = max(1, int(round(smooth_s / dt / 2)))
        n = len(v)
        a = np.zeros(n)
        offs = np.arange(-half, half + 1) * dt
        denom = float((offs ** 2).sum())
        for k, o in zip(range(-half, half + 1), offs):
            a += o * np.roll(v, -k)
        a /= denom
        a[:half] = a[half]
        a[-half:] = a[-half - 1]
        return a

@dataclass
class VehicleSpec:
    name: str
    test_weight_lb: float
    A_lbf: float
    B_lbf_mph: float
    C_lbf_mph2: float
    architecture: str = '?'
    source: str = 'D3 master summary sheet'

    @property
    def m_test(self) -> float:
        return self.test_weight_lb * LB

    def road_load(self, v: np.ndarray) -> np.ndarray:
        v_mph = v / MPH
        return (self.A_lbf + self.B_lbf_mph * v_mph + self.C_lbf_mph2 * v_mph ** 2) * LBF

def load_zip(zip_path: Path, vehicle: str) -> list[Run]:
    runs = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.lower().endswith('.txt'):
                continue
            if 'charging' in name.lower():
                continue
            raw = z.read(name).decode('utf-8', errors='replace')
            lines = [ln for ln in raw.split('\n') if ln.strip()]
            if len(lines) < 50:
                continue
            hdr = [_canon(h) for h in lines[0].split('\t')]
            try:
                arr = np.array([[float(x) for x in ln.split('\t')] for ln in lines[1:]], dtype=float)
            except ValueError:
                continue
            if arr.shape[1] != len(hdr):
                continue
            sig = {h: arr[:, i] for i, h in enumerate(hdr)}
            if 't' not in sig or 'v_mph' not in sig or 'F_N' not in sig:
                continue
            tid = re.search('(\\d{6,})', Path(name).name)
            runs.append(Run(tid.group(1) if tid else Path(name).stem, vehicle, sig))
    return sorted(runs, key=lambda r: r.test_id)

def annotate(runs: list[Run], summary_pdf: Path | None) -> list[Run]:
    if summary_pdf is None or not summary_pdf.exists():
        return runs
    try:
        import pypdf
    except ImportError:
        return runs
    text = '\n'.join((p.extract_text() for p in pypdf.PdfReader(str(summary_pdf)).pages))
    for r in runs:
        m = re.search(f'{r.test_id}\\s+(.+?)\\s+\\d{{2}}/\\d{{2}}/\\d{{2}}(?:,\\s*[\\d:]+\\s*[AP]M\\s*)?\\s*(-?\\d+)', text)
        if m:
            cyc = re.sub('\\s*(CS|HS|HSt|CSt)\\s*$', '', m.group(1)).strip()
            r.cycle = re.sub('\\s+', ' ', cyc)
            r.T_cell_F = float(m.group(2))
    return runs

@dataclass
class BalanceResult:
    run: Run
    rmse_N: float
    rmse_frac: float
    m_eff_fitted: float
    m_test: float
    mass_err_pct: float
    r2: float
    n: int
    notes: list[str] = field(default_factory=list)

def force_balance(run: Run, spec: VehicleSpec, smooth_s: float=0.5) -> BalanceResult:
    a = run.accel(smooth_s)
    v = run.v
    F_meas = run.F
    F_rl = spec.road_load(v)
    mask = v > 1.0
    a, v, F_meas, F_rl = (a[mask], v[mask], F_meas[mask], F_rl[mask])
    F_pred = spec.m_test * a + F_rl
    resid = F_meas - F_pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    scale = float(np.sqrt(np.mean(F_meas ** 2)))
    y = F_meas - F_rl
    m_fit = float(np.dot(a, y) / np.dot(a, a))
    ss_res = float(np.sum((F_meas - F_pred) ** 2))
    ss_tot = float(np.sum((F_meas - np.mean(F_meas)) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return BalanceResult(run=run, rmse_N=rmse, rmse_frac=rmse / scale, m_eff_fitted=m_fit, m_test=spec.m_test, mass_err_pct=100.0 * (m_fit - spec.m_test) / spec.m_test, r2=r2, n=int(mask.sum()))
