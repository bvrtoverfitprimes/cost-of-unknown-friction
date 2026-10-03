from __future__ import annotations
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
DATA = Path(__file__).resolve().parent.parent / 'data' / 'fmvss135'
PDF_DIR = DATA / 'pdf'
NUM = '-?\\d+(?:\\.\\d+)?'
CLEAN_TESTS = ('Cold Effectiveness', 'High Speed Effectiveness')

@dataclass
class Stop:
    report: str
    year: int
    make: str
    model: str
    vehicle_type: str
    test: str
    load: str
    speed_kmh: float
    distance_m: float
    requirement_m: float | None
    pedal_force_N: float | None
    pfc_skidpad: float | None
    pfc_track: float | None
    pfc_tire: str | None
    pfc_spread: float | None
    mass_llvw_kg: float | None
    mass_gvwr_kg: float | None

def _pfc(text: str):
    tire_m = re.search('ASTM\\s+E1337\\s*w/\\s*(\\w+)\\s*tire', text, re.I)
    tire = tire_m.group(1) if tire_m else None
    m = re.search('Average PFC during the test period was\\s*(' + NUM + ')\\s*\\(Skid Pad\\)\\s*(?:and)?\\s*(' + NUM + ')?\\s*\\(Test Track\\)', text, re.I | re.S)
    if m:
        a = float(m.group(1))
        b = float(m.group(2)) if m.group(2) else None
        return (a, b, tire, 0.0)
    vals = [float(x) for x in re.findall('(?:ROAD\\s*\\n?\\s*PFC\\s*:|PFC\\s+of)\\s*(' + NUM + ')', text, re.I)]
    vals = [v for v in vals if 0.3 <= v <= 1.5]
    if not vals:
        return (None, None, tire, None)
    vals.sort()
    med = vals[len(vals) // 2]
    return (med, None, tire, max(vals) - min(vals))

def _weights(reader) -> tuple[float | None, float | None]:
    llvw = gvwr = None
    for p in reader.pages[8:15]:
        try:
            quick = p.extract_text() or ''
            if 'TOTAL' not in quick:
                continue
            t = p.extract_text(extraction_mode='layout')
        except Exception:
            continue
        if llvw is None:
            m = re.search('TOTAL\\s+LLVW\\s*-?\\s*\\|?\\s*kg\\s*(' + NUM + ')', t, re.I)
            if m:
                llvw = float(m.group(1))
        if gvwr is None:
            m = re.search('TOTAL\\s+GVW\\s*R\\s*-?\\s*\\|?\\s*kg\\s*(' + NUM + ')', t, re.I)
            if m:
                gvwr = float(m.group(1))
        if llvw and gvwr:
            break
    if llvw and (not 500 <= llvw <= 5000):
        llvw = None
    if gvwr and (not 500 <= gvwr <= 6000):
        gvwr = None
    if llvw and gvwr and (gvwr <= llvw):
        llvw = gvwr = None
    return (llvw, gvwr)

def parse(pdf: Path, meta: dict) -> list[Stop]:
    import pypdf
    try:
        r = pypdf.PdfReader(str(pdf))
        text = '\n'.join((p.extract_text() or '' for p in r.pages[:14]))
    except Exception:
        return []
    skid, track, tire, spread = _pfc(text)
    m_llvw, m_gvwr = _weights(r)
    out: list[Stop] = []
    for name in CLEAN_TESTS:
        for m in re.finditer(re.escape(name) + '\\s*(?:\\(\\d+\\))?\\s*(GVWR|LLVW)([^\\n]{0,150})', text):
            load, rest = (m.group(1), m.group(2))
            nums = [float(x) for x in re.findall(NUM, rest)]
            if len(nums) < 5:
                continue
            speed, dist = (nums[0], nums[-1])
            req = nums[3] if len(nums) > 3 else None
            pedal = nums[-2] if len(nums) > 5 else None
            if not (60 <= speed <= 200 and 15 <= dist <= 250):
                continue
            out.append(Stop(report=pdf.stem, year=meta.get('vehicle_model_year', 0), make=meta.get('make', ''), model=meta.get('model', ''), vehicle_type=meta.get('vehicle_type', ''), test=name, load=load, speed_kmh=speed, distance_m=dist, requirement_m=req, pedal_force_N=pedal, pfc_skidpad=skid, pfc_track=track, pfc_tire=tire, pfc_spread=spread, mass_llvw_kg=m_llvw, mass_gvwr_kg=m_gvwr))
    best: dict[tuple, Stop] = {}
    for s in out:
        k = (s.test, s.load, round(s.speed_kmh, 1))
        if k not in best or s.distance_m < best[k].distance_m:
            best[k] = s
    return list(best.values())

def parse_all() -> list[Stop]:
    idx = {Path(r['testReportUrl']).stem: r for r in json.loads((DATA / 'index.json').read_text(encoding='utf-8')) if r.get('testReportUrl')}
    rows: list[Stop] = []
    for pdf in sorted(PDF_DIR.glob('*.pdf')):
        rows.extend(parse(pdf, idx.get(pdf.stem, {})))
    return rows

def main() -> int:
    rows = parse_all()
    print(f'parsed {len(rows)} stops from {len({r.report for r in rows})} reports')
    have_pfc = [r for r in rows if r.pfc_skidpad]
    print(f'  with measured PFC: {len(have_pfc)}')
    out = DATA / 'stops.csv'
    if rows:
        import csv
        with out.open('w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(asdict(rows[0])))
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))
        print(f'  wrote {out}')
    for r in rows[:12]:
        print(f'   {r.year} {r.make[:12]:12s} {r.model[:14]:14s} {r.test[:20]:20s} {r.load} {r.speed_kmh:6.1f} km/h  {r.distance_m:6.2f} m  PFC={r.pfc_skidpad}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
