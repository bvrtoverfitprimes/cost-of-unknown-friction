from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
API = 'https://ovsc.api.nhtsa.dot.gov/vehicle?fmvss_code=135'
DATA = Path(__file__).resolve().parent.parent / 'data' / 'fmvss135'
PDF_DIR = DATA / 'pdf'
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
HEADERS = ['-H', 'Accept: application/json, text/plain, */*', '-H', 'Accept-Language: en-US,en;q=0.9', '-H', 'Origin: https://www.nhtsa.gov', '-H', 'Referer: https://www.nhtsa.gov/']

def fetch_index(force: bool=False) -> list[dict]:
    DATA.mkdir(parents=True, exist_ok=True)
    idx = DATA / 'index.json'
    if force or not idx.exists():
        subprocess.run(['curl', '-sS', '-L', '--compressed', '-A', UA, *HEADERS, '-o', str(idx), API], check=True, timeout=180)
    return json.loads(idx.read_text(encoding='utf-8'))

def download(rec: dict) -> Path | None:
    url = rec.get('testReportUrl') or ''
    if not url.lower().endswith('.pdf'):
        return None
    dest = PDF_DIR / Path(url).name
    if dest.exists() and dest.stat().st_size > 20000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(['curl', '-sS', '-L', '--compressed', '-A', UA, '-H', 'Referer: https://www.nhtsa.gov/', '-o', str(dest), '-w', '%{http_code}', url], capture_output=True, timeout=600)
    code = r.stdout.decode(errors='replace').strip()
    if code != '200' or not dest.exists() or dest.stat().st_size < 20000:
        dest.unlink(missing_ok=True)
        return None
    return dest

def main() -> int:
    recs = fetch_index()
    print(f'FMVSS 135 vehicle reports in index: {len(recs)}')
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(recs)
    got = fail = 0
    total = 0
    for i, rec in enumerate(recs[:limit]):
        p = download(rec)
        if p is None:
            fail += 1
            continue
        got += 1
        total += p.stat().st_size
        if got % 10 == 0 or i < 5:
            print(f'  [{i + 1}/{limit}] {got} ok, {fail} failed, {total / 1000000.0:.0f} MB')
    print(f'\ndownloaded {got}, failed {fail}, {total / 1000000.0:.1f} MB total -> {PDF_DIR}')
    return 0
if __name__ == '__main__':
    sys.exit(main())
