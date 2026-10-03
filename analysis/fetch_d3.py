from __future__ import annotations
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36'
DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'd3'

def _curl(url: str, referer: str | None=None, tries: int=4) -> str:
    cmd = ['curl', '-sS', '-L', '-A', UA, '-w', '\n__HTTP__%{http_code}']
    if referer:
        cmd += ['-H', f'Referer: {referer}']
    cmd.append(url)
    delay = 2.0
    for attempt in range(tries):
        out = subprocess.run(cmd, capture_output=True, timeout=120)
        body = out.stdout.decode('utf-8', errors='replace')
        body, _, code = body.rpartition('\n__HTTP__')
        if code == '404':
            raise LookupError(f'404 {url}')
        if code == '200' and 'Just a moment' not in body[:2000]:
            return body
        if attempt < tries - 1:
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f'could not fetch {url} (last HTTP {code})')

def shared_token(vehicle_slug: str) -> str:
    html = _curl(f'https://www.anl.gov/taps/{vehicle_slug}')
    m = re.search('app\\.box\\.com/embed_widget/s/([a-z0-9]+)', html)
    if not m:
        raise LookupError(f'no Box iframe on /taps/{vehicle_slug}')
    return m.group(1)

@dataclass
class BoxItem:
    file_id: int
    name: str
    size: int
    ext: str

    @property
    def download_url(self) -> str:
        return self._url

    def set_url(self, shared_name: str) -> 'BoxItem':
        self._url = f'https://app.box.com/index.php?rm=box_download_shared_file&shared_name={shared_name}&file_id=f_{self.file_id}'
        return self

def list_folder(shared_name: str) -> list[BoxItem]:
    html = _curl(f'https://app.box.com/s/{shared_name}')
    items: list[BoxItem] = []
    seen: set[int] = set()
    for m in re.finditer('\\{"typedID":"f_(\\d+)","type":"file".*?"extension":"([^"]*)","name":"((?:[^"\\\\]|\\\\.)*)","itemSize":(\\d+)', html):
        fid = int(m.group(1))
        if fid in seen:
            continue
        seen.add(fid)
        items.append(BoxItem(file_id=fid, name=json.loads(f'"{m.group(3)}"'), size=int(m.group(4)), ext=m.group(2)).set_url(shared_name))
    return items

def download(item: BoxItem, dest: Path, shared_name: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == item.size:
        return dest
    subprocess.run(['curl', '-sS', '-L', '-A', UA, '-H', f'Referer: https://app.box.com/s/{shared_name}', '-o', str(dest), item.download_url], check=True, timeout=600)
    return dest

def fetch_vehicle(vehicle_slug: str, want_ext: str='zip') -> list[Path]:
    sn = shared_token(vehicle_slug)
    got = []
    for it in list_folder(sn):
        if it.ext != want_ext:
            continue
        got.append(download(it, DATA_DIR / vehicle_slug / it.name, sn))
    return got
if __name__ == '__main__':
    import sys
    for slug in sys.argv[1:]:
        try:
            paths = fetch_vehicle(slug)
        except Exception as exc:
            print(f'{slug:40s} FAILED  {exc}')
            continue
        for p in paths:
            print(f'{slug:40s} {p.name:45s} {p.stat().st_size / 1000000.0:7.2f} MB')
        if not paths:
            print(f'{slug:40s} (no zip archives found)')
