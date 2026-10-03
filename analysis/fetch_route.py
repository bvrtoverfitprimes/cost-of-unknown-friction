import json
import subprocess
import sys
import time
from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / 'data' / 'routes'

def get(url, data=None):
    cmd = ['curl', '-s', '--ssl-no-revoke', '-A', 'friction-research/1.0', url]
    if data is not None:
        cmd += ['--data-urlencode', 'data=' + data]
    for _ in range(4):
        r = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return json.loads(r.stdout)
        except json.JSONDecodeError:
            time.sleep(5)
    raise RuntimeError(r.stdout[:300])

def main():
    name, a, b = (sys.argv[1], sys.argv[2], sys.argv[3])
    osrm = get(f'https://router.project-osrm.org/route/v1/driving/{a};{b}?overview=full&geometries=geojson&annotations=true')
    rt = osrm['routes'][0]
    coords = rt['geometry']['coordinates']
    nodes = rt['legs'][0]['annotation']['nodes']
    print(f"OSRM: {rt['distance']:.0f} m, {len(nodes)} nodes")
    q = '[out:json][timeout:120];node(id:' + ','.join(map(str, nodes)) + ');out tags;way(bn);out body;'
    ov = get('https://overpass-api.de/api/interpreter', data=q)
    node_tags, ways = ({}, [])
    for el in ov['elements']:
        if el['type'] == 'node':
            node_tags[el['id']] = el.get('tags', {})
        elif el['type'] == 'way':
            ways.append(el)
    pair_way = {}
    for w in ways:
        ns = w['nodes']
        for i in range(len(ns) - 1):
            pair_way[ns[i], ns[i + 1]] = w
            pair_way[ns[i + 1], ns[i]] = w
    seg = []
    for i in range(len(nodes) - 1):
        w = pair_way.get((nodes[i], nodes[i + 1]))
        tags = w.get('tags', {}) if w else {}
        seg.append({'maxspeed': tags.get('maxspeed'), 'highway': tags.get('highway')})
    ctrl = [node_tags.get(n, {}).get('highway') for n in nodes]
    import numpy as np
    idx = np.unique(np.linspace(0, len(coords) - 1, min(len(coords), 300)).astype(int))
    sub = []
    for i in range(0, len(idx), 100):
        chunk = [coords[j] for j in idx[i:i + 100]]
        lat = ','.join((f'{c[1]:.6f}' for c in chunk))
        lon = ','.join((f'{c[0]:.6f}' for c in chunk))
        while True:
            e = get(f'https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}')
            if 'elevation' in e:
                break
            print('  elevation API:', e.get('reason'), '- waiting 65 s')
            time.sleep(65)
        sub += e['elevation']
        time.sleep(20)
    elev = list(np.interp(np.arange(len(coords)), idx, sub))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({'name': name, 'from': a, 'to': b, 'coords': coords, 'nodes': nodes, 'segments': seg, 'controls': ctrl, 'elevation': elev, 'osrm_distance': rt['distance'], 'sources': ['OSRM demo server / OpenStreetMap', 'Overpass API / OpenStreetMap', 'Open-Meteo elevation API (Copernicus DEM GLO-90)'], 'fetched': time.strftime('%Y-%m-%d')}, open(OUT / f'{name}.json', 'w'))
    ms = [s['maxspeed'] for s in seg]
    print(f'maxspeed tagged on {sum((m is not None for m in ms))}/{len(ms)} segments; values {sorted(set((m for m in ms if m)))}')
    print(f"controls: {sum((c in ('traffic_signals', 'stop') for c in ctrl))} signals/stops")
    print(f'elevation {min(elev):.0f}-{max(elev):.0f} m')
if __name__ == '__main__':
    main()
