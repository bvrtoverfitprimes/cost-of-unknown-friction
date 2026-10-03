# Unknown friction in minimum-time vehicle transit

This repository holds data, tables, code, and formal proofs on the cost of unknown tire-road friction in
minimum-time vehicle transit.

## Contents

| Path | Contents |
| --- | --- |
| `data/*.json` | The computed values behind the tables, one record per analysis. |
| `data/epa_roadload_derived.csv` | Drag area and rolling resistance implied by the EPA road-load coefficients of model year 2013. |
| `tables/*.tex` | The generated LaTeX tables. |
| `analysis/*.py` | The analyses that produce the records in `data/`, the solver and vehicle models they use, and the scripts that fetch the public datasets. Run them from the repository root. |
| `data/routes/`, `data/windsor/`, `data/utqg/` | Route summaries and small reference files used by the analyses. |
| `code/make_tables.py` | Formats the records in `data/` into the tables. It does not rerun the analyses that produce the records. |
| `lean/` | Formal versions of structural results on the acceleration envelope, for Lean 4 with Mathlib. |

## Regenerating the tables

```
pip install -r code/requirements.txt
python code/make_tables.py
```

This rewrites the files in `tables/` from the records in `data/`. The tables in `fmvss.tex`,
`d3data.tex`, `d3bounds.tex`, `info.tex`, and `zero60.tex` are computed from the raw public datasets
listed below by the full analysis, and are included in `tables/` as computed. The table in
`mfcoeffs.tex` lists the tire model coefficients used by the full analysis and is also included as
computed.

## Checking the formal proofs

```
cd lean
lake exe cache get
lake build
```

## Data sources

The measured data are public and are not redistributed here.

- U.S. Environmental Protection Agency, Fuel Economy Test Car List Data, model year 2013.
- Argonne National Laboratory, Downloadable Dynamometer Database.
- National Highway Traffic Safety Administration, FMVSS No. 135 compliance test reports.
- Route summaries are derived from OpenStreetMap data, © OpenStreetMap contributors, available under
  the Open Database License (ODbL).
- Elevation summaries are derived from the Copernicus GLO-90 digital elevation model of the European
  Space Agency, accessed through the Open-Meteo elevation API.

## Licence

The code in `code/` and `lean/` is released under the MIT License (`LICENSE`). The tables and the
data records are released under the Creative Commons Attribution 4.0
International License (`LICENSE-CC-BY-4.0.md`). Third-party data keep the terms of their sources.

## Citation

Please cite this repository (see `CITATION.cff`).
