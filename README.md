# The Cost of Unknown Friction in Minimum-Time Vehicle Transit: supplementary material

This repository holds the supplementary material for the paper The Cost of Unknown Friction in
Minimum-Time Vehicle Transit. It contains the tables behind every
numerical statement of the paper, the computed values from which those tables are generated, the code
that generates them, and the proofs of the results on the transient deficit.

## Contents

| Path | Contents |
| --- | --- |
| `supplement/supplement.pdf` | The supplementary material. Section S1 gives the numerical evidence (Tables S1 to S42), and Section S2 gives the proofs for the transient deficit. |
| `supplement/supplement.tex` | LaTeX source of the supplementary material. It compiles on its own with `pdflatex`. |
| `data/*.json` | The computed values behind the tables, one record per analysis. |
| `data/epa_roadload_derived.csv` | Drag area and rolling resistance implied by the EPA road-load coefficients of model year 2013. |
| `tables/*.tex` | The generated LaTeX tables, as they appear in the supplement and the paper. |
| `code/make_tables.py` | Regenerates the tables from `data/`. |
| `lean/` | Formal versions of the structural results, for Lean 4 with Mathlib. |

## Regenerating the tables

```
pip install -r code/requirements.txt
python code/make_tables.py
```

This rewrites the files in `tables/` from the records in `data/`. The tables in `fmvss.tex`,
`d3data.tex`, `d3bounds.tex`, `info.tex`, and `zero60.tex` are computed from the raw public datasets
listed below by the full analysis, and are included in `tables/` as computed.

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

The code in `code/` and `lean/` is released under the MIT License (`LICENSE`). The supplementary
material, the tables, and the data records are released under the Creative Commons Attribution 4.0
International License (`LICENSE-CC-BY-4.0.md`). Third-party data keep the terms of their sources.

## Citation

If you use this material, please cite the paper and this repository (see `CITATION.cff`).
