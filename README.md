# tool-patient-export-umap

Embeds cohort patient exports with UMAP and produces an interactive plot, embedding coordinates, and a feature report.

Patient UMAP Export is an **Export app**. It receives the patient data export of a cohort,
embeds every patient in two dimensions with [UMAP](https://umap-learn.readthedocs.io/) and
renders the result as an interactive scatter plot.

It is started from the cohort overview: **Export → App-based export**, pick this app, adjust
the hyperparameters and press **Export**. The plot and tables are shown in the dialog once the
run has finished and can be downloaded from there.

## What this tool does

1. **Pivot** – the export arrives in long format (one row per stored value). It is pivoted to
   one row per patient and one column per feature. The feature label is the export feature
   name, or `ontologyId:datatypeId` when no feature selection was configured.
   When a patient holds several values for a feature (e.g. one per visit), `aggregation`
   decides which one is used.
2. **Type and encode** – a feature is numeric when at least 90 % of its values parse as
   numbers (`true/false/yes/no` count as 1/0). Numeric features are median-imputed,
   categorical features one-hot encoded. Features are dropped when they are too sparse
   (`min_feature_coverage`), constant, or have more than `max_categories` values.
3. **Scale** – optional standardisation to zero mean and unit variance.
4. **Embed** – UMAP with the configured neighbourhood, distance and metric.
5. **Plot** – a self-contained HTML page (inline SVG, no scripts or external assets). Hover a
   point to see the patient id and colour value.

A table that already has one row per patient (a native wide-format export) is accepted as well;
its `patientId` column (or the first column) is used as the patient id.

## Configuration (Hyperparameters)

| Name | Default | Description |
|---|---|---|
| `n_neighbors` | 15 | Local neighbourhood size; capped at patients − 1 |
| `min_dist` | 0.1 | Minimum distance between embedded points |
| `metric` | euclidean | `euclidean`, `manhattan`, `cosine`, `correlation`, `chebyshev` |
| `random_state` | 42 | Seed; negative for a faster, non-deterministic run |
| `aggregation` | last | `last` / `first` (by visit timestamp) or `mean` |
| `min_feature_coverage` | 0.5 | Share of patients that need a value for a feature |
| `include_categorical` | true | One-hot encode non-numeric features |
| `max_categories` | 20 | Drop categorical features with more values |
| `scale` | true | Standardise features before embedding |
| `color_by` | *(empty)* | Feature to colour the points by |
| `exclude_color_feature` | true | Leave the `color_by` feature out of the embedding |

With `exclude_color_feature` enabled the plot answers *"do the other features separate this
label?"*, e.g. colour by diagnosis without the diagnosis shaping the embedding.

## Input

- `input` (CSV, required): the patient data export.

## Outputs

- `plot` (HTML): interactive UMAP scatter plot.
- `embedding` (CSV): `patientId`, `umap_1`, `umap_2` and the `color_by` feature if set.
- `feature_report` (CSV): every feature with its kind, coverage, number of model columns and
  the reason it was dropped.

The UMAP axes have no unit; only distances between points carry meaning.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
```

`data/patient_export_sample.csv` is a small synthetic export in the format the local learning
API produces.

## Container and local runtime

The Docker base image provides the FL-Net API. `requirements.txt` lists the
additional runtime dependencies; `requirements-dev.txt` also installs the API
and pytest for local development.

Copy `.env.example` to `.env`, then set `APP_ID` and the platform URLs before
starting the tool with `python main.py`.

Build with `docker build -t tool-patient-export-umap .`.
The image runs as `nonroot` and keeps the Numba cache under `/tmp/numba_cache`.
