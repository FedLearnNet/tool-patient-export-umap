"""
Turns a patient data export into a two-dimensional UMAP embedding.

The local learning API hands export apps the patient data in long format - one row per stored
value with the columns ``patientId``, ``name`` (or ``ontologyId``/``datatypeId``), ``value`` and
the visit information. UMAP needs one row per patient, so the rows are pivoted first, every
feature is typed and encoded, and only then is the embedding computed.

This module is kept free of pyfedappwrap so it can be tested without the app runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

PATIENT_ID = "patientId"
LONG_FORMAT_COLUMNS = {PATIENT_ID, "value"}

AGGREGATIONS = ("last", "first", "mean")
METRICS = ("euclidean", "manhattan", "cosine", "correlation", "chebyshev")

# UMAP needs a neighbourhood, so a handful of patients is the least that yields a picture
MIN_PATIENTS = 4
# Share of a feature's present values that must parse as numbers for it to count as numeric
NUMERIC_SHARE = 0.9
_BOOLEAN_VALUES = {"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0}


@dataclass
class PipelineSettings:
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"
    random_state: int = 42
    aggregation: str = "last"
    min_feature_coverage: float = 0.5
    include_categorical: bool = True
    max_categories: int = 20
    scale: bool = True
    color_by: str = ""
    exclude_color_feature: bool = True

    def validate(self) -> None:
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got '{self.aggregation}'")
        if self.metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got '{self.metric}'")
        if self.n_neighbors < 2:
            raise ValueError("n_neighbors must be at least 2")
        if not 0.0 <= self.min_dist <= 1.0:
            raise ValueError("min_dist must be between 0 and 1")
        if not 0.0 <= self.min_feature_coverage <= 1.0:
            raise ValueError("min_feature_coverage must be between 0 and 1")
        if self.max_categories < 2:
            raise ValueError("max_categories must be at least 2")


@dataclass
class FeatureInfo:
    feature: str
    kind: str
    coverage: float
    columns: int = 0
    reason: str = ""


@dataclass
class PipelineResult:
    embedding: pd.DataFrame
    features: list[FeatureInfo]
    n_neighbors: int
    color_column: Optional[str] = None
    color_kind: Optional[str] = None
    notes: list[str] = field(default_factory=list)

    @property
    def used_features(self) -> list[FeatureInfo]:
        return [f for f in self.features if f.kind != "dropped"]

    def features_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "feature": f.feature,
            "kind": f.kind,
            "coverage": round(f.coverage, 4),
            "model_columns": f.columns,
            "reason": f.reason,
        } for f in self.features], columns=["feature", "kind", "coverage", "model_columns", "reason"])


def _feature_names(df: pd.DataFrame) -> pd.Series:
    """Feature label of every long-format row: the export feature name, else ontology:datatype."""
    names = pd.Series(pd.NA, index=df.index, dtype="string")
    if "name" in df.columns:
        named = df["name"].astype("string").str.strip()
        names = named.where(named.notna() & (named != ""))
    if "ontologyId" in df.columns:
        fallback = df["ontologyId"].astype("string").str.strip()
        if "datatypeId" in df.columns:
            datatype = df["datatypeId"].astype("string").str.strip()
            fallback = fallback.where(datatype.isna() | (datatype == ""), fallback + ":" + datatype)
        names = names.where(names.notna(), fallback)
    return names


def to_wide(df: pd.DataFrame, aggregation: str = "last") -> pd.DataFrame:
    """
    Pivots the export to one row per patient and one column per feature.

    A patient can hold several values for the same feature (one per visit). ``last`` and ``first``
    pick by visit timestamp when there is one and by export order otherwise; ``mean`` averages
    numeric values and keeps the last value of anything that does not parse as a number.
    """
    if not LONG_FORMAT_COLUMNS.issubset(df.columns):
        # Already one row per patient, e.g. a native wide-format export
        id_column = PATIENT_ID if PATIENT_ID in df.columns else df.columns[0]
        wide = df.drop_duplicates(subset=id_column, keep="last").set_index(id_column)
        wide.index = wide.index.astype(str)
        wide.index.name = PATIENT_ID
        return wide.drop(columns=[c for c in ("id",) if c in wide.columns])

    rows = pd.DataFrame({
        PATIENT_ID: df[PATIENT_ID].astype("string"),
        "feature": _feature_names(df) if ("name" in df.columns or "ontologyId" in df.columns)
        else pd.Series(pd.NA, index=df.index, dtype="string"),
        "value": df["value"],
        "_row": np.arange(len(df)),
    })
    if rows["feature"].isna().all():
        raise ValueError("The export has no feature column: expected 'name' or 'ontologyId'")

    has_timestamp = "visitTimestamp" in df.columns
    if has_timestamp:
        rows["_visit"] = pd.to_datetime(df["visitTimestamp"], errors="coerce", utc=True, format="mixed")

    present = rows["value"].notna() & (rows["value"].astype("string").str.strip() != "")
    rows = rows[present & rows[PATIENT_ID].notna() & rows["feature"].notna()]
    if rows.empty:
        raise ValueError("The export contains no values")

    # Rows without a visit date sort first, so "last" prefers the most recent dated value
    order = ["_visit", "_row"] if has_timestamp else ["_row"]
    rows = rows.sort_values(order, na_position="first", kind="stable")

    grouped = rows.groupby([PATIENT_ID, "feature"], sort=False)["value"]
    values = grouped.first() if aggregation == "first" else grouped.last()
    if aggregation == "mean":
        numeric = rows.assign(value=pd.to_numeric(rows["value"], errors="coerce")).dropna(subset=["value"])
        means = numeric.groupby([PATIENT_ID, "feature"])["value"].mean()
        values = values.astype(object)
        values.loc[means.index] = means.astype(object)

    wide = values.unstack("feature")
    wide.columns.name = None
    wide.index = wide.index.astype(str)
    wide.index.name = PATIENT_ID
    return wide


def as_numeric(column: pd.Series) -> Optional[pd.Series]:
    """The column as floats when it is numeric or boolean, otherwise None."""
    present = column.dropna()
    if present.empty:
        return None
    text = present.astype(str).str.strip().str.lower()
    if text.isin(_BOOLEAN_VALUES.keys()).all():
        return column.map(lambda v: np.nan if pd.isna(v) else _BOOLEAN_VALUES[str(v).strip().lower()]).astype(float)
    numeric = pd.to_numeric(column, errors="coerce")
    if numeric.notna().sum() >= NUMERIC_SHARE * present.size:
        return numeric.astype(float)
    return None


def encode(wide: pd.DataFrame, settings: PipelineSettings) -> tuple[pd.DataFrame, list[FeatureInfo]]:
    """
    Builds the numeric matrix UMAP runs on.

    Numeric features are median-imputed, categorical ones one-hot encoded (a missing value is
    all zeros). Features that are too sparse, constant or have too many categories are dropped
    and reported with the reason.
    """
    blocks: list[pd.DataFrame] = []
    features: list[FeatureInfo] = []

    for name in wide.columns:
        column = wide[name]
        coverage = float(column.notna().mean()) if len(column) else 0.0
        if coverage < settings.min_feature_coverage or coverage == 0.0:
            features.append(FeatureInfo(str(name), "dropped", coverage,
                                        reason=f"coverage below {settings.min_feature_coverage:.0%}"))
            continue

        numeric = as_numeric(column)
        if numeric is not None:
            if numeric.nunique(dropna=True) <= 1:
                features.append(FeatureInfo(str(name), "dropped", coverage, reason="constant"))
                continue
            blocks.append(numeric.fillna(numeric.median()).rename(str(name)).to_frame())
            features.append(FeatureInfo(str(name), "numeric", coverage, columns=1))
            continue

        if not settings.include_categorical:
            features.append(FeatureInfo(str(name), "dropped", coverage, reason="categorical features disabled"))
            continue
        categories = column.astype("string").str.strip()
        count = int(categories.nunique(dropna=True))
        if count <= 1:
            features.append(FeatureInfo(str(name), "dropped", coverage, reason="constant"))
            continue
        if count > settings.max_categories:
            features.append(FeatureInfo(str(name), "dropped", coverage,
                                        reason=f"{count} categories (max {settings.max_categories})"))
            continue
        dummies = pd.get_dummies(categories, prefix=str(name), prefix_sep="=", dtype=float)
        blocks.append(dummies)
        features.append(FeatureInfo(str(name), "categorical", coverage, columns=int(dummies.shape[1])))

    if not blocks:
        raise ValueError("No usable features left after filtering; lower min_feature_coverage "
                         "or enable categorical features")
    return pd.concat(blocks, axis=1), features


def embed(matrix: np.ndarray, settings: PipelineSettings) -> tuple[np.ndarray, int]:
    """Runs UMAP; returns the 2D coordinates and the neighbourhood size actually used."""
    import umap  # imported lazily, numba compilation makes the import slow

    samples = matrix.shape[0]
    n_neighbors = max(2, min(settings.n_neighbors, samples - 1))
    random_state = settings.random_state if settings.random_state is not None and settings.random_state >= 0 else None
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=settings.min_dist,
        metric=settings.metric,
        random_state=random_state,
        # Spectral initialisation is unstable on very small graphs
        init="spectral" if samples > 2 * n_neighbors else "random",
    )
    return np.asarray(reducer.fit_transform(matrix), dtype=float), n_neighbors


def run_pipeline(df: pd.DataFrame, settings: PipelineSettings,
                 log: Callable[[str], None] = lambda _: None) -> PipelineResult:
    settings.validate()
    if df is None or df.empty:
        raise ValueError("The export is empty")

    wide = to_wide(df, settings.aggregation)
    log(f"Pivoted {len(df)} export rows to {wide.shape[0]} patients x {wide.shape[1]} features")
    if len(wide) < MIN_PATIENTS:
        raise ValueError(f"UMAP needs at least {MIN_PATIENTS} patients, the export has {len(wide)}")

    notes: list[str] = []
    color_column = None
    color_values = None
    color_kind = None
    color_by = (settings.color_by or "").strip()
    if color_by:
        if color_by in wide.columns:
            color_column = color_by
            numeric = as_numeric(wide[color_by])
            color_kind = "numeric" if numeric is not None else "categorical"
            color_values = numeric if numeric is not None else wide[color_by].astype("string")
            if settings.exclude_color_feature:
                wide = wide.drop(columns=[color_by])
        else:
            notes.append(f"color_by feature '{color_by}' is not part of the export; points are not coloured")

    matrix, features = encode(wide, settings)
    used = [f for f in features if f.kind != "dropped"]
    log(f"Using {len(used)} of {len(features)} features ({matrix.shape[1]} model columns)")

    values = matrix.to_numpy(dtype=float)
    if settings.scale:
        from sklearn.preprocessing import StandardScaler
        values = StandardScaler().fit_transform(values)

    coordinates, n_neighbors = embed(values, settings)
    if n_neighbors != settings.n_neighbors:
        notes.append(f"n_neighbors reduced from {settings.n_neighbors} to {n_neighbors} for {len(wide)} patients")
    log(f"UMAP finished with n_neighbors={n_neighbors}, min_dist={settings.min_dist}, metric={settings.metric}")

    embedding = pd.DataFrame({
        PATIENT_ID: wide.index,
        "umap_1": coordinates[:, 0],
        "umap_2": coordinates[:, 1],
    })
    if color_column is not None:
        embedding[color_column] = color_values.reindex(wide.index).to_numpy()

    return PipelineResult(
        embedding=embedding,
        features=features,
        n_neighbors=n_neighbors,
        color_column=color_column,
        color_kind=color_kind,
        notes=notes,
    )
