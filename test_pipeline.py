import numpy as np
import pandas as pd
import pytest

from plot import render_html, viridis
from umap_pipeline import PipelineSettings, as_numeric, encode, run_pipeline, to_wide


def long_export(patients: int = 40, seed: int = 0) -> pd.DataFrame:
    """Two well separated patient groups in the long format the local API exports."""
    rng = np.random.default_rng(seed)
    rows = []
    for patient in range(1, patients + 1):
        group = "A" if patient <= patients // 2 else "B"
        centre = 0.0 if group == "A" else 8.0
        values = {
            "age": rng.normal(40 + centre * 3, 3),
            "bmi": rng.normal(22 + centre, 1),
            "hba1c": rng.normal(5 + centre / 2, 0.3),
            "smoker": "yes" if rng.random() < 0.5 else "no",
            "diagnosis": "type 1" if group == "A" else "type 2",
        }
        for name, value in values.items():
            rows.append({"id": len(rows) + 1, "patientId": patient, "name": name, "value": value,
                         "visitId": 1, "visitTimestamp": "2026-01-01 10:00:00"})
    return pd.DataFrame(rows)


def test_to_wide_picks_latest_visit_and_falls_back_to_ontology_ids():
    df = pd.DataFrame([
        {"patientId": 1, "ontologyId": "LOINC", "datatypeId": "hba1c", "value": "6.1", "visitTimestamp": "2026-03-01"},
        {"patientId": 1, "ontologyId": "LOINC", "datatypeId": "hba1c", "value": "7.4", "visitTimestamp": "2026-05-01"},
        {"patientId": 1, "ontologyId": "LOINC", "datatypeId": "hba1c", "value": "5.0", "visitTimestamp": "2026-01-01"},
        {"patientId": 2, "ontologyId": "LOINC", "datatypeId": "hba1c", "value": "", "visitTimestamp": "2026-01-01"},
    ])
    wide = to_wide(df, "last")
    assert list(wide.columns) == ["LOINC:hba1c"]
    assert wide.loc["1", "LOINC:hba1c"] == "7.4"
    # an empty value is not a value
    assert "2" not in wide.index

    assert to_wide(df, "first").loc["1", "LOINC:hba1c"] == "5.0"
    assert to_wide(df, "mean").loc["1", "LOINC:hba1c"] == pytest.approx((6.1 + 7.4 + 5.0) / 3)


def test_to_wide_accepts_one_row_per_patient():
    df = pd.DataFrame({"patientId": [1, 2, 3], "age": [30, 40, 50]})
    wide = to_wide(df)
    assert list(wide.index) == ["1", "2", "3"]
    assert list(wide.columns) == ["age"]


def test_as_numeric_understands_booleans_and_rejects_text():
    assert as_numeric(pd.Series(["true", "false", None])).tolist()[:2] == [1.0, 0.0]
    assert as_numeric(pd.Series(["1.5", "2", None])).tolist()[:2] == [1.5, 2.0]
    assert as_numeric(pd.Series(["type 1", "type 2"])) is None


def test_encode_drops_sparse_constant_and_high_cardinality_features():
    wide = pd.DataFrame({
        "age": [30, 40, 50, 60],
        "sparse": [1, None, None, None],
        "constant": [1, 1, 1, 1],
        "sex": ["f", "m", "f", None],
        "code": ["a", "b", "c", "d"],
    }, index=["1", "2", "3", "4"])
    matrix, features = encode(wide, PipelineSettings(max_categories=3))
    kinds = {f.feature: (f.kind, f.reason) for f in features}
    assert kinds["age"][0] == "numeric"
    assert kinds["sex"][0] == "categorical"
    assert kinds["sparse"] == ("dropped", "coverage below 50%")
    assert kinds["constant"] == ("dropped", "constant")
    assert kinds["code"][0] == "dropped" and "categories" in kinds["code"][1]
    assert list(matrix.columns) == ["age", "sex=f", "sex=m"]
    # the missing sex becomes all zeros rather than a category of its own
    assert matrix.loc["4", ["sex=f", "sex=m"]].tolist() == [0.0, 0.0]


def test_encode_fails_without_usable_features():
    wide = pd.DataFrame({"constant": [1, 1, 1, 1]}, index=["1", "2", "3", "4"])
    with pytest.raises(ValueError, match="No usable features"):
        encode(wide, PipelineSettings())


def test_run_pipeline_separates_the_two_groups_and_colours_by_diagnosis():
    result = run_pipeline(long_export(), PipelineSettings(color_by="diagnosis"))

    embedding = result.embedding
    assert list(embedding.columns) == ["patientId", "umap_1", "umap_2", "diagnosis"]
    assert len(embedding) == 40
    assert result.color_kind == "categorical"
    # the colour feature is not part of the embedding
    assert "diagnosis" not in [f.feature for f in result.features]

    # UMAP places separate clusters arbitrarily, so check neighbourhoods rather than distances:
    # the nearest embedded neighbours of a patient should share its diagnosis
    points = embedding[["umap_1", "umap_2"]].to_numpy()
    labels = embedding["diagnosis"].to_numpy()
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    neighbours = np.argsort(distances, axis=1)[:, :5]
    purity = (labels[neighbours] == labels[:, None]).mean()
    assert purity > 0.95


def test_run_pipeline_is_reproducible_with_a_seed():
    first = run_pipeline(long_export(), PipelineSettings()).embedding
    second = run_pipeline(long_export(), PipelineSettings()).embedding
    pd.testing.assert_frame_equal(first, second)


def test_run_pipeline_shrinks_the_neighbourhood_for_small_cohorts():
    result = run_pipeline(long_export(patients=6), PipelineSettings(n_neighbors=15))
    assert result.n_neighbors == 5
    assert any("n_neighbors reduced" in note for note in result.notes)


def test_run_pipeline_rejects_tiny_cohorts_and_bad_settings():
    with pytest.raises(ValueError, match="at least"):
        run_pipeline(long_export(patients=2), PipelineSettings())
    with pytest.raises(ValueError, match="metric"):
        run_pipeline(long_export(), PipelineSettings(metric="nope"))


def test_unknown_colour_feature_is_reported_not_fatal():
    result = run_pipeline(long_export(), PipelineSettings(color_by="missing"))
    assert result.color_column is None
    assert any("missing" in note for note in result.notes)


def test_render_html_is_self_contained_and_escapes_labels():
    df = long_export()
    df.loc[df["name"] == "diagnosis", "value"] = "<script>alert(1)</script>"
    df.loc[(df["name"] == "diagnosis") & (df["patientId"] > 20), "value"] = "b&c"
    settings = PipelineSettings(color_by="diagnosis")
    page = render_html(run_pipeline(df, settings), settings)

    assert page.count("<circle") == 40
    assert "<script" not in page
    assert "&lt;script&gt;" in page
    assert "http://" not in page and "https://" not in page


def test_render_html_uses_a_colour_scale_for_numeric_features():
    settings = PipelineSettings(color_by="age")
    page = render_html(run_pipeline(long_export(), settings), settings)
    assert "linearGradient" in page


def test_viridis_endpoints():
    assert viridis(0) == "#440154"
    assert viridis(1) == "#fde725"
