from services.visualization_service.inference import infer_visualization
from services.visualization_service.schemas import QueryResult
from services.visualization_service.validation import validate_visualization_spec


def test_infers_bar_chart_for_category_counts():
    result = QueryResult(
        columns=["Sex", "Patient_Number"],
        rows=[
            ["Female", 252],
            ["Female", 1800],
            ["Male", 902],
            ["Male", 215],
        ],
    )

    spec = infer_visualization("Patient number by sex", result)

    assert spec is not None
    assert spec.type == "bar"
    assert spec.labels == ["Female", "Male"]
    assert spec.values == [2.0, 2.0]
    assert "duplicate_labels_aggregated" in spec.warnings


def test_infers_line_chart_for_temporal_result():
    result = QueryResult(
        columns=["month", "total_activity"],
        rows=[["2026-01", 10], ["2026-02", 20]],
    )

    spec = infer_visualization("activity trend over time", result)

    assert spec is not None
    assert spec.type == "line"


def test_validation_rejects_bad_chart_shape():
    result = QueryResult(columns=["label", "value"], rows=[["A", 1]])
    spec = infer_visualization("value by label", result)
    assert spec is not None
    spec.values = []

    validation = validate_visualization_spec(spec, result)

    assert validation.valid is False
    assert "missing_values" in validation.issues
