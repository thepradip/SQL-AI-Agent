"""Validation helpers for visualization specs."""

from __future__ import annotations

from .inference import normalize_records
from .schemas import QueryResult, ValidationResponse, VisualizationSpec


def validate_visualization_spec(visualization: VisualizationSpec | None, result: QueryResult) -> ValidationResponse:
    if visualization is None:
        return ValidationResponse(valid=False, score=0.0, issues=["missing_visualization"])

    issues: list[str] = []
    warnings: list[str] = list(visualization.warnings)
    records = normalize_records(result)

    if visualization.type in {"bar", "line", "pie"}:
        if not visualization.labels:
            issues.append("missing_labels")
        if not visualization.values:
            issues.append("missing_values")
        if visualization.labels and visualization.values and len(visualization.labels) != len(visualization.values):
            issues.append("labels_values_length_mismatch")
        if visualization.values and any(value < 0 for value in visualization.values) and visualization.type == "pie":
            issues.append("pie_values_must_be_non_negative")
        if visualization.label_key and result.columns and visualization.label_key not in result.columns:
            issues.append("label_key_not_in_result_columns")
        raw_value_key = (visualization.value_key or "").removesuffix("_sum").removesuffix("_count")
        if visualization.value_key and result.columns and raw_value_key not in result.columns and visualization.value_key not in result.columns:
            warnings.append("derived_value_key")

    if visualization.type == "number" and visualization.number_value is None:
        issues.append("missing_number_value")

    if visualization.records and len(visualization.records) > max(len(records), 30):
        warnings.append("visualization_records_exceed_result_preview")

    score = max(0.0, 1.0 - (0.25 * len(issues)) - (0.05 * len(warnings)))
    return ValidationResponse(
        valid=not issues,
        score=round(score, 4),
        issues=issues or ["none"],
        warnings=warnings,
    )
