"""Deterministic visualization inference for SQL query results."""

from __future__ import annotations

from collections import OrderedDict
from datetime import date, datetime
from numbers import Number
from typing import Any

from .schemas import QueryResult, VisualizationSpec

MAX_CHART_POINTS = 30


def infer_visualization(
    question: str,
    result: QueryResult,
    commentary: str | None = None,
    preferred_type: str | None = None,
) -> VisualizationSpec | None:
    records = normalize_records(result)
    if not result.columns or not records:
        return None

    numeric_columns = [column for column in result.columns if _is_numeric_column(records, column)]
    categorical_columns = [column for column in result.columns if column not in numeric_columns]

    if len(records) == 1:
        number_spec = _build_number_spec(records[0], result.columns, numeric_columns, categorical_columns)
        if number_spec:
            return number_spec

    chart_spec = _build_chart_spec(question, records, result.columns, numeric_columns, categorical_columns, preferred_type)
    if chart_spec:
        chart_spec.description = chart_spec.description or _description_for(chart_spec.type, commentary)
        return chart_spec

    return VisualizationSpec(
        type="table",
        title="Result table",
        description="The result is best represented as tabular data.",
        records=records,
        confidence=0.65,
        warnings=["no_chartable_numeric_series"],
    )


def normalize_records(result: QueryResult) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in result.rows:
        if isinstance(row, dict):
            records.append({column: _safe_value(row.get(column)) for column in result.columns})
            continue
        records.append({
            column: _safe_value(row[index]) if index < len(row) else None
            for index, column in enumerate(result.columns)
        })
    return records


def _build_number_spec(
    record: dict[str, Any],
    columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> VisualizationSpec | None:
    if len(columns) == 1:
        column = columns[0]
        return VisualizationSpec(
            type="number",
            title="Query result",
            number_value=_coerce_number_value(record.get(column)),
            number_label=_pretty_label(column),
            confidence=0.95,
        )

    if len(numeric_columns) == 1:
        value_key = numeric_columns[0]
        label_key = categorical_columns[0] if categorical_columns else value_key
        context = ", ".join(
            f"{_pretty_label(column)}: {record[column]}"
            for column in categorical_columns[1:4]
            if record.get(column) is not None
        ) or None
        return VisualizationSpec(
            type="number",
            title="Key metric",
            number_value=_coerce_number_value(record.get(value_key)),
            number_label=_pretty_label(label_key),
            number_context=context,
            confidence=0.9,
        )

    return None


def _build_chart_spec(
    question: str,
    records: list[dict[str, Any]],
    columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    preferred_type: str | None,
) -> VisualizationSpec | None:
    if not numeric_columns:
        return None

    value_key = _choose_value_key(question, numeric_columns)
    label_key = _choose_label_key(question, columns, categorical_columns, value_key)
    if not label_key:
        return None

    series = _prepare_series(records, label_key, value_key)
    if not series["labels"] or any(value is None for value in series["values"]):
        return None

    chart_type = _choose_chart_type(question, series["labels"], label_key, len(series["records"]), preferred_type)
    warnings = list(series["warnings"])
    if chart_type == "pie" and len(series["records"]) > 8:
        chart_type = "bar"
        warnings.append("pie_chart_too_many_slices")

    if len(series["records"]) > MAX_CHART_POINTS:
        warnings.append("chart_points_limited")

    return VisualizationSpec(
        type=chart_type,
        title=f"{_pretty_label(series['value_key'])} by {_pretty_label(label_key)}",
        description=series["description"],
        label_key=label_key,
        value_key=series["value_key"],
        x_key=label_key,
        y_key=series["value_key"],
        labels=series["labels"][:MAX_CHART_POINTS],
        values=series["values"][:MAX_CHART_POINTS],
        records=series["records"][:MAX_CHART_POINTS],
        confidence=0.86 if warnings else 0.92,
        warnings=warnings,
    )


def _prepare_series(records: list[dict[str, Any]], label_key: str, value_key: str) -> dict[str, Any]:
    labels = [_stringify_label(record.get(label_key), index) for index, record in enumerate(records, start=1)]
    values = [_coerce_float(record.get(value_key)) for record in records]
    warnings: list[str] = []

    if any(value is None for value in values):
        return {
            "records": records,
            "labels": labels,
            "values": values,
            "value_key": value_key,
            "description": None,
            "warnings": ["non_numeric_chart_values"],
        }

    if len(set(labels)) == len(labels):
        return {
            "records": records,
            "labels": labels,
            "values": values,
            "value_key": value_key,
            "description": None,
            "warnings": warnings,
        }

    aggregate_mode = "count" if _looks_like_identifier(value_key) else "sum"
    grouped: OrderedDict[str, list[float]] = OrderedDict()
    for label, value in zip(labels, values):
        grouped.setdefault(label, []).append(float(value))

    aggregate_key = f"{label_key}_{aggregate_mode}" if aggregate_mode == "count" else f"{value_key}_sum"
    aggregate_labels = list(grouped.keys())
    aggregate_values = [
        float(len(grouped[label])) if aggregate_mode == "count" else round(float(sum(grouped[label])), 4)
        for label in aggregate_labels
    ]
    aggregate_records = [
        {label_key: label, aggregate_key: value}
        for label, value in zip(aggregate_labels, aggregate_values)
    ]

    return {
        "records": aggregate_records,
        "labels": aggregate_labels,
        "values": aggregate_values,
        "value_key": aggregate_key,
        "description": "Aggregated duplicate labels for a readable chart.",
        "warnings": ["duplicate_labels_aggregated"],
    }


def _choose_value_key(question: str, numeric_columns: list[str]) -> str:
    lowered = question.lower()
    for column in numeric_columns:
        if column.lower() in lowered or _pretty_label(column).lower() in lowered:
            return column
    for hint in ("count", "total", "sum", "avg", "average", "score", "rate", "amount"):
        for column in numeric_columns:
            if hint in column.lower():
                return column
    return numeric_columns[0]


def _choose_label_key(question: str, columns: list[str], categorical_columns: list[str], value_key: str) -> str | None:
    lowered = question.lower()
    for column in categorical_columns:
        if column.lower() in lowered or _pretty_label(column).lower() in lowered:
            return column
    if categorical_columns:
        return categorical_columns[0]
    return next((column for column in columns if column != value_key), None)


def _choose_chart_type(
    question: str,
    labels: list[str],
    label_key: str,
    row_count: int,
    preferred_type: str | None,
) -> str:
    allowed = {"bar", "line", "pie"}
    if preferred_type in allowed:
        return preferred_type

    lowered = question.lower()
    temporal_label = _looks_temporal_key(label_key) or sum(_looks_temporal_value(label) for label in labels[:5]) >= 2
    if temporal_label or any(token in lowered for token in ("trend", "over time", "timeline", "line chart")):
        return "line"

    pie_hints = ("share", "distribution", "breakdown", "percentage", "percent", "composition", "ratio", "pie")
    if row_count <= 6 and any(hint in lowered for hint in pie_hints):
        return "pie"

    return "bar"


def _description_for(chart_type: str, commentary: str | None) -> str:
    if commentary:
        return "Chart generated from the SQL result and paired with the response commentary."
    return f"Auto-selected {chart_type} visualization for the SQL result."


def _is_numeric_column(records: list[dict[str, Any]], column: str) -> bool:
    values = [record.get(column) for record in records if record.get(column) is not None]
    return bool(values) and all(_is_numeric_value(value) for value in values)


def _is_numeric_value(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, Number):
        return True
    if isinstance(value, str):
        return _coerce_float(value) is not None
    return False


def _looks_temporal_key(value: str) -> bool:
    lowered = value.lower()
    return any(token in lowered for token in ("date", "time", "month", "year", "day", "week"))


def _looks_temporal_value(value: str) -> bool:
    for parser in (
        lambda raw: datetime.fromisoformat(raw.replace("Z", "+00:00")),
        lambda raw: datetime.strptime(raw, "%Y-%m-%d"),
        lambda raw: datetime.strptime(raw, "%Y/%m/%d"),
        lambda raw: datetime.strptime(raw, "%Y-%m"),
        lambda raw: datetime.strptime(raw, "%b %Y"),
        lambda raw: datetime.strptime(raw, "%B %Y"),
        lambda raw: datetime.strptime(raw, "%Y"),
    ):
        try:
            parser(value)
            return True
        except Exception:
            continue
    return False


def _looks_like_identifier(value: str) -> bool:
    lowered = value.lower()
    return any(token in lowered for token in ("_id", " id", "number", "patient", "code", "key"))


def _stringify_label(value: Any, index: int) -> str:
    if value is None:
        return f"Item {index}"
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)[:120]


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Number):
        return float(value)
    try:
        return float(str(value).replace(",", ""))
    except Exception:
        return None


def _coerce_number_value(value: Any) -> int | float | str | None:
    numeric = _coerce_float(value)
    if numeric is None:
        return _safe_value(value)
    if numeric.is_integer():
        return int(numeric)
    return round(numeric, 2)


def _safe_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, str):
        return value[:500]
    return value


def _pretty_label(value: str) -> str:
    return value.replace("_", " ").strip().title()
