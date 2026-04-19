"""Pydantic contracts for the visualization service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ChartType = Literal["number", "bar", "line", "pie", "table"]


class QueryResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    columns: list[str] = Field(default_factory=list, max_length=50)
    rows: list[list[Any] | dict[str, Any]] = Field(default_factory=list, max_length=5000)
    row_count: int | None = Field(default=None, ge=0)
    truncated: bool = False
    execution_time_ms: float | None = Field(default=None, ge=0)

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, columns: list[str]) -> list[str]:
        cleaned = [str(column).strip() for column in columns]
        if any(not column for column in cleaned):
            raise ValueError("columns cannot contain empty names")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("columns must be unique")
        return cleaned


class VisualizationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str = Field(default="", max_length=1000)
    result: QueryResult
    commentary: str | None = Field(default=None, max_length=2000)
    preferred_type: ChartType | None = None


class VisualizationSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: ChartType
    title: str | None = Field(default=None, max_length=160)
    description: str | None = Field(default=None, max_length=500)
    x_key: str | None = None
    y_key: str | None = None
    label_key: str | None = None
    value_key: str | None = None
    labels: list[str] | None = Field(default=None, max_length=50)
    values: list[float] | None = Field(default=None, max_length=50)
    records: list[dict[str, Any]] | None = Field(default=None, max_length=5000)
    number_value: int | float | str | None = None
    number_label: str | None = None
    number_context: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = "visualization-service"
    warnings: list[str] = Field(default_factory=list, max_length=20)


class ValidationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str = Field(default="", max_length=1000)
    result: QueryResult
    visualization: VisualizationSpec | None = None
    commentary: str | None = Field(default=None, max_length=2000)


class ValidationResponse(BaseModel):
    valid: bool
    score: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ServiceMetadata(BaseModel):
    name: str
    version: str
    endpoints: list[str]
