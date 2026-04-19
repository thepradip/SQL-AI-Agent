"""FastAPI service for SQL output visualization inference and validation."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .inference import infer_visualization
from .schemas import ServiceMetadata, ValidationRequest, ValidationResponse, VisualizationRequest, VisualizationSpec
from .validation import validate_visualization_spec

SERVICE_VERSION = "1.0.0"

app = FastAPI(
    title="SQL Output Visualization Service",
    description="Infers safe chart specs from SQL query results.",
    version=SERVICE_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "visualization", "version": SERVICE_VERSION}


@app.get("/meta", response_model=ServiceMetadata)
async def meta() -> ServiceMetadata:
    return ServiceMetadata(
        name="sql-output-visualization",
        version=SERVICE_VERSION,
        endpoints=[
            "GET /health",
            "GET /meta",
            "POST /v1/visualizations/infer",
            "POST /v1/visualizations/validate",
            "POST /v1/visualizations/render-spec",
        ],
    )


@app.post("/v1/visualizations/infer", response_model=VisualizationSpec | None)
async def infer(request: VisualizationRequest) -> VisualizationSpec | None:
    return infer_visualization(
        question=request.question,
        result=request.result,
        commentary=request.commentary,
        preferred_type=request.preferred_type,
    )


@app.post("/v1/visualizations/validate", response_model=ValidationResponse)
async def validate(request: ValidationRequest) -> ValidationResponse:
    return validate_visualization_spec(request.visualization, request.result)


@app.post("/v1/visualizations/render-spec", response_model=VisualizationSpec | None)
async def render_spec(request: VisualizationRequest) -> VisualizationSpec | None:
    return infer_visualization(
        question=request.question,
        result=request.result,
        commentary=request.commentary,
        preferred_type=request.preferred_type,
    )
