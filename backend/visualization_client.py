"""HTTP client for the standalone SQL output visualization service."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from config import get_settings

logger = logging.getLogger(__name__)


async def infer_visualization(user_query: str, query_result: dict[str, Any], commentary: str | None = None) -> dict[str, Any] | None:
    """Ask the visualization service for a chart spec without blocking SQL answers."""
    settings = get_settings()
    if not settings.visualization_service_enabled:
        return None

    payload = {
        "question": user_query,
        "result": {
            "columns": query_result.get("columns", []),
            "rows": query_result.get("rows", []),
            "row_count": query_result.get("row_count"),
            "truncated": query_result.get("truncated", False),
            "execution_time_ms": query_result.get("execution_time_ms"),
        },
        "commentary": commentary,
    }

    try:
        async with httpx.AsyncClient(timeout=settings.visualization_service_timeout_seconds) as client:
            response = await client.post(
                f"{settings.visualization_service_url.rstrip('/')}/v1/visualizations/infer",
                json=payload,
            )
            response.raise_for_status()
            visualization = response.json()
            return visualization if isinstance(visualization, dict) else None
    except Exception as exc:
        logger.warning("Visualization service unavailable: %s", exc)
        return None
