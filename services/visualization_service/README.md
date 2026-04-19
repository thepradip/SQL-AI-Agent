# SQL Output Visualization Service

Separate FastAPI service for converting SQL result sets into UI-safe chart specs.

## Run

```bash
python -m uvicorn services.visualization_service.app:app --host 127.0.0.1 --port 8011
```

## Endpoints

- `GET /health`
- `GET /meta`
- `POST /v1/visualizations/infer`
- `POST /v1/visualizations/validate`
- `POST /v1/visualizations/render-spec`

## Contract

The SQL agent sends the user question, SQL result columns/rows, and optional response commentary. The service returns a compact chart spec for `number`, `bar`, `line`, `pie`, or `table`.
