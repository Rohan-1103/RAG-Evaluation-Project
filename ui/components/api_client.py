"""
ui/components/api_client.py

The ONLY module under ui/ that imports httpx or constructs an HTTP
request. Every Streamlit page calls the FastAPI backend exclusively
through this client — never through src/ business logic directly.
This is the UI-side enforcement of the architecture rule stated when
src/api/app.py was first designed: "Streamlit is the UI layer only —
it calls the FastAPI backend, never imports from src/ directly."

get_api_client() is cached with st.cache_resource, which in Streamlit
is process-wide (shared across all browser sessions hitting this
server instance) — exactly what we want for a single httpx.Client
connection pool, mirroring why EmbeddingManager/ChromaVectorStore are
application-scoped on the FastAPI side rather than rebuilt per request.

Importing config.settings.get_settings() here is intentional and NOT
a violation of the "no src/ business logic in ui/" rule — config/ is
shared, app-wide configuration data (API_BASE_URL was specifically
added to Settings.api_base_url for exactly this consumer), distinct
from src/'s pipelines and evaluators.
"""

from __future__ import annotations

from typing import Any, BinaryIO

import httpx
import streamlit as st

from config.settings import get_settings


class APIError(Exception):
    """
    Raised when the backend returns a non-2xx response.

    Carries status_code/error_type/detail straight from FastAPI's
    structured exception handlers (src/api/app.py's
    _register_exception_handlers) so pages can render the exact same
    "collection_not_found" / "generation_error" / etc. category the
    backend already classified, rather than re-parsing a raw response.
    """

    def __init__(
        self, status_code: int, detail: str, error_type: str | None = None
    ) -> None:
        self.status_code = status_code
        self.detail = detail
        self.error_type = error_type
        super().__init__(f"[{status_code}] {error_type or 'error'}: {detail}")

class APIClient:
    """
    Synchronous httpx wrapper over every route registered in
    src/api/app.py's _register_routers(). One method per endpoint,
    named to match the endpoint's purpose, not its HTTP verb+path —
    pages call client.run_evaluation(...), never client.post("/api/v1/evaluate/run", ...).
    """
    def __init__(
        self,
        base_url: str,
        timeout: float = 600.0,
        api_key: str = "",
    ) -> None:
        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers=headers,
        )

# FIND _request() method and replace it entirely:
    def _request(
        self, method: str, path: str, **kwargs: Any
    ) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.ReadTimeout:
            raise APIError(
                status_code=408,
                detail=(
                    "The request timed out waiting for the backend to respond. "
                    "This usually means the comparison job has too many model "
                    "configurations for the free-tier API rate limits to complete "
                    "within the allowed time window.\n\n"
                    "Try one of these:\n"
                    "• Reduce the number of models (3 or fewer recommended)\n"
                    "• Use a single top_k value instead of multiple\n"
                    "• Run models one at a time via the Evaluate page\n"
                    "• Wait a few minutes for rate limit windows to reset"
                ),
                error_type="request_timeout",
            )
        except httpx.ConnectError:
            raise APIError(
                status_code=503,
                detail=(
                    "Cannot connect to the backend server. "
                    "Ensure the API is running: "
                    "uvicorn src.api.app:create_app --factory --port 8000"
                ),
                error_type="connection_error",
            )
        except httpx.TimeoutException as exc:
            raise APIError(
                status_code=408,
                detail=f"Request timed out: {exc}",
                error_type="request_timeout",
            )

        if response.status_code >= 400:
            try:
                body = response.json()
            except Exception:
                body = {}
            raise APIError(
                status_code=response.status_code,
                detail=body.get("detail", response.text),
                error_type=body.get("error"),
            )
        return response.json()

    def _request_bytes(
    self, method: str, path: str, **kwargs: Any) -> bytes:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.ReadTimeout:
            raise APIError(
                status_code=408,
                detail="Export request timed out. Try again in a moment.",
                error_type="request_timeout",
            )
        except httpx.ConnectError:
            raise APIError(
                status_code=503,
                detail="Cannot connect to the backend server.",
                error_type="connection_error",
            )
        if response.status_code >= 400:
            raise APIError(
                status_code=response.status_code,
                detail=response.text,
                error_type="export_error",
            )
        return response.content

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")
    
    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    def list_models(
        self,
        role: str | None = None,
        provider: str | None = None,
    ) -> dict[str, Any]:
        params = {
            k: v
            for k, v in {
                "role": role,
                "provider": provider,
            }.items()
            if v is not None
        }

        return self._request(
            "GET",
            "/api/v1/models",
            params=params,
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_files(
        self,
        collection_name: str,
        files: list[tuple[str, bytes]],
        upsert: bool = True,
    ) -> dict[str, Any]:
        """
        files: list of (filename, raw_bytes) tuples — built by pages
        from st.file_uploader's returned UploadedFile objects via
        (f.name, f.getvalue()).
        """
        multipart_files = [
            ("files", (name, content, "application/octet-stream"))
            for name, content in files
        ]
        data = {"collection_name": collection_name, "upsert": str(upsert).lower()}
        return self._request(
            "POST", "/api/v1/ingest/files", files=multipart_files, data=data
        )

    def list_collections(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/ingest/collections")

    def get_collection(self, name: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/ingest/collections/{name}")

    def delete_collection(self, name: str) -> dict[str, Any]:
        return self._request("DELETE", f"/api/v1/ingest/collections/{name}")

    def supported_formats(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/ingest/supported-formats")

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    def generate_dataset(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/v1/datasets/generate", json=payload)

    def list_datasets(self, **params: Any) -> dict[str, Any]:
        return self._request(
            "GET", "/api/v1/datasets", params={k: v for k, v in params.items() if v is not None}
        )

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/datasets/{dataset_id}")

    def get_dataset_metadata(self, dataset_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/datasets/{dataset_id}/metadata")

    def delete_dataset(self, dataset_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/api/v1/datasets/{dataset_id}")

    def edit_pair(
        self, dataset_id: str, pair_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return self._request(
            "PATCH", f"/api/v1/datasets/{dataset_id}/pairs/{pair_id}", json=payload
        )

    def delete_pair(self, dataset_id: str, pair_id: str) -> dict[str, Any]:
        return self._request(
            "DELETE", f"/api/v1/datasets/{dataset_id}/pairs/{pair_id}"
        )

    def export_dataset_csv(self, dataset_id: str) -> bytes:
        return self._request_bytes(
            "GET", f"/api/v1/datasets/{dataset_id}/export.csv"
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def run_evaluation(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/v1/evaluate/run", json=payload)

    def list_runs(self, **params: Any) -> dict[str, Any]:
        return self._request(
            "GET", "/api/v1/evaluate", params={k: v for k, v in params.items() if v is not None}
        )

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/evaluate/{run_id}")

    def delete_run(self, run_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/api/v1/evaluate/{run_id}")

    def export_run_csv(self, run_id: str) -> bytes:
        return self._request_bytes("GET", f"/api/v1/evaluate/{run_id}/export.csv")

    def export_runs_summary_csv(self, dataset_id: str | None = None) -> bytes:
        params = {"dataset_id": dataset_id} if dataset_id else {}
        return self._request_bytes(
            "GET", "/api/v1/evaluate/summary.csv", params=params
        )

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def run_comparison(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/v1/compare/run", json=payload)

    def list_comparisons(self, **params: Any) -> dict[str, Any]:
        return self._request(
            "GET", "/api/v1/compare", params={k: v for k, v in params.items() if v is not None}
        )

    def get_comparison(self, matrix_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/compare/{matrix_id}")

    def delete_comparison(
        self, matrix_id: str, delete_runs: bool = False
    ) -> dict[str, Any]:
        return self._request(
            "DELETE",
            f"/api/v1/compare/{matrix_id}",
            params={"delete_runs": str(delete_runs).lower()},
        )

    def export_comparison_csv(self, matrix_id: str) -> bytes:
        return self._request_bytes(
            "GET", f"/api/v1/compare/{matrix_id}/export.csv"
        )

@st.cache_resource(show_spinner=False)
def get_api_client() -> APIClient:
    """
    Get API client, reading credentials from:
    1. Streamlit secrets (production on Streamlit Cloud)
    2. Settings / .env file (local development)
    """
    try:
        # Streamlit Cloud: secrets available via st.secrets
        api_key = st.secrets.get("API_SECRET_KEY", "")
        base_url = st.secrets.get(
            "API_BASE_URL", "http://localhost:8000"
        )
    except Exception:
        # Local development: fall back to Settings
        settings = get_settings()
        api_key = settings.api.secret_key
        base_url = settings.api_base_url

    return APIClient(
        base_url=base_url,
        timeout=600.0,
        api_key=api_key,
    )

__all__ = ["APIClient", "APIError", "get_api_client"]