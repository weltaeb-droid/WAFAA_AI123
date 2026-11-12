"""WAFAA AI — FastAPI application.

Serves the JSON API under ``/api``, the generated sound library under
``/media`` and the compiled React single-page app at ``/``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config, db
from .routers import analysis, babies, content, library

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("wafaa")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    config.ensure_directories()
    if config.MODEL_PATH.exists():
        logger.info("WAFAA AI model file found at %s", config.MODEL_PATH)
    else:
        logger.info(
            "No WAFAA AI model file at %s — analysis stays in 'model not connected' mode.",
            config.MODEL_PATH,
        )
    yield


app = FastAPI(
    title="WAFAA AI",
    version=config.APP_VERSION,
    description=(
        "Experimental infant cry-analysis support tool. Not a medical device and "
        "not a substitute for professional healthcare advice."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Device-Id"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": {
                "code": "internal_error",
                "message": "Something went wrong while processing the request.",
            }
        },
    )


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "app": config.APP_NAME,
        "version": config.APP_VERSION,
        "model_connected": config.MODEL_PATH.exists(),
        "sample_rate": config.TARGET_SAMPLE_RATE,
    }


app.include_router(babies.router)
app.include_router(analysis.router)
app.include_router(library.router)
app.include_router(content.router)

# Generated, original audio assets.
if config.STATIC_AUDIO_DIR.exists():
    app.mount(
        "/media/audio",
        StaticFiles(directory=str(config.STATIC_AUDIO_DIR)),
        name="audio",
    )

# --- Frontend ----------------------------------------------------------------
_FRONTEND_READY = config.FRONTEND_DIST.is_dir() and (
    config.FRONTEND_DIST / "index.html"
).is_file()

if _FRONTEND_READY:
    assets_dir = config.FRONTEND_DIST / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith(("api/", "media/")):
            return JSONResponse(
                status_code=404,
                content={"detail": {"code": "not_found", "message": "Unknown endpoint."}},
            )
        candidate = (config.FRONTEND_DIST / full_path).resolve()
        try:
            inside = config.FRONTEND_DIST.resolve() in candidate.parents
        except OSError:  # pragma: no cover
            inside = False
        if full_path and inside and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(config.FRONTEND_DIST / "index.html")

else:  # pragma: no cover - development mode without a build

    @app.get("/", include_in_schema=False)
    def root_placeholder() -> dict:
        return {
            "app": config.APP_NAME,
            "message": (
                "API is running. Build the frontend with `npm --prefix frontend run "
                "build` or run the Vite dev server."
            ),
            "docs": "/docs",
        }
