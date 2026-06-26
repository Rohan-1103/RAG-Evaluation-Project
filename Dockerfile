# syntax=docker/dockerfile:1

# =============================================================================
# Stage 1: Builder — install Poetry-managed dependencies into a venv
#
# Pinned to python:3.11-slim deliberately, not 3.13. Every dependency
# conflict diagnosed during local setup (numpy 1.26.4's OverflowError on
# Python 3.13, the tokenizers/transformers version lock from
# sentence-transformers, langchain-chroma's numpy<2.0 ceiling) traces back
# to running on a Python version newer than several ML libraries had
# stable pre-built wheels for at the time. 3.11 has mature, conflict-free
# wheels for every dependency in pyproject.toml — this Dockerfile avoids
# re-encountering the exact issues already solved the hard way locally.
# =============================================================================
FROM python:3.11-slim AS builder

ENV POETRY_VERSION=1.8.3 \
    POETRY_HOME=/opt/poetry \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PIP_NO_CACHE_DIR=1

# build-essential is the Linux equivalent of the Visual C++ Build Tools
# that were the actual root cause of the hnswlib compilation failures
# during local Windows setup. Installing the compiler once at image-build
# time means every future `docker build` is unaffected by the host
# machine's toolchain — this exact class of failure cannot recur inside
# a container built from this image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 - --version ${POETRY_VERSION}
ENV PATH="${POETRY_HOME}/bin:${PATH}"

WORKDIR /build

# Copying only dependency manifests before any application code means
# this layer — the slowest one, since it compiles hnswlib and downloads
# torch — is cached across every rebuild that touches only src/ or ui/,
# not pyproject.toml. This is the single highest-impact layer-ordering
# decision in this file.
COPY pyproject.toml poetry.lock* ./

# --no-root: only install dependencies, not the project itself — the
# project's own source isn't copied into this stage yet, so there is
# nothing for Poetry to install as the "root" package at this point.
# --only main: excludes pytest/ruff/mypy/jupyterlab/ipython — the dev
# dependency group has no place in a runtime image.
RUN poetry install --no-root --only main --no-ansi

# =============================================================================
# Stage 2: Runtime — copy the built venv, add application code only
#
# Multi-stage build means build-essential, Poetry, and pip's download
# cache never exist in the final image — only the resolved venv and the
# application source. This is what keeps the shipped image meaningfully
# smaller than installing everything in one single-stage Dockerfile.
# =============================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/build/.venv \
    PATH="/build/.venv/bin:${PATH}"

# libgomp1: required at import time by torch/sentence-transformers' BLAS
# routines on Debian slim images. Without it, EmbeddingManager's
# HuggingFace model load fails inside the container with an obscure
# shared-library error even though the exact same code runs fine on a
# full local dev machine that already has it installed as a transitive
# system dependency of something else.
# curl: required by this image's own HEALTHCHECK instruction below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY --from=builder /build/.venv /build/.venv

# Copied explicitly by directory, not via a single `COPY . .`, so that
# .dockerignore's exclusions (data/, logs/, .venv/, tests/, .git/) are
# the only thing standing between "what's on disk locally" and "what
# ships in the image" — being explicit here is a second, redundant
# layer of the same guarantee .dockerignore provides.
COPY config/ ./config/
COPY src/ ./src/
COPY ui/ ./ui/
COPY scripts/ ./scripts/
COPY pyproject.toml ./

# Created here (not left to Settings.create_required_directories() at
# runtime) so the directories exist with correct ownership BEFORE the
# container ever drops to the unprivileged appuser below — that
# validator still runs at app startup and is harmless as a no-op when
# the directories already exist, but ownership must be set as root
# first or appuser would lack write permission to its own data/logs.
RUN mkdir -p /app/data /app/logs /home/appuser/.cache/huggingface && chown -R appuser:appuser /app /home/appuser/.cache

USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || curl -f http://localhost:8501/_stcore/health || exit 1

# No default CMD: this single image runs as two entirely different
# processes (uvicorn for the api service, streamlit for the ui service)
# depending on which docker-compose service starts it — the command is
# supplied explicitly per-service in docker-compose.yml below, never here.