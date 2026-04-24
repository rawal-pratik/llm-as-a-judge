"""
FastAPI application entry point.

Creates and configures the FastAPI app instance.
Registers routers and startup/shutdown events.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.routes import router
from logging_config import setup_logging
from config import LOG_LEVEL
from database.db import init_db

setup_logging(LOG_LEVEL)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, clean up on shutdown."""
    logger.info("Initializing database tables...")
    init_db()
    logger.info("Database ready.")
    yield


app = FastAPI(
    title="LLM-as-a-Judge",
    description="Evaluate LLM outputs using LLM judges with bias analysis and inter-rater metrics.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health")
async def health_check():
    """Basic liveness probe."""
    return {"status": "ok"}
