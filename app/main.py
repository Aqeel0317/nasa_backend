"""Entry point:  uvicorn app.main:app --reload"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from .config import settings
from .routes import router
from .routes_farm import router as farm_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(title=settings.app_name, version="1.0.0",
              description="Crop rotation decision support from NASA Earth observations")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)  # farm results compress ~5x
app.include_router(router)
app.include_router(farm_router)
