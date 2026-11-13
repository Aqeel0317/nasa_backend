"""Pydantic models: the contract between the React front end and the API."""
from typing import Literal

from pydantic import BaseModel, Field


class Weights(BaseModel):
    soil_health: float = Field(7, ge=0, le=10)
    water_saving: float = Field(5, ge=0, le=10)
    climate_resilience: float = Field(6, ge=0, le=10)
    profit: float = Field(6, ge=0, le=10)

    def as_engine(self) -> dict[str, float]:
        return {"Soil health": self.soil_health, "Water saving": self.water_saving,
                "Climate resilience": self.climate_resilience, "Profit": self.profit}


class SoilOverride(BaseModel):
    """A farmer's own soil test. Any field left empty keeps the SoilGrids value."""
    phh2o: float | None = Field(None, ge=3.5, le=10)
    soc: float | None = Field(None, ge=0.5, le=80, description="Organic carbon, g/kg")
    clay: float | None = Field(None, ge=0, le=90, description="%")
    sand: float | None = Field(None, ge=0, le=95, description="%")


class RotationRequest(BaseModel):
    lat: float = Field(..., ge=-60, le=75)
    lon: float = Field(..., ge=-180, le=180)
    irrigation_mm: float = Field(150, ge=0, le=1500, description="Water per season, mm")
    years: int = Field(3, ge=1, le=4)
    start_season: Literal["warm", "cool"] = "warm"
    weights: Weights = Weights()
    allowed_crops: list[str] | None = None
    allow_fallow: bool = False
    soil_override: SoilOverride | None = None
    top_k: int = Field(5, ge=1, le=10)


class CropOut(BaseModel):
    name: str
    family: str
    season: str
    role: str
    water_mm: float
    n_effect: float


class SuitabilityRow(BaseModel):
    crop: str
    season: str
    role: str
    now: float
    future: float


class RotationOut(BaseModel):
    rank: int
    sequence: list[str]
    seasons: list[str]
    scores: dict[str, float]
    total: float
    future_total: float
    robustness: float
    trajectory: dict
    explanations: list[str]


class RotationResponse(BaseModel):
    site: dict
    suitability: list[SuitabilityRow]
    rotations: list[RotationOut]
