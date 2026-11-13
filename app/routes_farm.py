"""Virtual farm endpoints: simulate a farmer's own fields (up to four) under a scenario."""
import datetime as dt
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .services.data_sources import fetch_soilgrids
from .simulation.crop_registry import CROPS
from .simulation.engine import IRRIGATION, insights, simulate_farm, texture_class
from .simulation.scenarios import apply_scenario
from .simulation.weather import get_farm_weather

router = APIRouter(prefix="/api/farm")
CropKey = Literal[tuple(CROPS)]  # type: ignore[valid-type]
SHOCKS = ("drought", "heatwave", "flood")


class FieldPlan(BaseModel):
    """One field: a summer (kharif) crop, then a winter (rabi) crop. Either may be empty."""
    name: str = Field("", max_length=40)
    kharif: CropKey | None = None
    rabi: CropKey | None = None
    acres: float = Field(2.0, gt=0, le=500)


DEFAULT_FIELDS = [
    FieldPlan(kharif="cotton", rabi="wheat"),
    FieldPlan(kharif="rice", rabi="wheat"),
    FieldPlan(kharif="maize", rabi="mustard"),
    FieldPlan(kharif="mung_bean", rabi="chickpea"),
]
MODEL_NAME = "Simplified AquaCrop-style model (Phase 1)"


class FarmRequest(BaseModel):
    lat: float = Field(31.42, ge=-60, le=75)
    lon: float = Field(73.08, ge=-180, le=180)
    year: int = Field(2024, ge=2001, description="Farm year starts on 1 May of this year")
    scenario: Literal["normal", "drought", "heatwave", "flood", "custom"] = "normal"
    temp_offset: float = Field(0.0, ge=-3, le=8)
    rain_pct: float = Field(100.0, ge=0, le=250)
    irrigation: Literal["rainfed", "deficit", "full"] = "full"
    clay: float | None = Field(None, ge=0, le=90)
    sand: float | None = Field(None, ge=0, le=95)
    fields: list[FieldPlan] = Field(default_factory=lambda: [f.model_copy() for f in DEFAULT_FIELDS],
                                    min_length=1, max_length=4)


def _arr(a, nd=3):
    return [None if not np.isfinite(v) else round(float(v), nd) for v in a]


@router.get("/crops")
def farm_crops() -> list[dict]:
    return [{"key": c.key, "name": c.name, "season_days": c.season_days,
             "sow": f"{c.sow_month:02d}-{c.sow_day:02d}", "height_m": c.height_m,
             "flooded": c.flooded, "yield_label": c.yield_label,
             "season_group": c.season_group} for c in CROPS.values()]


@router.post("/simulate")
def simulate(req: FarmRequest) -> dict:
    last_full = dt.date.today() - dt.timedelta(days=10)
    if dt.date(req.year + 1, 4, 30) > last_full:
        raise HTTPException(422, f"Farm year {req.year} is not complete yet; choose {last_full.year - 2} or earlier.")

    if req.clay is not None and req.sand is not None:
        soil, soil_source = {"clay": req.clay, "sand": req.sand}, "Farmer soil test"
    else:
        soil, soil_source = fetch_soilgrids(req.lat, req.lon)
    soil_class = texture_class(soil.get("clay", 25), soil.get("sand", 40))

    for f in req.fields:
        if f.kharif and CROPS[f.kharif].season_group != "kharif":
            raise HTTPException(422, f"{f.kharif} is not a kharif crop")
        if f.rabi and CROPS[f.rabi].season_group != "rabi":
            raise HTTPException(422, f"{f.rabi} is not a rabi crop")
    # Each crop is simulated once: the same crop on the same soil and weather grows the same.
    keys = list(dict.fromkeys(k for f in req.fields for k in (f.kharif, f.rabi) if k))
    if not keys:
        raise HTTPException(422, "Choose at least one crop.")

    weather = get_farm_weather(req.lat, req.lon, req.year)
    base_w = apply_scenario(weather, "normal")
    scen_w = apply_scenario(weather, req.scenario, req.temp_offset, req.rain_pct)
    base = simulate_farm(base_w, req.year, soil_class, req.irrigation, keys)
    scen = simulate_farm(scen_w, req.year, soil_class, req.irrigation, keys)
    # Best case for this field: normal weather and full irrigation. Gives farmers a
    # yardstick ("you get 60% of what this crop can produce here").
    best = simulate_farm(base_w, req.year, soil_class, "full", keys)
    # The same farm under every disaster, for the "what could hit you" comparison.
    shocks = {}
    for name in SHOCKS:
        res = scen if name == req.scenario else simulate_farm(
            apply_scenario(weather, name), req.year, soil_class, req.irrigation, keys)
        shocks[name] = {k: res[k]["summary"]["yield_t_ha"] for k in keys}

    plots = []
    for key in keys:
        crop = CROPS[key]
        s, b = scen[key], base[key]
        plots.append({
            "crop": key, "sow_date": s["sow_date"], "harvest_date": s["harvest_date"],
            "height_m": crop.height_m, "flooded": crop.flooded, "yield_label": crop.yield_label,
            "season_group": crop.season_group,
            "scenario": {
                "summary": s["summary"],
                "series": {k: _arr(v, 2 if k == "biomass" or k == "yield_est" else 3)
                           for k, v in s["series"].items()},
                "stage": s["stage"].tolist(),
                "waterlog": s["waterlog"].astype(int).tolist(),
            },
            "baseline": {
                "summary": b["summary"],
                "series": {k: _arr(b["series"][k]) for k in ("cc", "soil_water", "biomass")},
            },
            "insights": insights(key, b, s),
            "potential_yield_t_ha": best[key]["summary"]["yield_t_ha"],
        })

    return {
        "dates": [d.isoformat() for d in weather["dates"]],
        "weather": {
            "source": weather["source"], "observed": weather["observed"],
            "scenario": {"tmax": _arr(scen_w["tmax"], 1), "rain": _arr(scen_w["rain"], 1)},
            "baseline": {"tmax": _arr(base_w["tmax"], 1), "rain": _arr(base_w["rain"], 1)},
        },
        "soil": {"class": soil_class, "source": soil_source},
        "scenario": req.scenario, "irrigation": req.irrigation,
        "model": MODEL_NAME,
        "fields": [f.model_dump() for f in req.fields],
        "plots": plots,
        "shocks": shocks,
    }
