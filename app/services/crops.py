"""
crops.py - crop trait library.

Values are ILLUSTRATIVE defaults for a prototype. Replace/calibrate them with
FAO EcoCrop, FAO CROPWAT/AquaCrop parameters, or local extension data.

season:      "warm" (summer/kharif) or "cool" (winter/rabi)
temps (C):   t_min < t_opt_low <= t_opt_high < t_max  (season mean temperature)
water_mm:    seasonal crop water need
drought_tol, heat_tol, waterlog_tol: 0 (sensitive) .. 1 (tolerant)
n_effect:    kg N/ha left (+) or removed (-) for the next crop
residue_c:   0..1 index of carbon returned to soil (residue + roots)
value:       0..1 relative profitability index (replace with local prices)
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Crop:
    name: str
    family: str
    season: str
    role: str            # cash | legume | cover | fallow
    t_min: float
    t_opt_low: float
    t_opt_high: float
    t_max: float
    water_mm: float
    drought_tol: float
    heat_tol: float
    waterlog_tol: float
    n_effect: float
    residue_c: float
    root_depth: str      # shallow | medium | deep
    ph_min: float
    ph_max: float
    texture: str         # light | medium | heavy | any
    value: float


C = Crop
CROPS: dict[str, Crop] = {c.name: c for c in [
    # ---- warm season ----
    C("Maize", "Poaceae", "warm", "cash", 15, 22, 30, 38, 550, .4, .5, .3, -120, .8, "deep", 5.5, 7.5, "medium", .70),
    C("Rice", "Poaceae", "warm", "cash", 18, 24, 32, 38, 1200, .1, .6, .95, -100, .5, "shallow", 5.0, 7.5, "heavy", .75),
    C("Cotton", "Malvaceae", "warm", "cash", 18, 24, 33, 40, 750, .6, .8, .3, -100, .3, "deep", 5.8, 8.2, "medium", .80),
    C("Sorghum", "Poaceae", "warm", "cash", 15, 24, 33, 40, 450, .8, .8, .4, -80, .7, "deep", 5.5, 8.5, "any", .50),
    C("Pearl millet", "Poaceae", "warm", "cash", 18, 26, 34, 42, 350, .9, .9, .2, -50, .5, "medium", 5.5, 8.5, "light", .40),
    C("Mung bean", "Fabaceae", "warm", "legume", 18, 25, 33, 40, 300, .7, .7, .2, 40, .4, "medium", 6.2, 7.8, "medium", .55),
    C("Soybean", "Fabaceae", "warm", "legume", 15, 22, 30, 36, 500, .5, .5, .3, 60, .5, "medium", 6.0, 7.5, "medium", .65),
    C("Guar", "Fabaceae", "warm", "legume", 20, 26, 35, 42, 300, .85, .85, .1, 40, .3, "deep", 7.0, 8.5, "light", .50),
    C("Sunflower", "Asteraceae", "warm", "cash", 15, 20, 28, 36, 450, .7, .6, .2, -60, .5, "deep", 6.0, 7.8, "medium", .55),
    C("Sesbania (green manure)", "Fabaceae", "warm", "cover", 18, 24, 35, 42, 300, .6, .8, .8, 100, .9, "deep", 5.0, 9.0, "any", .10),
    # ---- cool season ----
    C("Wheat", "Poaceae", "cool", "cash", 3, 12, 22, 30, 450, .5, .4, .2, -100, .7, "medium", 6.0, 8.0, "medium", .65),
    C("Barley", "Poaceae", "cool", "cash", 2, 10, 20, 30, 350, .7, .5, .2, -70, .6, "medium", 6.0, 8.5, "any", .50),
    C("Chickpea", "Fabaceae", "cool", "legume", 5, 14, 25, 32, 300, .8, .5, .1, 40, .3, "deep", 6.0, 8.0, "medium", .60),
    C("Lentil", "Fabaceae", "cool", "legume", 5, 14, 24, 30, 280, .7, .4, .1, 50, .3, "medium", 6.0, 8.0, "medium", .60),
    C("Mustard / canola", "Brassicaceae", "cool", "cash", 3, 10, 22, 30, 350, .6, .4, .2, -60, .4, "deep", 5.5, 8.0, "any", .55),
    C("Potato", "Solanaceae", "cool", "cash", 5, 14, 22, 28, 500, .3, .3, .2, -120, .2, "shallow", 5.0, 6.8, "light", .85),
    C("Berseem clover", "Fabaceae", "cool", "cover", 5, 14, 25, 30, 500, .3, .4, .5, 100, .7, "medium", 6.5, 8.0, "heavy", .40),
    C("Oats (cover)", "Poaceae", "cool", "cover", 0, 10, 20, 28, 300, .5, .3, .3, -30, .8, "medium", 5.5, 7.8, "any", .30),
]}

FALLOW = Crop("Fallow (bare)", "none", "any", "fallow", -50, -50, 60, 60, 0, 1, 1, 1, 0, 0,
              "shallow", 0, 14, "any", 0)


def crops_for_season(season: str, allowed: list[str] | None = None) -> list[Crop]:
    pool = [c for c in CROPS.values() if c.season == season]
    if allowed:
        pool = [c for c in pool if c.name in allowed]
    return pool
