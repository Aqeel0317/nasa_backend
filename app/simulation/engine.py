"""
engine.py - a simplified, AquaCrop-style daily crop model (Phase 1).

For each plot it runs, day by day:
  1. Crop stage from days after sowing (FAO-56 stage lengths).
  2. Root-zone water balance (FAO-56 ch. 8): rain + irrigation - crop water use.
     Water stress coefficient Ks drops once depletion passes the readily
     available water.
  3. Canopy cover grows and senesces along its potential curve; water stress
     and waterlogging slow growth, severe stress speeds senescence.
  4. Biomass = WP* x sum(Tr / ET0), the AquaCrop water-productivity concept.
  5. Yield = harvest index x biomass, with the harvest index reduced by water
     and heat stress around flowering.

The model treats each plot as uniform. It is a teaching and comparison tool;
Phase 2 swaps it for AquaCrop-OSPy behind the same interface.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from .crop_registry import CROPS, STAGE_NAMES, YIELD_SCALE, CropParams

# Soil water at field capacity / wilting point (m3/m3) by texture class
SOIL_WATER = {"light": (0.20, 0.08), "medium": (0.31, 0.15), "heavy": (0.40, 0.24)}
DRAIN_MM_DAY = {"light": 40.0, "medium": 20.0, "heavy": 8.0}
IRRIGATION = ("rainfed", "deficit", "full")


def texture_class(clay: float, sand: float) -> str:
    if sand > 65:
        return "light"
    if clay > 40:
        return "heavy"
    return "medium"


def _sow_date(crop: CropParams, year: int) -> dt.date:
    y = year if crop.sow_month >= 5 else year + 1
    return dt.date(y, crop.sow_month, crop.sow_day)


def _kc(crop: CropParams, das: int) -> float:
    i, d, m, l = crop.stages
    if das < i:
        return crop.kc_ini
    if das < i + d:
        return crop.kc_ini + (crop.kc_mid - crop.kc_ini) * (das - i) / d
    if das < i + d + m:
        return crop.kc_mid
    return crop.kc_mid + (crop.kc_end - crop.kc_mid) * (das - i - d - m) / l


def _cc_potential(crop: CropParams, das: int) -> float:
    i, d, m, l = crop.stages
    cc0, ccx = 0.03, crop.cc_max
    if das < i:
        return cc0 + (0.12 * ccx - cc0) * das / i
    if das < i + d:
        return 0.12 * ccx + 0.88 * ccx * (das - i) / d
    if das < i + d + m:
        return ccx
    return ccx * (1 - 0.65 * (das - i - d - m) / l)


def _stage(crop: CropParams, das: int) -> int:
    edge = np.cumsum(crop.stages)
    return int(np.searchsorted(edge, das, side="right"))


def simulate_plot(crop: CropParams, weather: dict, year: int, soil_class: str,
                  irrigation: str) -> dict:
    dates: list[dt.date] = weather["dates"]
    n = len(dates)
    sow = _sow_date(crop, year)
    start = (sow - dates[0]).days
    fc, wp = SOIL_WATER[soil_class]
    drain = DRAIN_MM_DAY[soil_class]
    i_len, d_len, m_len, _ = crop.stages
    flower_from, flower_to = i_len + d_len, i_len + d_len + min(crop.flowering_days, m_len)

    out = {k: np.full(n, np.nan) for k in
           ("cc", "lai", "biomass", "soil_water", "ks", "heat", "ndvi", "height", "yield_est")}
    stage = np.full(n, -1, int)     # -1 = not planted yet, 4 = harvested
    waterlog = np.zeros(n, bool)

    cc, biomass, excess, irrigated = 0.03, 0.0, 0.0, 0.0
    ks_flower, heat_flower = [], []
    stress_days = heat_days = waterlog_days = 0
    zr = 0.15
    taw = 1000 * (fc - wp) * zr
    dr = 0.3 * taw                       # start at 30 % depletion
    cc_prev_pot = _cc_potential(crop, 0)

    for t in range(n):
        das = t - start
        if das < 0:
            continue
        if das >= crop.season_days:
            stage[t] = 4
            continue
        st = _stage(crop, das)
        stage[t] = st
        tmax, tmin, rain, et0 = (weather[k][t] for k in ("tmax", "tmin", "rain", "et0"))

        # --- root growth and available water -------------------------------
        zr_new = 0.15 + (crop.root_max_m - 0.15) * min(1.0, das / (i_len + d_len))
        zr = zr_new
        taw = 1000 * (fc - wp) * zr
        kc = _kc(crop, das)
        etc = kc * et0
        p = float(np.clip(crop.p + 0.04 * (5 - etc), 0.1, 0.8))
        raw = p * taw

        # --- irrigation decision (before today's water use) ------------------
        irr = 0.0
        if irrigation != "rainfed":
            trigger = 0.2 * raw if crop.flooded else raw
            if dr > trigger:
                irr = dr if irrigation == "full" else 0.5 * dr
        irrigated += irr

        # --- water balance ------------------------------------------------------
        ks = 1.0 if dr <= raw else max(0.0, (taw - dr) / ((1 - p) * taw))
        dr = dr - rain - irr + ks * etc
        if dr < 0:                       # more water than the root zone holds
            excess += -dr
            dr = 0.0
        excess = max(0.0, excess - drain)
        dr = min(dr, taw)
        is_waterlogged = excess > 30 and not crop.flooded
        waterlog[t] = is_waterlogged
        wl = 0.55 if (is_waterlogged and not crop.waterlog_tolerant) else 1.0

        # --- canopy ------------------------------------------------------------
        cc_pot = _cc_potential(crop, das)
        growth = cc_pot - cc_prev_pot
        if growth > 0:
            cc += growth * (ks ** 1.5) * wl
        elif cc_prev_pot > 0:
            cc *= cc_pot / cc_prev_pot
        if ks < 0.4 and st >= 1:         # severe stress: early senescence
            cc -= 0.012 * cc * (1 - ks / 0.4)
        cc = float(np.clip(cc, 0.0, cc_pot))
        cc_prev_pot = cc_pot

        # --- biomass (water productivity) -------------------------------------
        tmean = (tmax + tmin) / 2
        cold = float(np.clip((tmean - crop.t_base) / 8.0, 0.0, 1.0))
        tr_ratio = ks * wl * 1.05 * cc
        biomass += crop.wp_star * tr_ratio * cold      # g/m2

        # --- stress around flowering ------------------------------------------
        h = float(np.clip((tmax - crop.t_heat) / (crop.t_heat_max - crop.t_heat), 0, 1))
        if flower_from <= das < flower_to:
            ks_flower.append(ks * wl)
            heat_flower.append(h)
        stress_days += ks < 0.7
        heat_days += h > 0 and st >= 2
        waterlog_days += bool(is_waterlogged)

        out["cc"][t] = cc
        out["lai"][t] = -np.log(max(1e-3, 1 - cc / 1.005)) / 0.65
        out["biomass"][t] = biomass / 100            # t/ha
        out["soil_water"][t] = 100 * (1 - dr / taw)  # % of available water left
        out["ks"][t] = ks * wl
        out["heat"][t] = h
        out["ndvi"][t] = 0.12 + 0.78 * cc
        out["height"][t] = min(1.0, das / (i_len + d_len + 10)) * (0.4 + 0.6 * cc / crop.cc_max)
        hi_now = _harvest_index(crop, ks_flower, heat_flower)
        out["yield_est"][t] = hi_now * biomass / 100 * crop.fresh_factor * YIELD_SCALE.get(crop.key, 1.0)

    final_yield = float(np.nanmax(out["yield_est"])) if np.isfinite(out["yield_est"]).any() else 0.0
    return {
        "crop": crop.key, "sow_date": sow.isoformat(),
        "harvest_date": (sow + dt.timedelta(days=crop.season_days)).isoformat(),
        "series": out, "stage": stage, "waterlog": waterlog,
        "summary": {
            "yield_t_ha": round(final_yield, 2),
            "biomass_t_ha": round(float(np.nanmax(out["biomass"])), 2),
            "irrigation_mm": round(irrigated),
            "water_stress_days": int(stress_days),
            "heat_days": int(heat_days),
            "waterlog_days": int(waterlog_days),
            "flowering_water_factor": round(float(np.mean(ks_flower)) if ks_flower else 1.0, 2),
            "flowering_heat": round(float(np.mean(heat_flower)) if heat_flower else 0.0, 2),
        },
    }


def _harvest_index(crop: CropParams, ks_flower: list, heat_flower: list) -> float:
    if not ks_flower:
        return crop.harvest_index
    water_loss = crop.flowering_sensitivity * (1 - float(np.mean(ks_flower)))
    heat_loss = crop.flowering_sensitivity * 1.2 * float(np.mean(heat_flower))
    return crop.harvest_index * max(0.05, 1 - water_loss - heat_loss)


def simulate_farm(weather: dict, year: int, soil_class: str, irrigation: str,
                  crops: list[str] | None = None) -> dict:
    keys = crops or list(CROPS)
    return {k: simulate_plot(CROPS[k], weather, year, soil_class, irrigation) for k in keys}


def insights(crop_key: str, base: dict, scen: dict) -> list[dict]:
    """Rule-based explanations as {code, params}; the front end words them."""
    b, s = base["summary"], scen["summary"]
    notes = []
    change = 100 * (s["yield_t_ha"] - b["yield_t_ha"]) / b["yield_t_ha"] if b["yield_t_ha"] else 0
    notes.append({"code": "yield_change", "params": {"pct": round(change)}})
    if s["flowering_water_factor"] < 0.8:
        notes.append({"code": "flower_water", "params": {"pct": round(100 * (1 - s["flowering_water_factor"]))}})
    if s["flowering_heat"] > 0.05:
        notes.append({"code": "flower_heat", "params": {"pct": round(100 * s["flowering_heat"])}})
    if s["waterlog_days"] > 2:
        notes.append({"code": "waterlog", "params": {"days": s["waterlog_days"]}})
    if s["water_stress_days"] > b["water_stress_days"] + 5:
        notes.append({"code": "stress_days", "params": {"days": s["water_stress_days"]}})
    if CROPS[crop_key].flooded and s["irrigation_mm"] == 0 and s["flowering_water_factor"] < 0.9:
        notes.append({"code": "rice_rainfed", "params": {}})
    if len(notes) == 1 and abs(change) < 3:
        notes.append({"code": "robust", "params": {}})
    return notes
