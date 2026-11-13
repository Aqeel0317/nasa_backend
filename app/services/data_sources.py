"""
data_sources.py - every external API call lives here.

APIs used (all free; only earthaccess needs a free NASA Earthdata Login):
  1. NASA POWER (Agroclimatology)  - climate normals + 10-yr daily history
  2. ISRIC SoilGrids v2.0          - soil texture, organic carbon, pH, N, CEC
  3. NASA MODIS via ORNL DAAC      - NDVI history of the field (MOD13Q1, 250 m)
  4. Open-Meteo Climate API        - CMIP6 HighResMIP projections to 2050
  5. NASA Earthdata via earthaccess (optional) - SMAP soil moisture, GPM IMERG
     rainfall, GRACE groundwater granules for advanced analysis

Every fetcher returns (data, source_label). If a call fails, it returns
reasonable fallback values labelled "fallback" so the app never crashes
in the field with bad connectivity.
"""
from __future__ import annotations

import calendar
import datetime as dt
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import requests
from cachetools import TTLCache

log = logging.getLogger("rotaterra.data")
TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "60"))
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
FILL = -999.0


# Responses are kept on disk so they survive restarts. On Azure App Service /home
# is persistent storage shared by all workers; locally a .cache folder is used.
CACHE_DIR = Path(os.getenv("CACHE_DIR") or ("/home/cache" if os.getenv("WEBSITE_SITE_NAME") else ".cache"))
DISK_TTL = int(os.getenv("DISK_CACHE_SECONDS", str(30 * 86400)))


def _get(url: str, params: Any = None, headers: dict | None = None, ttl: int | None = None,
         timeout: float | None = None) -> Any:
    """GET JSON, served from the disk cache when a fresh copy exists.
    Past weather and soil do not change, so the default lifetime is long."""
    ttl = DISK_TTL if ttl is None else ttl
    key = hashlib.sha1(json.dumps([url, params], sort_keys=True, default=str).encode()).hexdigest()
    path = CACHE_DIR / f"{key}.json"
    if ttl > 0:
        try:
            if time.time() - path.stat().st_mtime < ttl:
                return json.loads(path.read_text("utf-8"))
        except (OSError, ValueError):
            pass
    r = requests.get(url, params=params, headers=headers, timeout=timeout or TIMEOUT)
    r.raise_for_status()
    js = r.json()
    if ttl > 0:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(js), "utf-8")
            tmp.replace(path)
        except OSError as e:
            log.warning("could not write cache: %s", e)
    return js


# --------------------------------------------------------------------------
# 1. NASA POWER - monthly climatology (crop-season climate)
# --------------------------------------------------------------------------
POWER_BASE = "https://power.larc.nasa.gov/api/temporal"


def fetch_power_climatology(lat: float, lon: float) -> tuple[dict, str]:
    """Monthly normals: temperature, rainfall, humidity, solar, root-zone wetness."""
    params = {
        "parameters": "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN,GWETROOT",
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "format": "JSON",
    }
    try:
        js = _get(f"{POWER_BASE}/climatology/point", params)
        p = js["properties"]["parameter"]
        days = [calendar.monthrange(2023, m)[1] for m in range(1, 13)]
        out = {
            "t_mean": [p["T2M"][m] for m in MONTHS],
            "t_max": [p["T2M_MAX"][m] for m in MONTHS],
            "t_min": [p["T2M_MIN"][m] for m in MONTHS],
            # PRECTOTCORR is mm/day -> convert to monthly totals
            "precip_mm": [p["PRECTOTCORR"][m] * d for m, d in zip(MONTHS, days)],
            "rh": [p["RH2M"][m] for m in MONTHS],
            "solar": [p["ALLSKY_SFC_SW_DWN"][m] for m in MONTHS],
            "root_wetness": [p["GWETROOT"][m] for m in MONTHS],  # 0 (dry) .. 1 (saturated)
        }
        return out, "NASA POWER climatology"
    except Exception as e:  # noqa: BLE001
        log.warning("POWER climatology fallback: %s", e)
        return _fallback_climate(lat), "fallback (demo values)"


def _fallback_climate(lat: float) -> dict:
    north = lat >= 0
    base = np.array([5, 7, 11, 16, 21, 25, 27, 26, 22, 16, 10, 6], float)
    if not north:
        base = np.roll(base, 6)
    rain = np.array([40, 40, 50, 60, 80, 90, 100, 90, 70, 60, 50, 45], float)
    return {
        "t_mean": base.tolist(),
        "t_max": (base + 6).tolist(),
        "t_min": (base - 6).tolist(),
        "precip_mm": rain.tolist(),
        "rh": [60] * 12,
        "solar": [15] * 12,
        "root_wetness": [0.5] * 12,
    }


# --------------------------------------------------------------------------
# 1b. NASA POWER - 10-year daily history (extremes + observed trends)
# --------------------------------------------------------------------------
def fetch_power_extremes(lat: float, lon: float, years: int = 10) -> tuple[dict, str]:
    end = dt.date.today().replace(month=1, day=1) - dt.timedelta(days=1)
    start = end.replace(year=end.year - years + 1, month=1, day=1)
    params = {
        "parameters": "T2M_MAX,PRECTOTCORR",
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    }
    try:
        js = _get(f"{POWER_BASE}/daily/point", params)
        p = js["properties"]["parameter"]
        dates = sorted(p["T2M_MAX"].keys())
        tmax = np.array([p["T2M_MAX"][d] for d in dates], float)
        rain = np.array([p["PRECTOTCORR"][d] for d in dates], float)
        yrs = np.array([int(d[:4]) for d in dates])
        tmax[tmax == FILL] = np.nan
        rain[rain == FILL] = np.nan

        heat_days, dry_spells, annual_rain, annual_tmax = [], [], [], []
        for y in np.unique(yrs):
            m = yrs == y
            heat_days.append(int(np.nansum(tmax[m] > 35)))
            dry_spells.append(_longest_run(rain[m] < 1.0))
            annual_rain.append(float(np.nansum(rain[m])))
            annual_tmax.append(float(np.nanmean(tmax[m])))
        x = np.unique(yrs)
        out = {
            "heat_days_per_year": float(np.mean(heat_days)),
            "max_dry_spell_days": float(np.mean(dry_spells)),
            "rain_trend_mm_per_decade": float(np.polyfit(x, annual_rain, 1)[0] * 10),
            "tmax_trend_c_per_decade": float(np.polyfit(x, annual_tmax, 1)[0] * 10),
            "rain_cv": float(np.std(annual_rain) / max(np.mean(annual_rain), 1)),
            "years": [int(v) for v in x],
            "annual_rain": annual_rain,
            "heat_days": heat_days,
        }
        return out, "NASA POWER daily"
    except Exception as e:  # noqa: BLE001
        log.warning("POWER daily fallback: %s", e)
        return {
            "heat_days_per_year": 20.0, "max_dry_spell_days": 25.0,
            "rain_trend_mm_per_decade": 0.0, "tmax_trend_c_per_decade": 0.3,
            "rain_cv": 0.25, "years": [], "annual_rain": [], "heat_days": [],
        }, "fallback (demo values)"


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


# --------------------------------------------------------------------------
# 2. ISRIC SoilGrids v2.0 - topsoil properties (0-30 cm)
# --------------------------------------------------------------------------
SOILGRIDS = "https://rest.isric.org/soilgrids/v2.0/properties/query"
SOIL_PROPS = ["clay", "sand", "silt", "soc", "phh2o", "nitrogen", "cec", "bdod"]
DEPTHS = {"0-5cm": 5, "5-15cm": 10, "15-30cm": 15}


def fetch_soilgrids(lat: float, lon: float) -> tuple[dict, str]:
    """Returns depth-weighted 0-30 cm means in conventional units.
    clay/sand/silt %, soc g/kg, ph, nitrogen g/kg, cec cmol/kg, bdod g/cm3."""
    params = [("lon", lon), ("lat", lat), ("value", "mean")]
    params += [("property", p) for p in SOIL_PROPS]
    params += [("depth", d) for d in DEPTHS]
    try:
        # SoilGrids is often slow; give up quickly and use typical values instead.
        js = _get(SOILGRIDS, params=params, timeout=12)
        out = {}
        for layer in js["properties"]["layers"]:
            factor = layer["unit_measure"].get("d_factor", 1) or 1
            vals, wts = [], []
            for d in layer["depths"]:
                v = d["values"].get("mean")
                if v is not None and d["label"] in DEPTHS:
                    vals.append(v / factor)
                    wts.append(DEPTHS[d["label"]])
            if vals:
                out[layer["name"]] = float(np.average(vals, weights=wts))
        if len(out) < 4:
            raise ValueError("SoilGrids returned no data (water/urban pixel?)")
        return out, "ISRIC SoilGrids 250 m"
    except Exception as e:  # noqa: BLE001
        log.warning("SoilGrids fallback: %s", e)
        return {"clay": 25.0, "sand": 40.0, "silt": 35.0, "soc": 8.0,
                "phh2o": 7.2, "nitrogen": 0.9, "cec": 15.0, "bdod": 1.4}, "fallback (demo values)"


# --------------------------------------------------------------------------
# 3. NASA MODIS NDVI via ORNL DAAC TESViS REST API - "field memory"
# --------------------------------------------------------------------------
ORNL = "https://modis.ornl.gov/rst/api/v1"


def fetch_modis_ndvi(lat: float, lon: float, n_dates: int = 46) -> tuple[dict, str]:
    """~2 years of 16-day NDVI at 250 m. API allows max 10 dates per request."""
    hdr = {"Accept": "application/json"}
    try:
        dates = _get(f"{ORNL}/MOD13Q1/dates", {"latitude": lat, "longitude": lon}, hdr)["dates"]
        dates = dates[-n_dates:]
        series = []
        for i in range(0, len(dates), 10):
            chunk = dates[i:i + 10]
            js = _get(f"{ORNL}/MOD13Q1/subset", {
                "latitude": lat, "longitude": lon, "band": "250m_16_days_NDVI",
                "startDate": chunk[0]["modis_date"], "endDate": chunk[-1]["modis_date"],
                "kmAboveBelow": 0, "kmLeftRight": 0,
            }, hdr)
            for s in js["subset"]:
                v = s["data"][0]
                if v > -2000:  # -3000 = fill
                    series.append((s["calendar_date"], v * 0.0001))
        if not series:
            raise ValueError("no NDVI values")
        return summarize_ndvi(series), "NASA MODIS MOD13Q1 (ORNL DAAC)"
    except Exception as e:  # noqa: BLE001
        log.warning("MODIS fallback: %s", e)
        t = np.arange(46)
        vals = 0.35 + 0.25 * np.sin(2 * np.pi * t / 23)  # two seasons/yr demo
        start = dt.date.today() - dt.timedelta(days=16 * 46)
        series = [((start + dt.timedelta(days=16 * int(i))).isoformat(), float(v))
                  for i, v in zip(t, vals)]
        return summarize_ndvi(series), "fallback (demo values)"


def summarize_ndvi(series: list[tuple[str, float]]) -> dict:
    vals = np.array([v for _, v in series])
    # Count green-up peaks (each ~= one crop cycle)
    peaks = [i for i in range(1, len(vals) - 1)
             if vals[i] >= vals[i - 1] and vals[i] > vals[i + 1] and vals[i] > 0.4]
    span_years = max(len(vals) * 16 / 365, 0.5)
    return {
        "series": series,
        "mean_ndvi": float(vals.mean()),
        "peak_ndvi": float(vals.max()),
        "crop_cycles_per_year": round(len(peaks) / span_years, 1),
        "bare_soil_fraction": float(np.mean(vals < 0.2)),  # time exposed to erosion
    }


# --------------------------------------------------------------------------
# 4. Open-Meteo Climate API (CMIP6 HighResMIP) - future climate to 2050
# --------------------------------------------------------------------------
OPEN_METEO_CLIMATE = "https://climate-api.open-meteo.com/v1/climate"


def fetch_climate_projection(lat: float, lon: float,
                             model: str = "MRI_AGCM3_2_S") -> tuple[dict, str]:
    """Change between 2001-2010 and 2041-2050: temperature (deg C) and rainfall (%)."""
    def window(start: str, end: str) -> tuple[float, float]:
        js = _get(OPEN_METEO_CLIMATE, {
            "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
            "models": model, "daily": "temperature_2m_mean,precipitation_sum",
        })
        d = js["daily"]
        tkey = next(k for k in d if k.startswith("temperature_2m_mean"))
        pkey = next(k for k in d if k.startswith("precipitation_sum"))
        t = np.array(d[tkey], float)
        p = np.array(d[pkey], float)
        years = (int(end[:4]) - int(start[:4]) + 1)
        return float(np.nanmean(t)), float(np.nansum(p) / years)

    try:
        t0, p0 = window("2001-01-01", "2010-12-31")
        t1, p1 = window("2041-01-01", "2050-12-31")
        return {"delta_t": t1 - t0, "delta_p_pct": (p1 - p0) / max(p0, 1) * 100,
                "model": model}, f"Open-Meteo CMIP6 ({model})"
    except Exception as e:  # noqa: BLE001
        log.warning("Climate projection fallback: %s", e)
        return {"delta_t": 1.5, "delta_p_pct": -5.0, "model": "assumed"}, "fallback (demo values)"


# --------------------------------------------------------------------------
# 5. OPTIONAL: NASA Earthdata granule search via earthaccess
#    pip install earthaccess ; needs free Earthdata Login (~/.netrc or env vars)
# --------------------------------------------------------------------------
USEFUL_COLLECTIONS = {
    "SMAP soil moisture 9 km daily": "SPL3SMP_E",
    "GPM IMERG daily rainfall": "GPM_3IMERGDF",
    "GRACE-FO groundwater (GLDAS-assimilated)": "GRACEDADM_CLSM025GL_7D",
}


def search_earthdata(short_name: str, lat: float, lon: float,
                     start: str, end: str, count: int = 5) -> list[str]:
    """Return download links for granules covering the field. Advanced use:
    open them with xarray to build seasonal soil-moisture / rainfall stats."""
    import earthaccess  # imported lazily so the app works without it

    earthaccess.login(strategy="environment")
    bbox = (lon - 0.1, lat - 0.1, lon + 0.1, lat + 0.1)
    results = earthaccess.search_data(short_name=short_name, bounding_box=bbox,
                                      temporal=(start, end), count=count)
    return [g.data_links()[0] for g in results if g.data_links()]


# --------------------------------------------------------------------------
_CACHE: TTLCache = TTLCache(maxsize=256, ttl=int(os.getenv("CACHE_TTL_SECONDS", "86400")))


def build_site_data(lat: float, lon: float) -> dict:
    """Gather everything the engine needs. The five calls run in parallel.
    Complete results are cached per location (default 24 h); results that
    used fallback values are not cached, so the next request retries."""
    key = (round(lat, 4), round(lon, 4))
    if key in _CACHE:
        return _CACHE[key]
    with ThreadPoolExecutor(max_workers=5) as pool:
        f_clim = pool.submit(fetch_power_climatology, lat, lon)
        f_ext = pool.submit(fetch_power_extremes, lat, lon)
        f_soil = pool.submit(fetch_soilgrids, lat, lon)
        f_ndvi = pool.submit(fetch_modis_ndvi, lat, lon)
        f_fut = pool.submit(fetch_climate_projection, lat, lon)
        (climate, s1), (extremes, s2), (soil, s3) = f_clim.result(), f_ext.result(), f_soil.result()
        (ndvi, s4), (future, s5) = f_ndvi.result(), f_fut.result()
    site = {
        "lat": lat, "lon": lon, "climate": climate, "extremes": extremes,
        "soil": soil, "ndvi": ndvi, "future": future,
        "sources": {"Climate normals": s1, "Weather extremes": s2,
                    "Soil": s3, "Vegetation history": s4, "Future climate": s5},
    }
    if not any("fallback" in v for v in site["sources"].values()):
        _CACHE[key] = site
    return site
