"""
weather.py - daily weather for one farm year (1 May to 30 April).

Observed: NASA POWER daily point data (Tmax, Tmin, rainfall).
Fallback: synthetic daily weather built from monthly normals, clearly labelled.
Reference evapotranspiration (ET0) uses the Hargreaves equation (FAO-56 eq. 52),
which needs only temperature and latitude.
"""
from __future__ import annotations

import datetime as dt
import logging
import math

import numpy as np
from cachetools import TTLCache

from ..services.data_sources import POWER_BASE, _get, fetch_power_climatology

log = logging.getLogger("rotaterra.weather")
_CACHE: TTLCache = TTLCache(maxsize=64, ttl=7 * 86400)

# Approximate monthly normals for the default demo farm (central Punjab, Pakistan),
# used only when NASA POWER cannot be reached. Jan..Dec.
_PUNJAB_DEMO = {
    "t_max": [19, 22, 27, 34, 39, 40, 36, 35, 35, 32, 26, 21],
    "t_min": [5, 8, 13, 19, 24, 27, 27, 27, 24, 18, 11, 6],
    "rain": [15, 20, 25, 15, 15, 35, 110, 100, 40, 8, 5, 10],
}
DEMO_FARM = (31.42, 73.08)


def farm_year_dates(year: int) -> list[dt.date]:
    start, end = dt.date(year, 5, 1), dt.date(year + 1, 4, 30)
    return [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]


def extraterrestrial_radiation(lat: float, doy: np.ndarray) -> np.ndarray:
    """Ra in MJ/m2/day, FAO-56 eq. 21."""
    phi = math.radians(lat)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
    return (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))


def hargreaves_et0(tmax, tmin, lat, doy) -> np.ndarray:
    tmean = (tmax + tmin) / 2
    ra_mm = 0.408 * extraterrestrial_radiation(lat, doy)
    return np.maximum(0.0, 0.0023 * ra_mm * (tmean + 17.8) * np.sqrt(np.maximum(tmax - tmin, 0)))


def _from_power(lat, lon, dates) -> dict:
    js = _get(f"{POWER_BASE}/daily/point", {
        "parameters": "T2M_MAX,T2M_MIN,PRECTOTCORR", "community": "AG",
        "latitude": lat, "longitude": lon, "format": "JSON",
        "start": dates[0].strftime("%Y%m%d"), "end": dates[-1].strftime("%Y%m%d"),
    })
    p = js["properties"]["parameter"]
    keys = [d.strftime("%Y%m%d") for d in dates]
    tmax = np.array([p["T2M_MAX"].get(k, -999) for k in keys], float)
    tmin = np.array([p["T2M_MIN"].get(k, -999) for k in keys], float)
    rain = np.array([p["PRECTOTCORR"].get(k, -999) for k in keys], float)
    for arr in (tmax, tmin, rain):
        bad = arr <= -998
        if bad.mean() > 0.1:
            raise ValueError("too many missing days in NASA POWER response")
        if bad.any():   # fill the odd missing day by interpolation
            idx = np.arange(len(arr))
            arr[bad] = np.interp(idx[bad], idx[~bad], arr[~bad])
    return {"tmax": tmax, "tmin": tmin, "rain": np.maximum(rain, 0)}


def _synthetic(lat, lon, dates) -> tuple[dict, str]:
    near_demo = abs(lat - DEMO_FARM[0]) < 3 and abs(lon - DEMO_FARM[1]) < 3
    if near_demo:
        normals, label = _PUNJAB_DEMO, "synthetic demo weather (approximate Punjab normals)"
    else:
        clim, _ = fetch_power_climatology(lat, lon)
        normals = {"t_max": clim["t_max"], "t_min": clim["t_min"], "rain": clim["precip_mm"]}
        label = "synthetic demo weather (from climate normals)"
    rng = np.random.default_rng(abs(hash((round(lat, 2), round(lon, 2), dates[0].year))) % 2**32)
    # interpolate monthly normals to days (normals sit mid-month)
    doy = np.array([d.timetuple().tm_yday for d in dates])
    mid = np.array([15 + 30.4 * i for i in range(12)])
    def daily(vals):
        return np.interp(doy, np.r_[mid - 365, mid, mid + 365], np.r_[vals, vals, vals])
    tmax = daily(normals["t_max"]) + rng.normal(0, 1.8, len(dates))
    tmin = daily(normals["t_min"]) + rng.normal(0, 1.5, len(dates))
    tmin = np.minimum(tmin, tmax - 3)
    rain = np.zeros(len(dates))
    for i, d in enumerate(dates):
        monthly = normals["rain"][d.month - 1]
        wet_days = max(1.0, min(15.0, monthly / 9))
        if rng.random() < wet_days / 30:
            rain[i] = rng.gamma(1.4, monthly / wet_days / 1.4)
    return {"tmax": tmax, "tmin": tmin, "rain": rain}, label


def get_farm_weather(lat: float, lon: float, year: int) -> dict:
    """Daily weather + ET0 for 1 May `year` to 30 April `year+1`."""
    key = (round(lat, 3), round(lon, 3), year)
    if key in _CACHE:
        return _CACHE[key]
    dates = farm_year_dates(year)
    try:
        w, source, observed = _from_power(lat, lon, dates), "NASA POWER daily (observed)", True
    except Exception as e:  # noqa: BLE001
        log.warning("farm weather fallback: %s", e)
        (w, source), observed = _synthetic(lat, lon, dates), False
    doy = np.array([d.timetuple().tm_yday for d in dates])
    w["et0"] = hargreaves_et0(w["tmax"], w["tmin"], lat, doy)
    w.update(dates=dates, source=source, observed=observed)
    if observed:
        _CACHE[key] = w
    return w
