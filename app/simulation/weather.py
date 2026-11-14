"""
weather.py - daily weather for one farm year (1 May to 30 April).

Observed: NASA POWER daily point data (Tmax, Tmin, rainfall).
Fallback: synthetic daily weather built from monthly normals, clearly labelled.
Reference evapotranspiration (ET0) uses the Hargreaves equation (FAO-56 eq. 52),
which needs only temperature and latitude.
"""
from __future__ import annotations

import calendar
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
    """Daily weather + ET0 for 1 May `year` to 30 April `year+1`.

    A finished farm year is NASA POWER observations throughout. The season that is
    still running is built in three layers (see _current_season): NASA observations up
    to the latest published day, the Open-Meteo forecast for the next 16 days, then
    NASA long-term daily normals for the rest of the season."""
    today = dt.date.today()
    dates = farm_year_dates(year)
    running = dates[-1] >= today - dt.timedelta(days=7)
    key = (round(lat, 3), round(lon, 3), year, today.isoformat() if running else "")
    if key in _CACHE:
        return _CACHE[key]
    if running:
        w, source, observed, timeline = _current_season(lat, lon, dates, today)
    else:
        timeline = {"observed_until": dates[-1].isoformat(), "forecast_until": None}
        try:
            w, source, observed = _from_power(lat, lon, dates), "NASA POWER daily (observed)", True
        except Exception as e:  # noqa: BLE001
            log.warning("farm weather fallback: %s", e)
            (w, source), observed = _synthetic(lat, lon, dates), False
            timeline = {"observed_until": None, "forecast_until": None}
    doy = np.array([d.timetuple().tm_yday for d in dates])
    w["et0"] = hargreaves_et0(w["tmax"], w["tmin"], lat, doy)
    w.update(dates=dates, source=source, observed=observed, timeline=timeline)
    if observed:
        _CACHE[key] = w
    return w


OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"


def _power_partial(lat, lon, start: dt.date, end: dt.date) -> dict:
    """NASA POWER daily values for start..end; days not yet published are NaN."""
    js = _get(f"{POWER_BASE}/daily/point", {
        "parameters": "T2M_MAX,T2M_MIN,PRECTOTCORR", "community": "AG",
        "latitude": lat, "longitude": lon, "format": "JSON",
        "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"),
    }, ttl=6 * 3600)
    p = js["properties"]["parameter"]
    out = {}
    for name, key in (("tmax", "T2M_MAX"), ("tmin", "T2M_MIN"), ("rain", "PRECTOTCORR")):
        out[name] = {k: (np.nan if v is None or v <= -998 else float(v)) for k, v in p[key].items()}
    return out


def _forecast_days(lat, lon) -> dict:
    """Open-Meteo daily forecast (16 days ahead, plus the last 10 days so NASA's
    publishing delay is covered). {date: (tmax, tmin, rain)}"""
    js = _get(OPEN_METEO_FORECAST, {
        "latitude": lat, "longitude": lon, "timezone": "auto", "past_days": 10, "forecast_days": 16,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
    }, ttl=3 * 3600, timeout=20)
    d = js["daily"]
    return {dt.date.fromisoformat(t): (a, b, c) for t, a, b, c in zip(
        d["time"], d["temperature_2m_max"], d["temperature_2m_min"], d["precipitation_sum"]) if a is not None and b is not None}


def _normals(lat, lon, dates) -> dict:
    """Typical weather for each calendar day: the mean of the last 10 years of NASA
    POWER daily observations at this point, smoothed over about two weeks.
    (NASA's monthly climatology T2M_MAX/T2M_MIN are the month's extremes, not a
    typical day, so they are not used here.)"""
    today = dt.date.today()
    start, end = dt.date(today.year - 10, 1, 1), dt.date(today.year - 1, 12, 31)
    js = _get(f"{POWER_BASE}/daily/point", {
        "parameters": "T2M_MAX,T2M_MIN,PRECTOTCORR", "community": "AG",
        "latitude": lat, "longitude": lon, "format": "JSON",
        "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"),
    }, timeout=90)
    p = js["properties"]["parameter"]
    sums = {k: np.zeros(366) for k in ("tmax", "tmin", "rain")}
    counts = np.zeros(366)
    for key, tmax in p["T2M_MAX"].items():
        tmin, rain = p["T2M_MIN"].get(key, -999), p["PRECTOTCORR"].get(key, -999)
        if min(tmax, tmin, rain) <= -998:
            continue
        d = dt.date(int(key[:4]), int(key[4:6]), int(key[6:]))
        i = (d - dt.date(d.year, 1, 1)).days if not (d.month == 2 and d.day == 29) else 58
        if d.month > 2 and calendar.isleap(d.year):
            i -= 1                     # keep 1 March on the same index in leap years
        sums["tmax"][i] += tmax; sums["tmin"][i] += tmin; sums["rain"][i] += max(0.0, rain); counts[i] += 1
    counts[365] = 0
    ok = counts[:365] > 0
    if ok.sum() < 300:
        raise ValueError("not enough NASA POWER history for normals")
    kernel = np.ones(15) / 15
    out = {}
    for k in sums:
        mean = np.where(ok, sums[k][:365] / np.maximum(counts[:365], 1), np.nan)
        mean = np.interp(np.arange(365), np.flatnonzero(ok), mean[ok])
        smooth = np.convolve(np.r_[mean[-7:], mean, mean[:7]], kernel, mode="valid")   # circular
        out[k] = smooth
    idx = np.array([min(364, (d - dt.date(d.year, 1, 1)).days - (1 if d.month > 2 and calendar.isleap(d.year) else 0)) for d in dates])
    return {k: out[k][idx] for k in out}


def _normals_fallback(lat, lon, dates) -> dict:
    """Used only if the daily history is unavailable: monthly mean temperature with a
    typical day-night spread, and monthly rain spread over the month."""
    clim, _ = fetch_power_climatology(lat, lon)
    doy = np.array([d.timetuple().tm_yday for d in dates])
    mid = np.array([15 + 30.4 * i for i in range(12)])
    def daily(vals):
        return np.interp(doy, np.r_[mid - 365, mid, mid + 365], np.r_[vals, vals, vals])
    tmean = daily(clim["t_mean"])
    days_in = np.array([calendar.monthrange(d.year, d.month)[1] for d in dates])
    rain_month = np.array([clim["precip_mm"][d.month - 1] for d in dates])
    return {"tmax": tmean + 6.5, "tmin": tmean - 6.5, "rain": rain_month / days_in}


def _current_season(lat, lon, dates, today):
    n = len(dates)
    try:
        normal, normal_src = _normals(lat, lon, dates), "NASA POWER 10-year daily normals"
    except Exception as e:  # noqa: BLE001
        log.warning("season: daily normals unavailable, using climatology: %s", e)
        normal, normal_src = _normals_fallback(lat, lon, dates), "NASA POWER climate normals"
    tmax, tmin, rain = normal["tmax"].copy(), normal["tmin"].copy(), normal["rain"].copy()
    observed_until = forecast_until = None
    parts = []
    # 1. NASA POWER observations up to the latest published day
    try:
        obs = _power_partial(lat, lon, dates[0], min(today - dt.timedelta(days=1), dates[-1]))
        for i, d in enumerate(dates):
            k = d.strftime("%Y%m%d")
            vals = (obs["tmax"].get(k, np.nan), obs["tmin"].get(k, np.nan), obs["rain"].get(k, np.nan))
            if d >= today or any(np.isnan(v) for v in vals):
                continue
            tmax[i], tmin[i], rain[i] = vals[0], vals[1], max(0.0, vals[2])
            observed_until = d
        if observed_until:
            parts.append(f"NASA POWER observed to {observed_until.isoformat()}")
    except Exception as e:  # noqa: BLE001
        log.warning("season: NASA POWER unavailable: %s", e)
    # 2. Forecast for the days after the last observation (and the next 16 days)
    try:
        fc = _forecast_days(lat, lon)
        start = observed_until or (dates[0] - dt.timedelta(days=1))
        for i, d in enumerate(dates):
            if d > start and d in fc:
                tmax[i], tmin[i], rain[i] = fc[d][0], fc[d][1], max(0.0, fc[d][2] or 0.0)
                forecast_until = d
        if forecast_until:
            parts.append(f"Open-Meteo forecast to {forecast_until.isoformat()}")
    except Exception as e:  # noqa: BLE001
        log.warning("season: forecast unavailable: %s", e)
    parts.append(f"then {normal_src}")
    tmin = np.minimum(tmin, tmax - 1)
    w = {"tmax": np.asarray(tmax, float), "tmin": np.asarray(tmin, float), "rain": np.asarray(rain, float)}
    timeline = {"observed_until": observed_until.isoformat() if observed_until else None,
                "forecast_until": forecast_until.isoformat() if forecast_until else None,
                "today": today.isoformat()}
    return w, "; ".join(parts), observed_until is not None, timeline
