"""
data_sources.py - every external API call lives here.

APIs used (all free; only earthaccess needs a free NASA Earthdata Login):
  1. NASA POWER (Agroclimatology)  - climate normals + 10-yr daily history
  2. ISRIC SoilGrids v2.0          - soil texture, organic carbon, pH, N, CEC
  3. NASA MODIS NDVI via NASA GIBS (images), ORNL DAAC as backup - greenness history
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
        "parameters": "T2M,T2M_RANGE,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN,GWETROOT",
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
            # Typical day: mean temperature +/- half the mean daily range. (POWER's
            # climatology T2M_MAX / T2M_MIN are the month's record extremes.)
            "t_max": [p["T2M"][m] + p["T2M_RANGE"][m] / 2 for m in MONTHS],
            "t_min": [p["T2M"][m] - p["T2M_RANGE"][m] / 2 for m in MONTHS],
            "t_max_record": [p["T2M_MAX"][m] for m in MONTHS],
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
    def query(la: float, lo: float) -> dict:
        params = [("lon", round(lo, 4)), ("lat", round(la, 4)), ("value", "mean")]
        params += [("property", p) for p in SOIL_PROPS]
        params += [("depth", d) for d in DEPTHS]
        js = _get(SOILGRIDS, params=params, timeout=25)   # SoilGrids is slow; success is cached
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
        return out

    # A district picked from the list is its town centre, which SoilGrids maps as
    # "no soil" (city or water). Then try farmland just outside the town (SoilGrids
    # allows about 5 requests a minute, so at most 5 points).
    tries = [(0, 0), (0.1, 0), (-0.1, 0), (0, 0.12), (0, -0.12)]   # ~11 km: past a big town
    last = None
    for dla, dlo in tries:
        try:
            out = query(lat + dla, lon + dlo)
            if len(out) >= 4:
                return out, "ISRIC SoilGrids 250 m" + ("" if (dla, dlo) == (0, 0) else " (nearest farmland)")
            last = ValueError("SoilGrids returned no data (water/urban pixel)")
        except Exception as e:  # noqa: BLE001
            last = e
            if "timed out" in str(e).lower() or "timeout" in type(e).__name__.lower():
                break                      # the service is down: don't wait five times
    log.warning("SoilGrids fallback: %s", last)
    return {"clay": 25.0, "sand": 40.0, "silt": 35.0, "soc": 8.0,
            "phh2o": 7.2, "nitrogen": 0.9, "cec": 15.0, "bdod": 1.4}, "fallback (demo values)"


# --------------------------------------------------------------------------
# 3. NASA MODIS NDVI via ORNL DAAC TESViS REST API - "field memory"
# --------------------------------------------------------------------------
ORNL = "https://modis.ornl.gov/rst/api/v1"


GIBS_WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
GIBS_NDVI_LAYER = "MODIS_Terra_L3_NDVI_16Day"
GIBS_NDVI_COLORS = "https://gibs.earthdata.nasa.gov/colormaps/v1.0/MODIS_L3_NDVI.xml"
_ndvi_colors: dict | None = None


def _ndvi_colormap() -> dict:
    """NASA's published colour table for the NDVI images: each colour is one NDVI band."""
    global _ndvi_colors
    if _ndvi_colors is None:
        import re
        r = requests.get(GIBS_NDVI_COLORS, timeout=30)
        r.raise_for_status()
        cm = {}
        for m in re.finditer(r'rgb="(\d+),(\d+),(\d+)" transparent="(\w+)" value="\[([-\d.]+),([-\d.]+)\)"', r.text):
            if m.group(4) != "true":
                cm[(int(m.group(1)), int(m.group(2)), int(m.group(3)))] = (float(m.group(5)) + float(m.group(6))) / 2
        if len(cm) < 50:
            raise ValueError("NDVI colour table not understood")
        _ndvi_colors = cm
    return _ndvi_colors


def _png_pixels(buf: bytes) -> list[tuple[int, int, int, int]]:
    """Minimal PNG reader (8-bit RGB, RGBA or palette), standard library only."""
    import struct
    import zlib
    if buf[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    pos, idat, plte, trns, w = 8, [], b"", b"", 0
    while pos < len(buf):
        (n,) = struct.unpack(">I", buf[pos:pos + 4])
        kind, data = buf[pos + 4:pos + 8], buf[pos + 8:pos + 8 + n]
        pos += 12 + n
        if kind == b"IHDR":
            w, h, depth, ctype = struct.unpack(">IIBB", data[:10])
        elif kind == b"PLTE":
            plte = data
        elif kind == b"tRNS":
            trns = data
        elif kind == b"IDAT":
            idat.append(data)
    if depth != 8 or ctype not in (2, 3, 6):
        raise ValueError(f"unsupported PNG type {ctype}/{depth}")
    bpp = {2: 3, 3: 1, 6: 4}[ctype]
    raw, stride = zlib.decompress(b"".join(idat)), w * bpp
    prev, rows = bytearray(stride), []
    for y in range(h):
        f, line = raw[y * (stride + 1)], raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)]
        cur = bytearray(stride)
        for x in range(stride):
            a = cur[x - bpp] if x >= bpp else 0
            b = prev[x]
            c = prev[x - bpp] if x >= bpp else 0
            v = line[x]
            if f == 1:
                v += a
            elif f == 2:
                v += b
            elif f == 3:
                v += (a + b) >> 1
            elif f == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                v += a if pa <= pb and pa <= pc else (b if pb <= pc else c)
            cur[x] = v & 0xFF
        rows.append(cur)
        prev = cur
    out = []
    for row in rows:
        for i in range(w):
            if ctype == 3:
                k = row[i]
                out.append((plte[3 * k], plte[3 * k + 1], plte[3 * k + 2], trns[k] if k < len(trns) else 255))
            else:
                p = row[i * bpp:(i + 1) * bpp]
                out.append((p[0], p[1], p[2], p[3] if bpp == 4 else 255))
    return out


def _gibs_ndvi_on(lat: float, lon: float, day: str) -> float | None:
    """NDVI of the farmland around a point on one 16-day composite: a 22 x 22 km box,
    read from NASA's MODIS NDVI image, and the greener quarter of its pixels (the
    fields, not the town). Cached on disk, since a past composite never changes."""
    key = hashlib.sha1(f"gibs-ndvi|{lat:.3f}|{lon:.3f}|{day}".encode()).hexdigest()
    path = CACHE_DIR / f"{key}.json"
    try:
        return json.loads(path.read_text("utf-8"))["v"]
    except (OSError, ValueError, KeyError):
        pass
    d = 0.1
    r = requests.get(GIBS_WMS, params={
        "SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.3.0", "LAYERS": GIBS_NDVI_LAYER,
        "CRS": "EPSG:4326", "BBOX": f"{lat - d},{lon - d},{lat + d},{lon + d}",
        "WIDTH": 40, "HEIGHT": 40, "FORMAT": "image/png", "TIME": day,
    }, timeout=25)
    r.raise_for_status()
    colors = _ndvi_colormap()
    vals = [colors[p[:3]] for p in _png_pixels(r.content) if p[3] > 0 and p[:3] in colors]
    v = float(np.percentile(vals, 75)) if len(vals) >= 40 else None
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"v": v}), "utf-8")
    except OSError:
        pass
    return v


def _gibs_ndvi_series(lat: float, lon: float, n_dates: int = 46) -> list[tuple[str, float]]:
    """About two years of 16-day NDVI from NASA GIBS (fast, served from a CDN)."""
    today = dt.date.today()
    days = [dt.date(y, 1, 1) + dt.timedelta(days=16 * k) for y in range(today.year - 2, today.year + 1) for k in range(23)]
    days = [d for d in days if d <= today - dt.timedelta(days=24)][-n_dates:]   # composites appear ~3 weeks late
    with ThreadPoolExecutor(max_workers=8) as pool:
        got = list(pool.map(lambda d: _safe(_gibs_ndvi_on, lat, lon, d.isoformat()), days))
    series = [(d.isoformat(), v) for d, v in zip(days, got) if v is not None]
    if len(series) < 20:
        raise ValueError(f"only {len(series)} NDVI images")
    return series


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001
        log.info("NDVI image skipped: %s", e)
        return None


def fetch_modis_ndvi(lat: float, lon: float, n_dates: int = 46) -> tuple[dict, str]:
    """~2 years of 16-day MODIS NDVI around the field. First NASA GIBS (images, fast and
    reliable), then the ORNL DAAC point service, then demo values."""
    try:
        return summarize_ndvi(_gibs_ndvi_series(lat, lon, n_dates)), "NASA MODIS NDVI 16-day (NASA GIBS), farmland around the field"
    except Exception as e:  # noqa: BLE001
        log.warning("GIBS NDVI unavailable, trying ORNL: %s", e)
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
                # A 4 x 4 km box (17 x 17 pixels) around the point, not one pixel: a
                # district picked from the list is its town centre, and a single pixel
                # there is roofs and roads. The greener quarter of the box is the farmland.
                "kmAboveBelow": 2, "kmLeftRight": 2,
            }, hdr, timeout=20)
            for s in js["subset"]:
                px = np.array([v for v in s["data"] if v > -2000], float)   # -3000 = fill
                if px.size:
                    series.append((s["calendar_date"], float(np.percentile(px, 75)) * 0.0001))
        if not series:
            raise ValueError("no NDVI values")
        return summarize_ndvi(series), "NASA MODIS MOD13Q1 (ORNL DAAC), 4 km around the field"
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
    # Count green-up peaks (each ~= one crop cycle): on a lightly smoothed curve, peaks
    # above 0.4 that rise clearly above the dip before them and are at least ~80 days
    # apart, so a wobble between two images is not counted as another crop.
    sm = np.convolve(np.r_[vals[:1], vals, vals[-1:]], np.ones(3) / 3, mode="valid") if len(vals) >= 3 else vals
    peaks, low = [], float(sm[0]) if len(sm) else 0.0
    for i in range(1, len(sm) - 1):
        low = min(low, float(sm[i]))
        if sm[i] >= sm[i - 1] and sm[i] > sm[i + 1] and sm[i] > 0.4 and sm[i] - low >= 0.08:
            if peaks and i - peaks[-1] < 5:
                if sm[i] > sm[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)
            low = float(sm[i])
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
