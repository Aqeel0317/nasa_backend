"""
scenarios.py - change a farm year's weather to test "what if" conditions.

The same weather applies to every plot, as on a real farm. Heatwaves are placed
from mid to late August and in early March, when the kharif crops and wheat are flowering
under the default sowing calendar.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

SCENARIOS = ("normal", "drought", "heatwave", "flood", "custom")


def _window(dates, month, day_from, day_to):
    return np.array([d.month == month and day_from <= d.day <= day_to for d in dates])


def apply_scenario(w: dict, scenario: str, temp_offset: float = 0.0,
                   rain_pct: float = 100.0) -> dict:
    tmax, tmin, rain = w["tmax"].copy(), w["tmin"].copy(), w["rain"].copy()
    dates = w["dates"]
    if scenario == "drought":
        rain *= 0.35
        tmax += 1.5            # drier air, more sunshine
    elif scenario == "heatwave":
        hot = _window(dates, 8, 15, 31) | _window(dates, 3, 1, 14)
        tmax[hot] += 6.0
        tmin[hot] += 3.0
    elif scenario == "flood":
        rain *= 1.7
        for when in (dt.date(dates[0].year, 8, 5), dt.date(dates[0].year, 8, 20),
                     dt.date(dates[0].year + 1, 2, 15)):
            i = (when - dates[0]).days
            rain[i:i + 2] += 80.0   # two-day storm
    elif scenario == "custom":
        tmax += temp_offset
        tmin += temp_offset
        rain *= rain_pct / 100.0
    return {**w, "tmax": tmax, "tmin": tmin, "rain": rain}
