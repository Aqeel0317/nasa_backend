"""
engine.py - turns site data + farmer priorities into ranked rotations.

Pipeline
  1. detect_seasons()      split the year into a warm and a cool 6-month season
  2. season_climate()      season temperature / rain / soil wetness (now or 2041-50)
  3. crop_suitability()    0..1 fit of each crop to climate, water and soil
  4. search_rotations()    beam search over multi-year crop sequences
  5. simulate()            season-by-season soil-health trajectory + water ledger
  6. score()               weighted by farmer priorities, stress-tested on future climate
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .crops import FALLOW, Crop, crops_for_season


# ------------------------------------------------------------------ seasons
def detect_seasons(t_mean: list[float]) -> dict[str, list[int]]:
    """Warm season = the 6 consecutive months with the highest mean temperature."""
    t = np.array(t_mean)
    best = max(range(12), key=lambda s: sum(t[(s + i) % 12] for i in range(6)))
    warm = [(best + i) % 12 for i in range(6)]
    cool = [m for m in range(12) if m not in warm]
    cool = sorted(cool, key=lambda m: (m - (best + 6)) % 12)
    return {"warm": warm, "cool": cool}


def season_climate(site: dict, season: str, future: bool = False) -> dict:
    clim = site["climate"]
    months = detect_seasons(clim["t_mean"])[season]
    dt = site["future"]["delta_t"] if future else 0.0
    dp = 1 + (site["future"]["delta_p_pct"] / 100 if future else 0.0)
    return {
        "t_mean": float(np.mean([clim["t_mean"][m] for m in months]) + dt),
        "t_max": float(np.max([clim["t_max"][m] for m in months]) + dt),
        "rain_mm": float(np.sum([clim["precip_mm"][m] for m in months]) * dp),
        "wetness": float(np.mean([clim["root_wetness"][m] for m in months])),
    }


# ------------------------------------------------------------------ suitability
def texture_class(soil: dict) -> str:
    if soil.get("sand", 0) > 65:
        return "light"
    if soil.get("clay", 0) > 40:
        return "heavy"
    return "medium"


def _trapezoid(x, a, b, c, d) -> float:
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    return (x - a) / (b - a) if x < b else (d - x) / (d - c)


def crop_suitability(crop: Crop, sc: dict, soil: dict, irrigation_mm: float) -> dict:
    if crop is FALLOW:
        return {"score": 1.0, "irrigation_needed": 0.0, "notes": []}
    notes = []
    temp = _trapezoid(sc["t_mean"], crop.t_min, crop.t_opt_low, crop.t_opt_high, crop.t_max)

    heat = 1.0
    if sc["t_max"] > crop.t_max - 2:
        heat = 1 - min(1, (sc["t_max"] - crop.t_max + 2) / 8) * (1 - crop.heat_tol)
        if heat < 0.8:
            notes.append("heat stress risk at peak temperatures")

    effective_rain = 0.7 * sc["rain_mm"] + 60 * sc["wetness"]  # rain + stored soil water
    gap = max(0.0, crop.water_mm - effective_rain)
    irrigation_needed = min(gap, irrigation_mm)
    unmet = max(0.0, gap - irrigation_mm) / crop.water_mm
    water = 1 - unmet * (1 - 0.6 * crop.drought_tol)
    if unmet > 0.3:
        notes.append(f"~{gap - irrigation_mm:.0f} mm water shortfall")
    if sc["rain_mm"] > 2 * crop.water_mm and crop.waterlog_tol < 0.4:
        water *= 0.8
        notes.append("waterlogging risk in wet seasons")

    ph = soil.get("phh2o", 7)
    ph_s = 1.0 if crop.ph_min <= ph <= crop.ph_max else max(
        0.0, 1 - 0.3 * min(abs(ph - crop.ph_min), abs(ph - crop.ph_max)))
    tex = 1.0 if crop.texture in ("any", texture_class(soil)) else 0.75

    score = (0.35 * temp + 0.35 * water + 0.15 * ph_s + 0.15 * tex) * heat
    if temp == 0:
        score *= 0.2  # outside survivable temperature range
    return {"score": round(float(score), 3), "irrigation_needed": irrigation_needed, "notes": notes}


def suitability_table(site: dict, irrigation_mm: float, allowed=None) -> list[dict]:
    rows = []
    for season in ("warm", "cool"):
        now, fut = season_climate(site, season), season_climate(site, season, future=True)
        for c in crops_for_season(season, allowed):
            rows.append({
                "crop": c.name, "season": season, "role": c.role,
                "now": crop_suitability(c, now, site["soil"], irrigation_mm)["score"],
                "2041-2050": crop_suitability(c, fut, site["soil"], irrigation_mm)["score"],
            })
    return rows


# ------------------------------------------------------------------ simulation
@dataclass
class Result:
    sequence: list[str]
    seasons: list[str]
    scores: dict
    total: float
    future_total: float = 0.0
    robustness: float = 0.0
    trajectory: dict = field(default_factory=dict)
    explanations: list[str] = field(default_factory=list)


def simulate(seq: list[Crop], seasons: list[str], site: dict,
             irrigation_mm: float, future: bool = False) -> dict:
    soil = site["soil"]
    soc = float(soil.get("soc", 8.0))            # g/kg, starting topsoil carbon
    n_bank, pest, irrigation_total, need_total = 0.0, 0.0, 0.0, 0.0
    soc_traj, n_traj, pest_traj, suit = [soc], [0.0], [0.0], []
    notes: list[str] = []
    prev: Crop | None = None

    for crop, season in zip(seq, seasons):
        sc = season_climate(site, season, future)
        s = crop_suitability(crop, sc, soil, irrigation_mm)
        suit.append(s["score"])

        # Soil carbon: residue input minus decomposition (faster when hot)
        decomp = 0.25 * (1 + max(0, sc["t_mean"] - 20) / 30)
        gain = crop.residue_c * 0.6 * max(s["score"], 0.3)
        if crop is FALLOW:
            gain, decomp = 0.0, decomp + 0.15   # bare soil: erosion + oxidation
        soc = max(1.0, soc + gain - decomp)

        # Nitrogen: legumes bank N, heavy feeders draw it down
        n_bank = 0.7 * n_bank + crop.n_effect
        if prev and prev.family == "Fabaceae" and crop.family == "Poaceae":
            notes.append(f"{prev.name} before {crop.name} supplies nitrogen to the cereal")

        # Pest/disease pressure builds when the same family repeats
        if prev and crop.family == prev.family and crop is not FALLOW:
            pest += 1.0
            notes.append(f"{prev.name} → {crop.name}: same plant family raises pest carry-over")
        else:
            pest *= 0.4

        irrigation_total += s["irrigation_needed"]
        need_total += crop.water_mm
        soc_traj.append(round(soc, 2))
        n_traj.append(round(n_bank, 1))
        pest_traj.append(round(pest, 2))
        prev = crop

    families = {c.family for c in seq if c is not FALLOW}
    fallow_n = sum(c is FALLOW for c in seq)
    if fallow_n:
        notes.append(f"{fallow_n} bare-fallow season(s) leave soil exposed; consider a cover crop")

    soc_change = (soc - soc_traj[0]) / max(soc_traj[0], 1)
    soil_health = np.clip(
        0.45 * (0.5 + soc_change * 2) +
        0.25 * (0.5 + np.tanh(n_bank / 120) / 2) +
        0.15 * (1 - min(max(pest_traj) / 3, 1)) +
        0.15 * min(len(families) / 4, 1), 0, 1)
    water_eff = 1 - irrigation_total / max(need_total, 1)
    resilience = float(np.mean(suit)) * (0.7 + 0.3 * np.mean(
        [(c.drought_tol + c.heat_tol) / 2 for c in seq]))
    profit = float(np.mean([c.value * s for c, s in zip(seq, suit)])) / 0.85

    return {
        "scores": {"Soil health": float(soil_health), "Water saving": float(water_eff),
                   "Climate resilience": float(resilience), "Profit": float(min(profit, 1))},
        "trajectory": {"soc_g_kg": soc_traj, "n_bank_kg_ha": n_traj, "pest_pressure": pest_traj,
                       "irrigation_mm": irrigation_total, "suitability": suit},
        "notes": list(dict.fromkeys(notes)),
        "min_suit": min(suit),
    }


def weighted(scores: dict, weights: dict) -> float:
    w = np.array([weights.get(k, 0) for k in scores], float)
    w = w / w.sum() if w.sum() else np.ones(len(scores)) / len(scores)
    return float(np.dot(w, list(scores.values())))


# ------------------------------------------------------------------ search
def search_rotations(site: dict, weights: dict, years: int = 3, irrigation_mm: float = 200,
                     allowed: list[str] | None = None, allow_fallow: bool = True,
                     start_season: str = "warm", beam: int = 150, top_k: int = 5,
                     min_suitability: float = 0.35) -> list[Result]:
    order = ["warm", "cool"] if start_season == "warm" else ["cool", "warm"]
    seasons = [order[i % 2] for i in range(years * 2)]

    # Pre-filter: only crops that can realistically grow here today
    pools = {}
    for s in ("warm", "cool"):
        sc = season_climate(site, s)
        pools[s] = [c for c in crops_for_season(s, allowed)
                    if crop_suitability(c, sc, site["soil"], irrigation_mm)["score"] >= min_suitability]
        # With fewer than two crops a season cannot rotate (the same crop may not follow
        # itself), so the field is allowed to rest instead of returning no plan at all.
        if allow_fallow or len(pools[s]) < 2:
            pools[s].append(FALLOW)

    beams: list[list[Crop]] = [[]]
    for i, season in enumerate(seasons):
        candidates = []
        for seq in beams:
            for c in pools[season]:
                new = seq + [c]
                # hard rules: no same crop 2 years running in the same season
                if len(new) >= 3 and new[-1] is new[-3] and c is not FALLOW:
                    continue
                sim = simulate(new, seasons[:len(new)], site, irrigation_mm)
                candidates.append((weighted(sim["scores"], weights), new))
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = [seq for _, seq in candidates[:beam]]

    results, seen = [], set()
    for seq in beams:
        key = tuple(c.name for c in seq)
        if key in seen:
            continue
        seen.add(key)
        now = simulate(seq, seasons, site, irrigation_mm)
        fut = simulate(seq, seasons, site, irrigation_mm, future=True)
        total, ftotal = weighted(now["scores"], weights), weighted(fut["scores"], weights)
        r = Result([c.name for c in seq], seasons, now["scores"], total, ftotal,
                   ftotal / total if total else 0, now["trajectory"], now["notes"])
        drop = (1 - r.robustness) * 100
        r.explanations.append(
            f"Under projected 2041-2050 climate this plan keeps {r.robustness:.0%} of its score"
            + (" (climate-robust)." if drop < 5 else f" ({drop:.0f}% drop - check heat/water-sensitive crops)."))
        results.append(r)
        if len(results) == top_k:
            break
    # final ranking blends today's score with future robustness
    results.sort(key=lambda r: 0.7 * r.total + 0.3 * r.future_total, reverse=True)
    return results
