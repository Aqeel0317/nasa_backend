"""
plans.py - turn the rotation search into a short list of genuinely different
strategies, put the farmer's own rotation next to them (scored by the same model),
and explain each plan: why it is suggested, the evidence (dataset, value, date),
benefits, information gaps and risks to check with a local expert.

Every number in a report comes from the model run or from the site data; anything
that is a fallback or a simplification is said so in "gaps".
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from . import engine as en
from .crops import CROPS, FALLOW

MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July", "August",
             "September", "October", "November", "December"]
MONTHS_UR = ["جنوری", "فروری", "مارچ", "اپریل", "مئی", "جون", "جولائی", "اگست", "ستمبر",
             "اکتوبر", "نومبر", "دسمبر"]

LINKS = {
    "NASA POWER": "https://power.larc.nasa.gov/",
    "NASA MODIS NDVI (GIBS)": "https://nasa-gibs.github.io/gibs-api-docs/available-visualizations/",
    "NASA MODIS MOD13Q1": "https://lpdaac.usgs.gov/products/mod13q1v061/",
    "ISRIC SoilGrids": "https://soilgrids.org/",
    "Open-Meteo CMIP6": "https://open-meteo.com/en/docs/climate-api",
    "Open-Meteo forecast": "https://open-meteo.com/en/docs",
    "NASA EONET": "https://eonet.gsfc.nasa.gov/",
    "NASA SMAP L4": "https://nsidc.org/data/spl4smgp",
}

# Rotations Punjab farmers actually grow: always considered, so the comparison
# includes familiar options (wheat, cotton, rice) and not only what the search found.
COMMON = [("Cotton", "Wheat"), ("Rice", "Wheat"), ("Maize", "Wheat"), ("Mung bean", "Wheat"),
          ("Maize", "Potato"), ("Sorghum", "Berseem clover"), ("Guar", "Chickpea"), ("Cotton", "Chickpea")]

STRATEGIES = [
    # key, how it is ranked, label en, label ur
    ("overall", None, "Best overall for your priorities", "آپ کی ترجیحات کے لیے سب سے بہتر"),
    ("income", "Profit", "Most income", "سب سے زیادہ آمدنی"),
    ("soil", "Soil health", "Best for your soil", "آپ کی مٹی کے لیے بہترین"),
    ("water", "Water saving", "Least water", "سب سے کم پانی"),
    ("climate", "Climate resilience", "Safest in a hotter future", "گرم مستقبل میں سب سے محفوظ"),
]


def _blend(r) -> float:
    return 0.7 * r.total + 0.3 * r.future_total


def _similar(a: list[str], b: list[str]) -> bool:
    """Near-duplicates: half or more of the seasons hold the same crop, or the
    plans use exactly the same crops in another order."""
    same_pos = sum(x == y for x, y in zip(a, b)) / max(1, min(len(a), len(b)))
    return same_pos >= 0.5 or Counter(a) == Counter(b)


def _evaluate(names: list[str], seasons: list[str], site, weights, irrigation_mm):
    seq = [CROPS.get(n, FALLOW) for n in names]
    now = en.simulate(seq, seasons, site, irrigation_mm)
    fut = en.simulate(seq, seasons, site, irrigation_mm, future=True)
    total, ftotal = en.weighted(now["scores"], weights), en.weighted(fut["scores"], weights)
    r = en.Result(names, seasons, now["scores"], total, ftotal, ftotal / total if total else 0,
                  now["trajectory"], now["notes"])
    return r


def build_plans(site: dict, weights: dict, years: int, irrigation_mm: float, allowed, allow_fallow: bool,
                start_season: str, top_k: int, current: list[str] | None):
    order = ["warm", "cool"] if start_season == "warm" else ["cool", "warm"]
    seasons = [order[i % 2] for i in range(years * 2)]

    # 1. candidates: the beam search (wide), plus the common local rotations
    pool = en.search_rotations(site, weights, years, irrigation_mm, allowed, allow_fallow,
                               start_season, top_k=60)
    seen = {tuple(r.sequence) for r in pool}
    for warm, cool in COMMON:
        if allowed and not {warm, cool} <= set(allowed):
            continue
        pair = (warm, cool) if start_season == "warm" else (cool, warm)
        names = list(pair) * years
        if tuple(names) not in seen:
            seen.add(tuple(names))
            pool.append(_evaluate(names, seasons, site, weights, irrigation_mm))

    # 2. one plan per strategy, skipping near-duplicates of plans already chosen
    chosen: list[tuple[str, object]] = []
    for key, dim, _en, _ur in STRATEGIES:
        ranked = sorted(pool, key=(lambda r: _blend(r)) if dim is None else (lambda r, d=dim: (r.scores[d], _blend(r))),
                        reverse=True)
        pick = next((r for r in ranked if all(not _similar(r.sequence, c.sequence) for _, c in chosen)), None)
        if pick is not None:
            chosen.append((key, pick))
        if len(chosen) >= top_k:
            break
    for r in sorted(pool, key=_blend, reverse=True):          # fill up if fewer strategies were possible
        if len(chosen) >= top_k:
            break
        if all(not _similar(r.sequence, c.sequence) for _, c in chosen):
            chosen.append(("overall", r))

    for _, r in chosen:
        drop = (1 - r.robustness) * 100
        line = (f"Under projected 2041-2050 climate this plan keeps {r.robustness:.0%} of its score"
                + (" (climate-robust)." if drop < 5 else f" ({drop:.0f}% drop - check heat/water-sensitive crops)."))
        if line not in r.explanations:
            r.explanations.append(line)

    # 3. the farmer's own rotation, scored the same way
    cur = None
    if current:
        names = [n if n in CROPS else "Fallow" for n in current][:2]
        if len(names) == 1:
            names.append("Fallow")
        pair = names if start_season == "warm" else names[::-1]
        cur = _evaluate(list(pair) * years, seasons, site, weights, irrigation_mm)

    chosen.sort(key=lambda kr: _blend(kr[1]), reverse=True)
    return chosen, cur


# ------------------------------------------------------------------ reports
def _t(en_text: str, ur_text: str, source: str | None = None, url: str | None = None, date: str | None = None) -> dict:
    item = {"en": en_text, "ur": ur_text}
    if source:
        item.update(source=source, url=url, date=date)
    return item


def _crop_risks(names, seasons, site, irrigation_mm):
    out = []
    for name, season in dict.fromkeys(zip(names, seasons)):
        crop = CROPS.get(name)
        if not crop:
            continue
        s = en.crop_suitability(crop, en.season_climate(site, season), site["soil"], irrigation_mm)
        f = en.crop_suitability(crop, en.season_climate(site, season, future=True), site["soil"], irrigation_mm)
        out.append((name, s["score"], f["score"], s["notes"]))
    return out


def plan_report(r, site: dict, irrigation_mm: float, strategy: str | None, current=None, is_current: bool = False) -> dict:
    clim, src = site["climate"], site.get("sources", {})
    traj = r.trajectory
    hot = int(np.argmax(clim["t_max"]))
    rain_year = float(np.sum(clim["precip_mm"]))
    ndvi, fut, soil = site.get("ndvi") or {}, site.get("future") or {}, site["soil"]
    series = ndvi.get("series") or []
    crops_here = sorted({n for n in r.sequence if n in CROPS})
    risks_by_crop = _crop_risks(r.sequence, r.seasons, site, irrigation_mm)

    # ---- why
    why = []
    if is_current:
        why.append(_t("This is the rotation you grow now, scored by the same model so you can compare.",
                      "یہ وہ فصلی چکر ہے جو آپ ابھی اگاتے ہیں، اسی ماڈل سے جانچا گیا تاکہ آپ موازنہ کر سکیں۔"))
    else:
        label = next((s for s in STRATEGIES if s[0] == strategy), STRATEGIES[0])
        why.append(_t(f"Chosen as: {label[2]}.", f"اس لیے چنا گیا: {label[3]}۔"))
        best_dim = max(r.scores, key=r.scores.get)
        names = {"Soil health": ("soil health", "مٹی کی صحت"), "Water saving": ("water saving", "پانی کی بچت"),
                 "Climate resilience": ("climate resilience", "موسمی مضبوطی"), "Profit": ("income", "آمدنی")}
        why.append(_t(f"Its strongest side is {names[best_dim][0]} ({r.scores[best_dim] * 100:.0f} out of 100).",
                      f"اس کی سب سے مضبوط بات {names[best_dim][1]} ہے (100 میں سے {r.scores[best_dim] * 100:.0f})۔"))
    for note in r.explanations[:2]:
        if "supplies nitrogen" in note:
            why.append(_t(note + ".", "دال والی فصل اگلی فصل کو نائٹروجن دیتی ہے۔"))

    # ---- evidence: dataset, value, date
    evidence = [
        _t(f"Typical daily high in {MONTHS_EN[hot]}: {clim['t_max'][hot]:.0f} °C; rain {rain_year:.0f} mm a year.",
           f"{MONTHS_UR[hot]} میں عام دن کا زیادہ درجہ حرارت {clim['t_max'][hot]:.0f} °C؛ سال میں {rain_year:.0f} ملی میٹر بارش۔",
           src.get("Climate normals", "NASA POWER climatology"), LINKS["NASA POWER"], "long-term monthly normals"),
    ]
    if series:
        evidence.append(_t(
            f"Satellite greenness (NDVI): about {ndvi.get('crop_cycles_per_year', 0):g} crops a year here, land bare {ndvi.get('bare_soil_fraction', 0) * 100:.0f}% of the time.",
            f"سیٹلائٹ سبزہ (NDVI): یہاں سال میں تقریباً {ndvi.get('crop_cycles_per_year', 0):g} فصلیں، زمین {ndvi.get('bare_soil_fraction', 0) * 100:.0f}% وقت خالی۔",
            src.get("Vegetation history", "NASA MODIS"), LINKS["NASA MODIS NDVI (GIBS)"], f"{series[0][0]} to {series[-1][0]}"))
    smap = site.get("smap")
    if smap:
        evidence.append(_t(
            f"Root-zone soil moisture now: {smap['value']:.2f} m³/m³ (how much water the top metre of soil holds).",
            f"اس وقت جڑوں کی مٹی کی نمی: {smap['value']:.2f} m³/m³ (اوپر کے ایک میٹر میں کتنا پانی ہے)۔",
            "NASA SMAP L4", smap.get("url"), smap.get("date")))
    evidence.append(_t(
        f"Soil: organic carbon {soil.get('soc', 0):.1f} g/kg, pH {soil.get('phh2o', 0):.1f}, clay {soil.get('clay', 0):.0f}%.",
        f"مٹی: نامیاتی کاربن {soil.get('soc', 0):.1f} گرام/کلو، pH {soil.get('phh2o', 0):.1f}، چکنی مٹی {soil.get('clay', 0):.0f}%۔",
        src.get("Soil", "ISRIC SoilGrids"), LINKS["ISRIC SoilGrids"], "SoilGrids 2.0 (0-30 cm)"))
    if fut:
        evidence.append(_t(
            f"Climate by 2041-2050: {fut.get('delta_t', 0):+.1f} °C and {fut.get('delta_p_pct', 0):+.0f}% rain compared with 2001-2010.",
            f"2041-2050 تک موسم: 2001-2010 کے مقابلے میں {fut.get('delta_t', 0):+.1f} °C اور بارش {fut.get('delta_p_pct', 0):+.0f}%۔",
            src.get("Future climate", "Open-Meteo CMIP6"), LINKS["Open-Meteo CMIP6"], "CMIP6 HighResMIP"))
    for name, now, later, _ in risks_by_crop:
        evidence.append(_t(f"{name}: crop fit {now * 100:.0f}/100 today, {later * 100:.0f}/100 by 2041-2050.",
                           f"{name}: آج موزونیت {now * 100:.0f}/100، 2041-2050 تک {later * 100:.0f}/100۔",
                           "Aap ka Khet crop model", None, "this run"))

    # ---- benefits (and the costs go to risks), against the farmer's own rotation
    benefits, risks = [], []
    soc0, soc1 = traj["soc_g_kg"][0], traj["soc_g_kg"][-1]
    if soc1 > soc0:
        benefits.append(_t(f"Builds soil organic carbon: {soc0:.1f} → {soc1:.1f} g/kg over the plan.",
                           f"مٹی کا نامیاتی کاربن بڑھتا ہے: {soc0:.1f} سے {soc1:.1f} گرام/کلو۔"))
    else:
        risks.append(_t(f"Soil organic carbon falls: {soc0:.1f} → {soc1:.1f} g/kg. Add residues, manure or a green-manure crop.",
                        f"مٹی کا نامیاتی کاربن کم ہوتا ہے: {soc0:.1f} سے {soc1:.1f} گرام/کلو۔ فصل کی باقیات، کھاد یا سبز کھاد شامل کریں۔"))
    n_end = traj["n_bank_kg_ha"][-1]
    if n_end > 20:
        benefits.append(_t(f"Leaves about {n_end:.0f} kg/ha nitrogen for the next crop: less urea to buy.",
                           f"اگلی فصل کے لیے تقریباً {n_end:.0f} کلو/ہیکٹر نائٹروجن چھوڑتا ہے: کم یوریا خریدنی پڑے گی۔"))
    if current is not None and not is_current:
        irr_d = traj["irrigation_mm"] - current.trajectory["irrigation_mm"]
        if irr_d < -50:
            benefits.append(_t(f"Needs about {-irr_d:.0f} mm less irrigation than your current rotation over the plan.",
                               f"آپ کے موجودہ چکر سے تقریباً {-irr_d:.0f} ملی میٹر کم آبپاشی۔"))
        elif irr_d > 50:
            risks.append(_t(f"Needs about {irr_d:.0f} mm more irrigation than your current rotation.",
                            f"آپ کے موجودہ چکر سے تقریباً {irr_d:.0f} ملی میٹر زیادہ آبپاشی چاہیے۔"))
        dp = (r.scores["Profit"] - current.scores["Profit"]) * 100
        if dp >= 3:
            benefits.append(_t(f"Income score {dp:.0f} points higher than your current rotation.",
                               f"آمدنی کا اسکور آپ کے موجودہ چکر سے {dp:.0f} پوائنٹ زیادہ۔"))
        elif dp <= -3:
            risks.append(_t(f"Income score {-dp:.0f} points lower than your current rotation: check prices and buyers first.",
                            f"آمدنی کا اسکور آپ کے موجودہ چکر سے {-dp:.0f} پوائنٹ کم: پہلے قیمت اور خریدار دیکھیں۔"))
        dr = (r.robustness - current.robustness) * 100
        if dr >= 3:
            benefits.append(_t(f"Holds up better in a hotter 2041-2050 climate ({r.robustness:.0%} vs {current.robustness:.0%} of its score kept).",
                               f"2041-2050 کے گرم موسم میں بہتر ({r.robustness:.0%} بمقابلہ {current.robustness:.0%})۔"))
    if len({CROPS[n].family for n in crops_here}) >= 3:
        benefits.append(_t("Mixes three or more plant families, which breaks pest and disease cycles.",
                           "تین یا زیادہ پودوں کے خاندان، جس سے کیڑوں اور بیماریوں کا چکر ٹوٹتا ہے۔"))

    for name, now, later, notes in risks_by_crop:
        if now < 0.6:
            reason = "; ".join(notes) if notes else "weak fit for this climate or soil"
            risks.append(_t(f"{name} fits only {now * 100:.0f}/100 here ({reason}).",
                            f"{name} یہاں صرف {now * 100:.0f}/100 موزوں ہے۔"))
        if later < now - 0.08:
            risks.append(_t(f"{name} gets harder to grow by 2041-2050 ({now * 100:.0f} → {later * 100:.0f}/100).",
                            f"{name} 2041-2050 تک اگانا مشکل ہو جائے گا ({now * 100:.0f} سے {later * 100:.0f}/100)۔"))
    for note in r.explanations:
        if "pest carry-over" in note:
            risks.append(_t(note + ".", "ایک ہی خاندان کی فصل دوبارہ: کیڑوں کا خطرہ بڑھتا ہے۔"))
        if "bare-fallow" in note:
            risks.append(_t(note + ".", "خالی زمین کے سیزن مٹی کو کھلا چھوڑتے ہیں؛ کور فصل لگائیں۔"))
    risks.append(_t("Check seed, local prices and buyers with your agriculture officer before changing crops.",
                    "فصل بدلنے سے پہلے بیج، مقامی قیمت اور خریدار اپنے زرعی افسر سے پوچھیں۔"))

    # ---- gaps: what the model does not know
    gaps = []
    for label, value in src.items():
        if "fallback" in str(value) or "demo" in str(value):
            gaps.append(_t(f"{label}: the live data service did not answer, so typical values were used.",
                           f"{label}: ڈیٹا سروس نے جواب نہیں دیا، اس لیے عام اقدار استعمال ہوئیں۔"))
    if "Farmer soil test" not in str(src.get("Soil", "")):
        gaps.append(_t("Soil comes from a 250 m soil map, not a test of your own field. A soil test would make this more accurate.",
                       "مٹی کا ڈیٹا 250 میٹر کے نقشے سے ہے، آپ کے کھیت کے ٹیسٹ سے نہیں۔ مٹی کا ٹیسٹ اسے زیادہ درست بنائے گا۔"))
    gaps.append(_t("Income uses a crop-value index, not today's mandi prices or your buyers.",
                   "آمدنی فصل کی قدر کے اشاریے سے ہے، آج کی منڈی قیمت یا آپ کے خریداروں سے نہیں۔"))
    gaps.append(_t(f"Future climate comes from one climate model ({fut.get('model', 'CMIP6')}); others may differ.",
                   f"مستقبل کا موسم ایک ماڈل ({fut.get('model', 'CMIP6')}) سے ہے؛ دوسرے ماڈل مختلف ہو سکتے ہیں۔"))
    gaps.append(_t("A simplified seasonal model: it compares options, it does not predict your exact harvest.",
                   "ایک سادہ موسمی ماڈل: یہ آپشنز کا موازنہ کرتا ہے، آپ کی اصل پیداوار کی پیش گوئی نہیں۔"))

    return {"why": why, "evidence": evidence, "benefits": benefits or [
        _t("No clear benefit over your current rotation on these measures.", "ان پیمانوں پر آپ کے موجودہ چکر سے کوئی واضح فائدہ نہیں۔")],
            "gaps": gaps, "risks": risks}
