# Aap ka Khet · آپ کا کھیت — API

**Crop-rotation planning for Pakistani smallholders, built on NASA Earth observations.**
NASA Space Apps Challenge 2026 · Challenge: **Field Shift: Adapting Farms with NASA Data** · Team **Shaheen X**

**Live app:** https://aapkakhet.readhaven.online · **Frontend repo:** [nasa_challenge_front](https://github.com/Aqeel0317/nasa_challenge_front)

This repository is the API behind the app. It reads NASA and other open Earth-observation data for a
farmer's own location, grows their crops day by day in a crop model, tests what a drought, heatwave or
heavy rain would do, and compares the farmer's current rotation with genuinely different alternatives —
each explained with its evidence, benefits, information gaps and risks.

---

## Why it matters

Pakistan's 2022 floods affected about 33 million people and destroyed about 4.4 million acres of crops.
Most smallholders choose next season's crops without seeing the weather risks ahead. Aap ka Khet turns
satellite data into plain answers a farmer can act on: *what to do today*, *what a disaster would cost in
rupees*, and *which rotation holds up better* — in Urdu or English.

## NASA and open data used

| Dataset | What the app uses it for | How it is read |
|---|---|---|
| **NASA POWER** (daily + climatology) | Daily heat, rain and root-zone wetness at the field; 10-year daily normals; very-hot-day counts; the running season | [POWER API](https://power.larc.nasa.gov/) |
| **NASA MODIS NDVI** (16-day) | How green the farmland around the field has been for two years; crop cycles per year; bare-soil time | NASA [GIBS](https://nasa-gibs.github.io/gibs-api-docs/) imagery, decoded with NASA's published colour table (ORNL DAAC MOD13Q1 as backup) |
| **NASA SMAP L4** root-zone soil moisture | Latest soil moisture (m³/m³) at the farm, shown on Today and in every plan's evidence | NASA GIBS imagery ([product](https://nsidc.org/data/spl4smgp)) |
| **NASA EONET** | Storms, floods and fires happening now near the farm | [EONET API](https://eonet.gsfc.nasa.gov/) |
| **ISRIC SoilGrids 2.0** | Soil texture, organic carbon, pH, nitrogen (0–30 cm) | [SoilGrids API](https://soilgrids.org/) |
| **Open-Meteo forecast** | The next 16 days of weather for the running season | [Open-Meteo](https://open-meteo.com/en/docs) |
| **Open-Meteo Climate API (CMIP6)** | Temperature and rain change by 2041–2050 | [Climate API](https://open-meteo.com/en/docs/climate-api) |

Every value shown to farmers carries its source and date. Anything the model computes itself (crop
growth, yields, field soil water) is labelled as a model estimate, and if a service does not answer the
app says so and uses typical values instead of failing.

**Town centres are handled.** A district picked from a list is its town centre, where a single 250 m
pixel is roofs and roads. Greenness is read over a 22 × 22 km box (its greener quarter is the farmland),
and SoilGrids is retried about 11 km out when the centre has no soil.

## How it works

```
farmer's place, crops, water ─┐
                              ├─► weather for the farm year ─► daily crop model ─► yields, soil water, stress
NASA POWER + forecast ────────┘        (observed → forecast → 10-yr normals)          │
                                                                                      ├─► disaster tests (drought, heatwave, heavy rain)
SoilGrids, MODIS, SMAP, CMIP6 ─► site profile ─► rotation search + scoring ───────────┴─► plans with evidence
```

- **Running season.** The current May–April farm year is built from NASA POWER observations to the
  latest published day, then the Open-Meteo forecast, then 10-year NASA daily normals (with realistic
  warm/cool spells and rain showers, so normals are not "perfect weather").
- **Crop model.** A simplified AquaCrop-style daily model: canopy, root-zone water balance with
  Hargreaves ET₀, water and heat stress at flowering, and waterlogging that damages the crop (roots
  short of air, worst at flowering). Yields are calibrated to typical Punjab yields.
- **Disasters never pay.** Drought, heatwave and heavy-rain tests are capped at the normal-year yield,
  and the response reports how many days the soil stayed waterlogged.
- **Plans.** A beam search over 3-year warm/cool sequences, plus the rotations Punjab farmers actually
  grow, scored on soil health, water saving, climate resilience and income. The farmer's **current
  rotation** is scored the same way and returned first; five **different strategies** follow (best
  overall, least water, best for soil, safest in a hotter future, best earning), with near-duplicates
  removed. Farm size, soil type, how the farmer sells, their own mandi prices and their equipment
  (tractor, laser leveller, tubewell) all feed the scoring.
- **Every plan explains itself** (English and Urdu): *Why suggested · Evidence (dataset, value, date,
  link) · Benefits · Information gaps · Risks to check with a local agriculture officer.*
- **Same inputs, same numbers.** Data that changes daily (forecast, latest observations) is fetched once
  per day and shared by all server workers, so a reload never changes the answer within a day.

## API

Interactive docs at `/docs` when running. Main endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/crops` | Crops the planner knows |
| `GET` | `/api/site?lat=&lon=` | Site profile: climate, extremes, soil, NDVI, future climate, sources |
| `POST` | `/api/rotations` | Plans: current rotation + diverse alternatives, each with a report |
| `GET` | `/api/farm/crops` | Crops the farm simulator knows (kharif / rabi) |
| `POST` | `/api/farm/simulate` | Up to 4 fields through a farm year, with every disaster test |
| `GET` | `/api/farm/live?lat=&lon=` | Last 14 days of NASA POWER, NASA SMAP soil moisture, EONET events nearby |

Example:

```bash
curl -X POST https://aapkakhet.readhaven.online/api/rotations \
  -H "Content-Type: application/json" \
  -d '{"lat":31.42,"lon":73.08,"years":3,"irrigation_mm":0,
       "weights":{"soil_health":9,"water_saving":10,"climate_resilience":7,"profit":4},
       "current_rotation":["Cotton","Wheat"]}'
```

## Run locally

Requires Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# http://localhost:8000/docs
```

No API keys are needed: every data source is open. Responses are cached on disk (`.cache/` locally,
`/home/cache` on Azure).

## Tests

```bash
pytest -q
```

13 tests, including regression tests that guard the app's promises:

- the same inputs give identical numbers;
- heavy rain is never a windfall, at any water level;
- plans include the farmer's current rotation, no two plans are near-duplicates, and every plan has all
  five report sections in English and Urdu.

Tests run on every push in GitHub Actions before deployment.

## Deployment

Azure App Service (Linux, Python 3.12, free F1 tier), started by `startup.sh` (gunicorn + uvicorn
workers). The GitHub Actions workflow in `.github/workflows/azure-deploy.yml` runs the tests and then
deploys. The frontend's Cloudflare Worker proxies `/api/*` here and pings `/api/health` every 10 minutes
so the free tier stays warm.

## Project layout

```
app/
  main.py, config.py         FastAPI app, settings
  routes.py                  /api: crops, site, rotations
  routes_farm.py             /api/farm: simulate, live
  schemas.py                 request / response models
  services/
    data_sources.py          NASA POWER, MODIS (GIBS), SMAP, SoilGrids, CMIP6; disk cache
    engine.py                seasonal suitability, rotation search and scoring
    plans.py                 current rotation, diverse strategies, evidence reports (EN + UR)
    crops.py                 planner crop parameters
  simulation/
    engine.py                daily crop model
    weather.py               farm-year weather: observed → forecast → normals
    scenarios.py             drought, heatwave, heavy rain, custom
    crop_registry.py         simulator crops and yield calibration
tests/test_api.py
```

## Limitations

- A simplified seasonal model: it compares options, it does not predict an exact harvest.
- Soil comes from a 250 m map unless the farmer enters a soil test.
- Future climate comes from one CMIP6 model (MRI-AGCM3-2-S).
- Income uses typical 2025 mandi prices unless the farmer enters their own.
- The app supports the farmer's judgement; it does not replace their local agriculture officer.

## Team

**Shaheen X** — NASA Space Apps Challenge 2026.

## Use of AI

The code was written with the help of an AI coding assistant. The demo video's voice-over and the
app's spoken guides are AI-generated speech (Deepgram, Azure AI Speech), labelled as such. All data
comes from the sources listed above; no data is AI-generated.

## Acknowledgements

NASA POWER, NASA GIBS, NASA MODIS (LP DAAC / ORNL DAAC), NASA SMAP (NSIDC), NASA EONET, ISRIC
SoilGrids, and Open-Meteo, for open data that makes this possible.
