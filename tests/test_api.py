"""Run with:  pytest -q   (works offline: data services fall back to demo values)"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/api/health").json() == {"status": "ok"}


def test_crops():
    crops = client.get("/api/crops").json()
    assert any(c["name"] == "Wheat" for c in crops)


def test_rotations():
    body = {"lat": 41.6, "lon": -93.6, "years": 2, "irrigation_mm": 150,
            "weights": {"soil_health": 8, "water_saving": 5, "climate_resilience": 6, "profit": 4},
            "soil_override": {"phh2o": 6.8}}
    r = client.post("/api/rotations", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data["rotations"]) >= 1
    assert len(data["rotations"][0]["sequence"]) == 4
    assert data["site"]["soil"]["phh2o"] == 6.8


def test_bad_crop():
    r = client.post("/api/rotations", json={"lat": 10, "lon": 10, "allowed_crops": ["Banana"]})
    assert r.status_code == 422


def test_farm_simulation():
    r = client.post("/api/farm/simulate", json={"year": 2024, "scenario": "drought", "irrigation": "rainfed"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data["dates"]) == 365
    assert len(data["fields"]) == 4
    # default fields: cotton/rice/maize/mung then wheat/mustard/chickpea, wheat simulated once
    assert {p["crop"] for p in data["plots"]} == {"cotton", "rice", "maize", "mung_bean", "wheat", "mustard", "chickpea"}
    assert set(data["shocks"]) == {"drought", "heatwave", "flood"}
    # on a rain-fed farm a drought should cut most harvests
    down = sum(p["scenario"]["summary"]["yield_t_ha"] < p["baseline"]["summary"]["yield_t_ha"] for p in data["plots"])
    assert down >= 5
    for p in data["plots"]:
        assert p["insights"][0]["code"] == "yield_change"
        # every season must end inside the May-April farm year
        assert p["harvest_date"] <= "2025-04-30"


def test_farm_custom_fields():
    body = {"year": 2024, "fields": [{"kharif": "rice", "rabi": "potato", "acres": 5}]}
    data = client.post("/api/farm/simulate", json=body).json()
    assert [p["crop"] for p in data["plots"]] == ["rice", "potato"]
    assert data["fields"][0]["acres"] == 5


def test_farm_field_rules():
    too_many = {"fields": [{"kharif": "rice"}] * 5}
    assert client.post("/api/farm/simulate", json=too_many).status_code == 422
    wrong_season = {"year": 2024, "fields": [{"kharif": "wheat"}]}
    assert client.post("/api/farm/simulate", json=wrong_season).status_code == 422

def test_farm_year_not_finished():
    r = client.post("/api/farm/simulate", json={"year": 2030})
    assert r.status_code == 422


def test_live_feed_shape():
    # Works with or without network: the shape is always the same.
    data = client.get("/api/live", params={"lat": 31.42, "lon": 73.08}).json()
    assert {"weather", "events", "fetched_at"} <= set(data)
    assert isinstance(data["weather"]["days"], list)
    assert isinstance(data["events"], list)


def test_farm_current_season():
    # The running season works (observed + forecast + normals, or normals when offline).
    import datetime as dt
    today = dt.date.today()
    year = today.year if today.month >= 5 else today.year - 1
    r = client.post("/api/farm/simulate", json={"year": year, "fields": [{"kharif": "cotton", "rabi": "wheat"}]})
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data["dates"]) in (365, 366)
    assert "timeline" in data["weather"]
