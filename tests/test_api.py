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
    assert len(data["plots"]) == 18
    assert {"wheat", "rice", "cotton", "maize", "potato", "berseem"} <= {p["crop"] for p in data["plots"]}
    # on a rain-fed farm a drought should cut most harvests
    down = sum(p["scenario"]["summary"]["yield_t_ha"] < p["baseline"]["summary"]["yield_t_ha"] for p in data["plots"])
    assert down >= 12
    for p in data["plots"]:
        assert p["insights"][0]["code"] == "yield_change"
        # every season must end inside the May-April farm year
        assert p["harvest_date"] <= "2025-04-30"


def test_farm_year_not_finished():
    r = client.post("/api/farm/simulate", json={"year": 2030})
    assert r.status_code == 422
