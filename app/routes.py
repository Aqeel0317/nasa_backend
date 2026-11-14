"""API endpoints. Plain `def` (not async) because the work is blocking I/O and
NumPy; FastAPI runs these in a thread pool automatically."""
from fastapi import APIRouter, HTTPException, Query

from .schemas import CropOut, RotationOut, RotationRequest, RotationResponse, SuitabilityRow
from .services import data_sources as ds
from .services import engine as en
from .services import plans
from .services.crops import CROPS

router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/crops", response_model=list[CropOut])
def list_crops() -> list[CropOut]:
    return [CropOut(name=c.name, family=c.family, season=c.season, role=c.role,
                    water_mm=c.water_mm, n_effect=c.n_effect) for c in CROPS.values()]


@router.get("/site")
def get_site(lat: float = Query(..., ge=-60, le=75),
             lon: float = Query(..., ge=-180, le=180)) -> dict:
    """Raw field snapshot: climate, extremes, soil, NDVI history, future climate."""
    site = ds.build_site_data(round(lat, 4), round(lon, 4))
    return {**site, "seasons": en.detect_seasons(site["climate"]["t_mean"]),
            "texture": en.texture_class(site["soil"])}


@router.post("/rotations", response_model=RotationResponse)
def recommend(req: RotationRequest) -> RotationResponse:
    unknown = set(req.allowed_crops or []) - set(CROPS)
    if unknown:
        raise HTTPException(422, f"Unknown crops: {sorted(unknown)}")

    cached = ds.build_site_data(round(req.lat, 4), round(req.lon, 4))
    # copy before changing: the cached dict is shared between requests
    site = {**cached, "soil": dict(cached["soil"]), "sources": dict(cached["sources"])}
    if req.soil_override:
        over = req.soil_override.model_dump(exclude_none=True)
        if over:
            site["soil"].update(over)
            site["sources"]["Soil"] = "Farmer soil test + SoilGrids"
    site["seasons"] = en.detect_seasons(site["climate"]["t_mean"])
    try:   # NASA SMAP root-zone soil moisture, latest day (evidence for every plan)
        site["smap"] = ds.fetch_smap_rootzone(round(req.lat, 2), round(req.lon, 2))
    except Exception:  # noqa: BLE001
        site["smap"] = None
    site["texture"] = en.texture_class(site["soil"])

    allowed = req.allowed_crops or None
    chosen, cur = plans.build_plans(site, req.weights.as_engine(), req.years, req.irrigation_mm,
                                    allowed, req.allow_fallow, req.start_season, req.top_k, req.current_rotation)
    suit = [SuitabilityRow(crop=r["crop"], season=r["season"], role=r["role"],
                           now=r["now"], future=r["2041-2050"])
            for r in en.suitability_table(site, req.irrigation_mm, allowed)]
    def out(r, rank, strategy=None, current=False):
        return RotationOut(rank=rank, sequence=r.sequence, seasons=r.seasons,
                           scores=r.scores, total=r.total, future_total=r.future_total,
                           robustness=r.robustness, trajectory=r.trajectory,
                           explanations=r.explanations, strategy=strategy, current=current,
                           report=plans.plan_report(r, site, req.irrigation_mm, strategy, cur, is_current=current))
    rotations = [out(r, i + 1, key) for i, (key, r) in enumerate(chosen)]
    current = out(cur, 0, None, True) if cur is not None else None
    return RotationResponse(site=site, suitability=suit, rotations=rotations, current=current, links=plans.LINKS)
