"""
crop_registry.py - parameters for the 18 crops of the virtual farm.

Sources for the default values (Phase 1, simplified model):
- Stage lengths and crop coefficients: FAO-56 (Allen et al. 1998), Tables 11-12,
  adjusted to typical Punjab (Pakistan) sowing calendars.
- Rooting depth and depletion fraction p: FAO-56 Table 22.
- Normalised water productivity (WP*): AquaCrop reference values - about 33.7 g/m2
  for C4 cereals (maize, sorghum; 30 for pearl millet), 15-19 for C3 crops.
- Harvest index and flowering heat thresholds: typical literature values.

These are starting values for a demonstration, not calibrated local parameters.
Fodder and green-manure crops use harvest index 1.0: their "yield" is dry biomass.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class CropParams:
    key: str
    name: str
    sow_month: int
    sow_day: int
    stages: tuple[int, int, int, int]   # initial, development, mid-season, late (days)
    kc_ini: float
    kc_mid: float
    kc_end: float
    root_max_m: float                    # maximum effective rooting depth
    p: float                             # depletion fraction before water stress
    cc_max: float                        # maximum canopy cover (0-1)
    wp_star: float                       # normalised water productivity, g/m2
    harvest_index: float
    t_base: float                        # base temperature for growth, deg C
    t_heat: float                        # Tmax where flowering heat damage starts
    t_heat_max: float                    # Tmax of full heat damage
    flowering_sensitivity: float         # 0-1, how much flowering stress cuts HI
    flowering_days: int                  # length of the flowering window from mid-season start
    waterlog_tolerant: bool
    flooded: bool                        # grown in standing water (rice)
    height_m: float                      # mature height, used by the 3D view
    yield_label: str                     # what the yield figure refers to
    fresh_factor: float = 1.0            # dry-to-fresh weight for the reported yield (potato)

    @property
    def season_days(self) -> int:
        return sum(self.stages)

    @property
    def season_group(self) -> str:
        """Kharif crops are sown May-September, rabi crops October-February."""
        return "kharif" if 5 <= self.sow_month <= 9 else "rabi"


C = CropParams
_ALL = [
    # ---------------- kharif (summer) ----------------
    C("cotton", "Cotton", 5, 1, (30, 50, 60, 55), 0.35, 1.18, 0.60, 1.4, 0.65,
      0.85, 15.0, 0.32, 12, 38, 45, 0.30, 50, False, False, 1.3, "seed cotton"),
    C("rice", "Rice", 7, 1, (20, 30, 40, 30), 1.05, 1.20, 0.90, 0.5, 0.20,
      0.95, 19.0, 0.43, 8, 35, 41, 0.50, 15, True, True, 1.0, "paddy"),
    C("maize", "Maize", 7, 1, (20, 35, 40, 25), 0.30, 1.20, 0.35, 1.2, 0.55,
      0.96, 33.7, 0.48, 8, 35, 42, 0.55, 15, False, False, 2.2, "grain"),
    C("sorghum", "Sorghum", 7, 1, (20, 35, 35, 20), 0.30, 1.10, 0.55, 1.5, 0.55,
      0.90, 33.7, 0.35, 8, 38, 45, 0.45, 15, False, False, 2.0, "grain"),
    C("pearl_millet", "Pearl millet", 7, 10, (15, 25, 30, 20), 0.30, 1.00, 0.30, 1.5, 0.55,
      0.85, 30.0, 0.25, 10, 40, 46, 0.40, 12, False, False, 2.0, "grain"),
    C("mung_bean", "Mung bean", 7, 1, (15, 20, 20, 15), 0.40, 1.05, 0.60, 0.9, 0.45,
      0.85, 15.0, 0.35, 10, 38, 44, 0.50, 15, False, False, 0.5, "grain"),
    C("soybean", "Soybean", 7, 1, (15, 30, 40, 25), 0.40, 1.15, 0.50, 1.0, 0.50,
      0.95, 15.0, 0.40, 8, 35, 41, 0.45, 20, False, False, 0.9, "grain"),
    C("guar", "Guar", 7, 1, (20, 30, 35, 25), 0.35, 1.00, 0.60, 1.2, 0.60,
      0.80, 15.0, 0.30, 12, 40, 46, 0.35, 20, False, False, 1.0, "seed"),
    C("sunflower", "Sunflower", 7, 15, (20, 30, 30, 20), 0.35, 1.00, 0.35, 1.2, 0.45,
      0.90, 18.0, 0.35, 6, 35, 42, 0.50, 12, False, False, 1.8, "seed"),
    C("sesbania", "Sesbania", 5, 15, (15, 20, 15, 10), 0.40, 1.05, 1.00, 1.2, 0.55,
      0.95, 17.0, 1.00, 10, 45, 50, 0.00, 1, True, False, 2.0, "green manure"),
    # ---------------- rabi (winter) ----------------
    # Punjab wheat sown in November heads in late February and flowers in early March.
    C("wheat", "Wheat", 11, 10, (30, 75, 40, 25), 0.30, 1.15, 0.25, 1.5, 0.55,
      0.95, 15.0, 0.48, 0, 30, 36, 0.50, 20, False, False, 0.9, "grain"),
    C("barley", "Barley", 11, 1, (25, 55, 40, 30), 0.30, 1.15, 0.25, 1.3, 0.55,
      0.90, 15.0, 0.42, 0, 30, 36, 0.45, 20, False, False, 0.8, "grain"),
    C("oats", "Oats", 10, 25, (20, 40, 60, 30), 0.30, 1.15, 0.25, 1.0, 0.55,
      0.90, 15.0, 1.00, 0, 30, 36, 0.00, 1, False, False, 1.0, "fodder"),
    C("mustard", "Mustard", 10, 15, (25, 45, 50, 35), 0.35, 1.00, 0.35, 1.3, 0.60,
      0.85, 15.0, 0.28, 3, 30, 36, 0.45, 25, False, False, 1.4, "seed"),
    C("chickpea", "Chickpea", 10, 15, (25, 50, 50, 35), 0.40, 1.00, 0.35, 1.0, 0.50,
      0.80, 15.0, 0.40, 3, 32, 38, 0.50, 25, False, False, 0.5, "grain"),
    C("lentil", "Lentil", 11, 1, (20, 50, 45, 35), 0.40, 1.10, 0.30, 0.8, 0.50,
      0.80, 15.0, 0.35, 3, 30, 36, 0.45, 20, False, False, 0.4, "grain"),
    C("berseem", "Berseem clover", 10, 15, (30, 50, 80, 30), 0.40, 1.00, 0.90, 0.6, 0.50,
      0.95, 15.0, 1.00, 3, 32, 38, 0.00, 1, True, False, 0.5, "fodder"),
    # Potato: harvest index on dry tuber weight, reported as fresh tubers (about 20 % dry matter)
    C("potato", "Potato", 10, 15, (25, 30, 35, 20), 0.50, 1.15, 0.75, 0.5, 0.35,
      0.90, 18.0, 0.70, 2, 28, 34, 0.40, 20, False, False, 0.5, "fresh tubers", 4.0),
]
CROPS: dict[str, CropParams] = {c.key: c for c in _ALL}

STAGE_NAMES = ("initial", "development", "mid", "late")

# Calibration: the simplified model over-predicts absolute yields, most of all for
# pulses. Each crop's yield is scaled so that a normal season with full irrigation at
# Faisalabad (2025-26 NASA POWER weather) gives a typical good Punjab yield, in t/ha:
# wheat 3.5 (~35 maunds/acre), cotton 2.2, rice paddy 4.0, maize 6.5, barley 2.5,
# sorghum 1.5, pearl millet 1.5, mung 1.0, soybean 1.5, guar 1.0, sunflower 2.0,
# mustard 1.3, chickpea 1.2, lentil 0.9, potato 25, oats 10 and berseem 12 (dry
# fodder), sesbania 6 (green manure). Other places keep the model's relative
# differences, and weather effects (percent changes) are unchanged.
TYPICAL_T_HA = {"wheat": 3.5, "cotton": 2.2, "rice": 4.0, "maize": 6.5, "barley": 2.5,
                "sorghum": 1.5, "pearl_millet": 1.5, "mung_bean": 1.0, "soybean": 1.5,
                "guar": 1.0, "sunflower": 2.0, "mustard": 1.3, "chickpea": 1.2, "lentil": 0.9,
                "potato": 25.0, "oats": 10.0, "berseem": 12.0, "sesbania": 6.0}
_MODEL_POTENTIAL_T_HA = {"wheat": 4.46, "cotton": 5.43, "rice": 6.01, "maize": 12.69, "barley": 5.52,
                         "sorghum": 7.74, "pearl_millet": 3.92, "mung_bean": 1.98, "soybean": 4.24,
                         "guar": 2.65, "sunflower": 3.65, "mustard": 3.74, "chickpea": 5.18, "lentil": 4.29,
                         "potato": 31.65, "oats": 14.73, "berseem": 19.32, "sesbania": 5.74}
YIELD_SCALE = {k: TYPICAL_T_HA[k] / _MODEL_POTENTIAL_T_HA[k] for k in TYPICAL_T_HA}
