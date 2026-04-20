# SOLAR-PAMPA Configuration
# La Pampa, Argentina solar forecasting pipeline
# Coordinates: lat=-37.5, lon=-66.5
# Timezone: America/Argentina/Buenos_Aires

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
ROOT_DIR       = Path(__file__).parent
DATA_RAW       = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODELS_DIR     = ROOT_DIR / "models"
REPORTS_DIR    = ROOT_DIR / "reports"
FIGURES_DIR    = REPORTS_DIR / "figures"

# --- Location ---
LATITUDE      = -37.5
LONGITUDE     = -66.5
TIMEZONE      = "America/Argentina/Buenos_Aires"
LOCATION_NAME = "La Pampa, Argentina"

# --- Date Range ---
START_DATE = "2018-01-01"
END_DATE   = "2023-12-31"

# --- Forecast ---
FORECAST_HORIZON_WEEKS = 4
TARGET_COLUMN          = "ghi_weekly_kwh"

# --- API Keys (loaded from .env) ---
NSRDB_API_KEY       = os.getenv("NSRDB_API_KEY", "")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# --- Model ---
RANDOM_SEED = 42
TEST_SIZE   = 0.2