# DataAgent for SOLAR-PAMPA
# Fetches solar irradiance data from NSRDB API for La Pampa, Argentina
# Falls back to synthetic data if no API key is available
# Outputs: data/processed/solar_raw_validated.parquet

import pandas as pd
import numpy as np
import requests
from pathlib import Path
from datetime import datetime
from agents import BaseAgent
from config import (
    LATITUDE, LONGITUDE, START_DATE, END_DATE,
    NSRDB_API_KEY, DATA_RAW, DATA_PROCESSED
)


class DataAgent(BaseAgent):

    def __init__(self, input_path: Path, output_path: Path):
        super().__init__(
            name        = "DataAgent",
            input_path  = input_path,
            output_path = output_path
        )
        self.output_file = self.output_path / "solar_raw_validated.parquet"

    def validate_inputs(self) -> bool:
        # DataAgent has no input files — it fetches from API
        # Just confirm output directory exists
        self.logger.info(f"Output directory: {self.output_path}")
        return True

    def run(self) -> bool:
        if NSRDB_API_KEY:
            self.logger.info("NSRDB API key found — fetching real data...")
            df = self._fetch_nsrdb()
        else:
            self.logger.warning("No NSRDB API key — generating synthetic data for development...")
            df = self._generate_synthetic_data()

        if df is None or df.empty:
            self.logger.error("No data was fetched or generated.")
            return False

        df = self._validate_schema(df)
        df.to_parquet(self.output_file, index=False)
        self.logger.info(f"Saved {len(df)} rows to {self.output_file}")
        self.logger.info(f"Columns: {list(df.columns)}")
        self.logger.info(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
        return True

    # ------------------------------------------------------------------
    # NSRDB Fetch
    # ------------------------------------------------------------------
    def _fetch_nsrdb(self) -> pd.DataFrame:
        """Fetch hourly solar data from NSRDB API."""
        base_url = "https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv"
        params = {
            "api_key"   : NSRDB_API_KEY,
            "lat"       : LATITUDE,
            "lon"       : LONGITUDE,
            "year"      : "2022",
            "interval"  : "60",
            "attributes": "ghi,dni,dhi,air_temperature,wind_speed,cloud_type",
            "name"      : "SOLAR-PAMPA",
            "email"     : "user@solarpampa.com",
            "utc"       : "false",
        }
        try:
            self.logger.info(f"Requesting NSRDB data for lat={LATITUDE}, lon={LONGITUDE}...")
            response = requests.get(base_url, params=params, timeout=60)
            response.raise_for_status()

            from io import StringIO
            # NSRDB CSV has 2 header rows — skip them
            df = pd.read_csv(StringIO(response.text), skiprows=2)
            df = self._parse_nsrdb_columns(df)
            return df

        except Exception as e:
            self.logger.error(f"NSRDB fetch failed: {e}")
            self.logger.warning("Falling back to synthetic data...")
            return self._generate_synthetic_data()

    def _parse_nsrdb_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse NSRDB raw CSV into standard schema."""
        df["timestamp"] = pd.to_datetime(
            df[["Year", "Month", "Day", "Hour", "Minute"]]
            .rename(columns={"Year":"year","Month":"month",
                              "Day":"day","Hour":"hour","Minute":"minute"})
        )
        rename_map = {
            "GHI"            : "ghi",
            "DNI"            : "dni",
            "DHI"            : "dhi",
            "Temperature"    : "temp_air",
            "Wind Speed"     : "wind_speed",
            "Cloud Type"     : "cloud_type",
        }
        df = df.rename(columns=rename_map)
        return df[["timestamp", "ghi", "dni", "dhi",
                   "temp_air", "wind_speed", "cloud_type"]]

    # ------------------------------------------------------------------
    # Synthetic Data (for development without API key)
    # ------------------------------------------------------------------
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """
        Generate realistic synthetic solar data for La Pampa.
        Uses seasonal patterns and random noise to simulate real conditions.
        """
        self.logger.info(f"Generating synthetic data: {START_DATE} → {END_DATE}")

        timestamps = pd.date_range(
            start=START_DATE,
            end=END_DATE,
            freq="h"
        )

        n          = len(timestamps)
        day_of_year = timestamps.day_of_year.values
        hour        = timestamps.hour.values

        # Solar angle approximation for La Pampa (Southern Hemisphere)
        declination  = 23.45 * np.sin(np.radians(360 / 365 * (day_of_year - 81)))
        solar_noon   = np.cos(np.radians(hour * 15 - 180))
        solar_factor = np.maximum(0, solar_noon)

        # Seasonal GHI (Southern Hemisphere: peak in Dec-Jan)
        seasonal     = 1 + 0.3 * np.cos(np.radians(360 / 365 * (day_of_year - 355)))
        ghi_clear    = 900 * seasonal * solar_factor

        # Cloud cover effect
        cloud_cover  = np.random.beta(2, 5, size=n)
        ghi          = ghi_clear * (1 - 0.75 * cloud_cover)
        ghi          = np.maximum(0, ghi + np.random.normal(0, 20, n))

        # Temperature: seasonal variation for La Pampa
        temp_air     = (
            18
            + 10 * np.cos(np.radians(360 / 365 * (day_of_year - 355)))
            + 5  * np.sin(np.radians(hour * 15))
            + np.random.normal(0, 2, n)
        )

        df = pd.DataFrame({
            "timestamp"  : timestamps,
            "ghi"        : ghi.round(2),
            "dni"        : (ghi * 0.85).round(2),
            "dhi"        : (ghi * 0.15).round(2),
            "temp_air"   : temp_air.round(2),
            "wind_speed" : np.abs(np.random.normal(4, 2, n)).round(2),
            "cloud_type" : (cloud_cover * 9).astype(int),
        })

        return df

    # ------------------------------------------------------------------
    # Schema Validation
    # ------------------------------------------------------------------
    def _validate_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure required columns exist and types are correct."""
        required = ["timestamp", "ghi", "dni", "dhi",
                    "temp_air", "wind_speed", "cloud_type"]

        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df              = df.sort_values("timestamp").reset_index(drop=True)
        df              = df.dropna(subset=["ghi", "dni", "dhi"])

        self.logger.info(f"Schema validated — {len(df)} rows, {len(df.columns)} columns")
        return df