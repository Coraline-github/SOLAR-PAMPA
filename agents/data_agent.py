# DataAgent for SOLAR-PAMPA
# Fetches solar irradiance data from NSRDB API for La Pampa, Argentina
# Falls back to synthetic data if no API key is available
# Outputs: data/processed/solar_raw_validated.parquet

from unittest import result
from unittest import result
from urllib import response

import pandas as pd
import numpy as np
import requests
from pathlib import Path
from datetime import datetime, time

from xgboost import data
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
    # PVGIS Fetch
    # ------------------------------------------------------------------
    def _fetch_nsrdb(self) -> pd.DataFrame:
        """Route to PVGIS for South America coverage."""
        self.logger.info("Using PVGIS (ERA5) — covers La Pampa, Argentina")
        return self._fetch_pvgis()

    def _fetch_pvgis(self) -> pd.DataFrame:
        """
        Fetch hourly solar data from PVGIS (EU Commission).
        Covers South America. Free, no API key needed.
        Fetches multiple years and combines them.
        """
        import pvlib.iotools
        import time

        years  = [2018, 2019, 2020, 2021, 2022]
        frames = []

        for year in years:
            self.logger.info(f"Fetching PVGIS data for year {year}...")
            try:
                result = pvlib.iotools.get_pvgis_hourly(
                    latitude       = LATITUDE,
                    longitude      = LONGITUDE,
                    start          = year,
                    end            = year,
                    raddatabase    = "PVGIS-ERA5",
                    components     = True,
                    surface_tilt   = 0,
                    surface_azimuth= 0,
                    outputformat   = "json",
                    usehorizon     = True,
                    pvcalculation  = False,
                    map_variables  = False,
                )

                # Handle tuple unpacking
                if isinstance(result, tuple):
                    data = result[0]
                else:
                    data = result

                # Print columns once for debugging
                if year == 2022:
                    self.logger.info(f"PVGIS raw columns: {data.columns.tolist()}")

                # Fix: reset index to get timestamp as column
                data = data.copy()
                data.index.name = "timestamp"
                data = data.reset_index()

                # Fix: convert timezone-aware timestamp to naive
                data["timestamp"] = pd.to_datetime(
                    data["timestamp"]).dt.tz_localize(None)

                # PVGIS horizontal surface components:
                # Gb(i) = beam irradiance = DNI proxy
                # Gd(i) = diffuse irradiance = DHI proxy  
                # Gr(i) = reflected = ground diffuse
                # GHI = beam + diffuse on horizontal surface
                data = data.rename(columns={
                    "Gb(i)"  : "dni",
                    "Gd(i)"  : "dhi",
                    "T2m"    : "temp_air",
                    "WS10m"  : "wind_speed",
                })

                # Calculate GHI = DNI + DHI (on horizontal surface)
                if "ghi" not in data.columns:
                    if "dni" in data.columns and "dhi" in data.columns:
                        data["ghi"] = data["dni"] + data["dhi"]
                    elif "poa_direct" in data.columns:
                        data["ghi"] = (data["poa_direct"] +
                                       data.get("poa_sky_diffuse", 0) +
                                       data.get("poa_ground_diffuse", 0))

                # Add cloud_type placeholder
                data["cloud_type"] = 0  

                keep = ["timestamp", "ghi", "dni", "dhi",
                        "temp_air", "wind_speed", "cloud_type"]
                data = data[[c for c in keep if c in data.columns]]
                frames.append(data)
                self.logger.info(f"  ✓ {year}: {len(data)} rows")              

            except Exception as e:
                import traceback
                self.logger.error(f"  ✗ {year} failed: {e}")
                traceback.print_exc()

        if not frames:
            self.logger.error("All PVGIS fetches failed — using synthetic data")
            return self._generate_synthetic_data()

        df = pd.concat(frames, ignore_index=True)
        self.logger.info(f"Combined: {len(df)} rows across {len(frames)} years")
        return df

    #def _parse_nsrdb_columns(self, df: pd.DataFrame) -> pd.DataFrame:
    #    """Parse NSRDB raw CSV into standard schema."""
    #    df["timestamp"] = pd.to_datetime(
    #        df[["Year", "Month", "Day", "Hour", "Minute"]]
    #        .rename(columns={"Year":"year","Month":"month",
    #                          "Day":"day","Hour":"hour","Minute":"minute"})
    #    )
    #    rename_map = {
    #        "GHI"            : "ghi",
    #        "DNI"            : "dni",
    #        "DHI"            : "dhi",
    #        "Temperature"    : "temp_air",
    #        "Wind Speed"     : "wind_speed",
    #        "Cloud Type"     : "cloud_type",
    #    }
    #    df = df.rename(columns=rename_map)
    #    return df[["timestamp", "ghi", "dni", "dhi",
    #               "temp_air", "wind_speed", "cloud_type"]]

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