# FeatureAgent for SOLAR-PAMPA
# Reads solar_raw_validated.parquet
# Creates physics-informed features for forecasting
# Outputs: data/processed/features_weekly.parquet

import pandas as pd
import numpy as np
import pvlib
from pathlib import Path
from agents import BaseAgent
from config import LATITUDE, LONGITUDE, TIMEZONE, TARGET_COLUMN


class FeatureAgent(BaseAgent):

    def __init__(self, input_path: Path, output_path: Path):
        super().__init__(
            name        = "FeatureAgent",
            input_path  = input_path,
            output_path = output_path
        )
        self.input_file  = self.input_path  / "solar_raw_validated.parquet"
        self.output_file = self.output_path / "features_weekly.parquet"

    def validate_inputs(self) -> bool:
        if not self.input_file.exists():
            self.logger.error(f"Input file not found: {self.input_file}")
            return False
        self.logger.info(f"Input file found: {self.input_file}")
        return True

    def run(self) -> bool:
        self.logger.info("Loading raw data...")
        df = pd.read_parquet(self.input_file)

        self.logger.info("Engineering features...")
        df = self._add_solar_geometry(df)
        df = self._add_clear_sky_index(df)
        df = self._add_time_features(df)
        df = self._add_cloud_features(df)

        self.logger.info("Aggregating to weekly...")
        weekly = self._aggregate_weekly(df)

        self.logger.info("Adding lag features...")
        weekly = self._add_lag_features(weekly)

        weekly.to_parquet(self.output_file, index=False)
        self.logger.info(f"Saved {len(weekly)} weeks to {self.output_file}")
        self.logger.info(f"Features: {list(weekly.columns)}")
        return True

    # ------------------------------------------------------------------
    # Solar Geometry (pvlib)
    # ------------------------------------------------------------------
    def _add_solar_geometry(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add solar position features using pvlib."""
        self.logger.info("Calculating solar geometry with pvlib...")

        location   = pvlib.location.Location(
            latitude  = LATITUDE,
            longitude = LONGITUDE,
            tz        = TIMEZONE,
            altitude  = 200
        )

        solar_pos  = location.get_solarposition(df["timestamp"])

        df["solar_elevation"] = solar_pos["elevation"].values
        df["solar_azimuth"]   = solar_pos["azimuth"].values
        df["cos_zenith"]      = np.cos(
            np.radians(solar_pos["zenith"].values)
        ).clip(0, 1)

        return df

    # ------------------------------------------------------------------
    # Clear Sky Index
    # ------------------------------------------------------------------
    def _add_clear_sky_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clear Sky Index (CSI) = GHI_measured / GHI_clear_sky
        CSI close to 1 = clear sky
        CSI close to 0 = heavy cloud cover
        """
        self.logger.info("Calculating clear sky index...")

        location  = pvlib.location.Location(
            latitude  = LATITUDE,
            longitude = LONGITUDE,
            tz        = TIMEZONE
        )

        times     = pd.DatetimeIndex(df["timestamp"])
        clear_sky = location.get_clearsky(times)
        ghi_clear = clear_sky["ghi"].values

        # Avoid division by zero during nighttime
        df["ghi_clear_sky"] = ghi_clear
        df["clear_sky_index"] = np.where(
            ghi_clear > 10,
            (df["ghi"] / ghi_clear).clip(0, 1.5),
            0.0
        )

        return df

    # ------------------------------------------------------------------
    # Time Features
    # ------------------------------------------------------------------
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical time features."""
        df["month"]       = df["timestamp"].dt.month
        df["day_of_year"] = df["timestamp"].dt.day_of_year
        df["hour"]        = df["timestamp"].dt.hour
        df["week"]        = df["timestamp"].dt.isocalendar().week.astype(int)
        df["year"]        = df["timestamp"].dt.year

        # Cyclical encoding — tells model Jan and Dec are close
        df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)
        df["doy_sin"]     = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["doy_cos"]     = np.cos(2 * np.pi * df["day_of_year"] / 365)

        return df

    # ------------------------------------------------------------------
    # Cloud Features
    # ------------------------------------------------------------------
    def _add_cloud_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive cloud-related features from cloud_type."""
        # cloud_type: 0=clear, 1-3=thin, 4-6=medium, 7-9=thick
        df["cloud_fraction"] = (df["cloud_type"] / 9).clip(0, 1)
        df["is_cloudy"]      = (df["cloud_type"] >= 4).astype(int)
        df["is_clear"]       = (df["cloud_type"] == 0).astype(int)
        return df

    # ------------------------------------------------------------------
    # Weekly Aggregation
    # ------------------------------------------------------------------
    def _aggregate_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate hourly data to weekly.
        GHI sum = total energy that week (proxy for kWh output)
        """
        df["week_start"] = df["timestamp"].dt.to_period("W").dt.start_time

        weekly = df.groupby("week_start").agg(
            ghi_weekly_kwh    = ("ghi",             "sum"),
            ghi_mean          = ("ghi",             "mean"),
            dni_mean          = ("dni",             "mean"),
            dhi_mean          = ("dhi",             "mean"),
            temp_mean         = ("temp_air",        "mean"),
            temp_max          = ("temp_air",        "max"),
            wind_mean         = ("wind_speed",      "mean"),
            csi_mean          = ("clear_sky_index", "mean"),
            cloud_fraction    = ("cloud_fraction",  "mean"),
            cloudy_hours      = ("is_cloudy",       "sum"),
            clear_hours       = ("is_clear",        "sum"),
            solar_elev_mean   = ("solar_elevation", "mean"),
            month_sin         = ("month_sin",       "first"),
            month_cos         = ("month_cos",       "first"),
            doy_sin           = ("doy_sin",         "first"),
            doy_cos           = ("doy_cos",         "first"),
            year              = ("year",            "first"),
        ).reset_index()

        # Convert GHI sum to approximate kWh/m²
        weekly["ghi_weekly_kwh"] = (weekly["ghi_weekly_kwh"] / 1000).round(3)

        return weekly

    # ------------------------------------------------------------------
    # Lag Features
    # ------------------------------------------------------------------
    def _add_lag_features(self, weekly: pd.DataFrame) -> pd.DataFrame:
        """
        Add previous weeks as features.
        Critical for time series — model can learn from recent history.
        """
        target = TARGET_COLUMN

        for lag in [1, 2, 4, 8, 52]:
            weekly[f"ghi_lag_{lag}w"] = weekly[target].shift(lag)

        # Rolling averages
        weekly["ghi_roll_4w"]  = weekly[target].shift(1).rolling(4).mean()
        weekly["ghi_roll_12w"] = weekly[target].shift(1).rolling(12).mean()

        # Drop rows where lags are NaN (first ~52 weeks)
        weekly = weekly.dropna().reset_index(drop=True)

        return weekly