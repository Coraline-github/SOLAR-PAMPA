# EvaluationAgent for SOLAR-PAMPA
# Reads predictions.parquet from ModelingAgent
# Generates evaluation metrics and plots
# Outputs: reports/metrics.json + reports/figures/*.png

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from agents import BaseAgent
from config import FIGURES_DIR, TARGET_COLUMN


class EvaluationAgent(BaseAgent):

    def __init__(self, input_path: Path, output_path: Path):
        super().__init__(
            name        = "EvaluationAgent",
            input_path  = input_path,
            output_path = output_path
        )
        self.predictions_file = self.input_path  / "predictions.parquet"
        self.metrics_file     = self.output_path / "metrics.json"
        self.figures_dir      = FIGURES_DIR
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Plot style
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("husl")

    def validate_inputs(self) -> bool:
        if not self.predictions_file.exists():
            self.logger.error(f"Predictions not found: {self.predictions_file}")
            return False
        self.logger.info(f"Predictions found: {self.predictions_file}")
        return True

    def run(self) -> bool:
        self.logger.info("Loading predictions...")
        df = pd.read_parquet(self.predictions_file)
        df["week_start"] = pd.to_datetime(df["week_start"])

        self.logger.info("Computing metrics...")
        metrics = self._compute_metrics(df)

        self.logger.info("Saving metrics...")
        self._save_metrics(metrics)

        self.logger.info("Generating plots...")
        self._plot_forecast(df)
        self._plot_residuals(df)
        self._plot_scatter(df)
        self._plot_feature_importance()
        self._plot_feature_importance()
        self._plot_cloud_impact()

        self.logger.info(f"All figures saved to: {self.figures_dir}")
        return True

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def _compute_metrics(self, df: pd.DataFrame) -> dict:
        y_true = df["actual"].values
        y_pred = df["predicted"].values

        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2   = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

        # Coverage: % of actuals within confidence interval
        coverage = np.mean(
            (y_true >= df["lower_bound"].values) &
            (y_true <= df["upper_bound"].values)
        ) * 100

        metrics = {
            "mae"              : round(float(mae), 4),
            "rmse"             : round(float(rmse), 4),
            "mape_pct"         : round(float(mape), 2),
            "r2"               : round(float(r2), 4),
            "ci_coverage_pct"  : round(float(coverage), 2),
            "n_test_weeks"     : int(len(df)),
        }

        for k, v in metrics.items():
            self.logger.info(f"  {k:20s}: {v}")

        return metrics

    def _save_metrics(self, metrics: dict):
        with open(self.metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        self.logger.info(f"Metrics saved: {self.metrics_file}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    def _plot_forecast(self, df: pd.DataFrame):
        """Forecast vs actual with confidence interval."""
        fig, ax = plt.subplots(figsize=(14, 5))

        ax.fill_between(
            df["week_start"],
            df["lower_bound"],
            df["upper_bound"],
            alpha=0.25,
            color="steelblue",
            label="95% Confidence Interval"
        )
        ax.plot(df["week_start"], df["actual"],
                color="black", linewidth=1.5, label="Actual")
        ax.plot(df["week_start"], df["predicted"],
                color="steelblue", linewidth=1.5,
                linestyle="--", label="Predicted")

        ax.set_title("SOLAR-PAMPA — Weekly Solar Output Forecast vs Actual",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Week")
        ax.set_ylabel("GHI (kWh/m²)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        ax.legend()
        plt.tight_layout()

        path = self.figures_dir / "forecast_vs_actual.png"
        fig.savefig(path, dpi=150)
        plt.close()
        self.logger.info(f"Saved: {path}")

    def _plot_residuals(self, df: pd.DataFrame):
        """Residual plot over time."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        # Residuals over time
        axes[0].plot(df["week_start"], df["residual"],
                     color="tomato", linewidth=1)
        axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
        axes[0].set_title("Residuals Over Time")
        axes[0].set_xlabel("Week")
        axes[0].set_ylabel("Residual (kWh/m²)")
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

        # Residual distribution
        axes[1].hist(df["residual"], bins=30,
                     color="tomato", edgecolor="white", alpha=0.8)
        axes[1].axvline(0, color="black", linestyle="--", linewidth=0.8)
        axes[1].set_title("Residual Distribution")
        axes[1].set_xlabel("Residual (kWh/m²)")
        axes[1].set_ylabel("Count")

        plt.suptitle("SOLAR-PAMPA — Residual Analysis",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()

        path = self.figures_dir / "residuals.png"
        fig.savefig(path, dpi=150)
        plt.close()
        self.logger.info(f"Saved: {path}")

    def _plot_scatter(self, df: pd.DataFrame):
        """Predicted vs actual scatter plot."""
        fig, ax = plt.subplots(figsize=(6, 6))

        ax.scatter(df["actual"], df["predicted"],
                   alpha=0.6, color="steelblue", edgecolors="white", s=40)

        # Perfect prediction line
        min_val = min(df["actual"].min(), df["predicted"].min())
        max_val = max(df["actual"].max(), df["predicted"].max())
        ax.plot([min_val, max_val], [min_val, max_val],
                color="black", linestyle="--", linewidth=1, label="Perfect")

        ax.set_title("Predicted vs Actual", fontsize=13, fontweight="bold")
        ax.set_xlabel("Actual (kWh/m²)")
        ax.set_ylabel("Predicted (kWh/m²)")
        ax.legend()
        plt.tight_layout()

        path = self.figures_dir / "predicted_vs_actual.png"
        fig.savefig(path, dpi=150)
        plt.close()
        self.logger.info(f"Saved: {path}")

    def _plot_feature_importance(self):
        """Extract and plot XGBoost feature importances."""
        import pickle

        model_file = self.input_path / "xgboost_solar.pkl"
        if not model_file.exists():
            self.logger.warning("Model file not found — skipping feature importance")
            return

        with open(model_file, "rb") as f:
            artifacts = pickle.load(f)

        model        = artifacts["model"]
        feature_names = artifacts["features"]

        # Get importances
        importances = model.feature_importances_
        feat_df     = pd.DataFrame({
            "feature"    : feature_names,
            "importance" : importances
        }).sort_values("importance", ascending=False).head(15)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(
            feat_df["feature"][::-1],
            feat_df["importance"][::-1],
            color="steelblue",
            edgecolor="white"
        )
        ax.set_title("XGBoost Feature Importances — Top 15",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance Score")
        ax.axvline(x=feat_df["importance"].mean(),
                   color="tomato", linestyle="--",
                   linewidth=1, label="Mean importance")
        ax.legend()
        plt.tight_layout()

        path = self.figures_dir / "feature_importance.png"
        fig.savefig(path, dpi=150)
        plt.close()
        self.logger.info(f"Saved: {path}")

        # Save as CSV for dashboard use
        feat_df.to_csv(
            self.figures_dir / "feature_importance.csv",
            index=False
        )
        self.logger.info("Feature importances saved to CSV")

    def _plot_cloud_impact(self):
        """
        Analyze and plot the impact of cloud cover on solar output.
        Addresses README objective: quantify atmospheric influence on GHI.
        """
        features_file = self.input_path.parent / "data" / "processed" / "features_weekly.parquet"

        if not features_file.exists():
            # Try alternate path
            features_file = Path("data") / "processed" / "features_weekly.parquet"

        if not features_file.exists():
            self.logger.warning("Features file not found — skipping cloud impact analysis")
            return

        df = pd.read_parquet(features_file)
        self.logger.info("Generating cloud cover impact analysis...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Atmospheric Conditions Impact on Solar Output — La Pampa, Argentina",
            fontsize=14, fontweight="bold", y=1.02
        )

        # --- Plot 1: Cloud fraction vs GHI scatter ---
        axes[0,0].scatter(
            df["cloud_fraction"],
            df["ghi_weekly_kwh"],
            alpha=0.4,
            color="steelblue",
            edgecolors="white",
            s=30
        )
        # Trend line
        z = np.polyfit(df["cloud_fraction"], df["ghi_weekly_kwh"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df["cloud_fraction"].min(),
                             df["cloud_fraction"].max(), 100)
        axes[0,0].plot(x_line, p(x_line),
                       color="tomato", linewidth=2, linestyle="--")
        axes[0,0].set_title("Cloud Fraction vs Weekly GHI")
        axes[0,0].set_xlabel("Cloud Fraction (0=clear, 1=overcast)")
        axes[0,0].set_ylabel("Weekly GHI (kWh/m²)")

        # --- Plot 2: Clear Sky Index vs GHI scatter ---
        axes[0,1].scatter(
            df["csi_mean"],
            df["ghi_weekly_kwh"],
            alpha=0.4,
            color="orange",
            edgecolors="white",
            s=30
        )
        z2 = np.polyfit(df["csi_mean"], df["ghi_weekly_kwh"], 1)
        p2 = np.poly1d(z2)
        x_line2 = np.linspace(df["csi_mean"].min(),
                              df["csi_mean"].max(), 100)
        axes[0,1].plot(x_line2, p2(x_line2),
                       color="tomato", linewidth=2, linestyle="--")
        axes[0,1].set_title("Clear Sky Index vs Weekly GHI")
        axes[0,1].set_xlabel("Clear Sky Index (0=cloudy, 1=clear)")
        axes[0,1].set_ylabel("Weekly GHI (kWh/m²)")

        # --- Plot 3: GHI by cloud category ---
        df["cloud_category"] = pd.cut(
            df["cloud_fraction"],
            bins   = [0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels = ["Clear\n(0-20%)", "Mostly Clear\n(20-40%)",
                      "Partly Cloudy\n(40-60%)", "Mostly Cloudy\n(60-80%)",
                      "Overcast\n(80-100%)"]
        )
        cloud_groups = df.groupby("cloud_category", observed=True)["ghi_weekly_kwh"]
        means  = cloud_groups.mean()
        stds   = cloud_groups.std()

        bars = axes[1,0].bar(
            means.index,
            means.values,
            yerr   = stds.values,
            color  = ["#FFD700", "#FFA500", "#87CEEB", "#4682B4", "#2F4F7F"],
            edgecolor = "white",
            capsize   = 4
        )
        axes[1,0].set_title("Average GHI by Cloud Cover Category")
        axes[1,0].set_xlabel("Cloud Cover Category")
        axes[1,0].set_ylabel("Avg Weekly GHI (kWh/m²)")
        axes[1,0].tick_params(axis="x", labelsize=8)

        # --- Plot 4: Correlation heatmap ---
        corr_cols = [
            "ghi_weekly_kwh", "cloud_fraction", "csi_mean",
            "cloudy_hours", "clear_hours", "temp_mean", "wind_mean"
        ]
        existing  = [c for c in corr_cols if c in df.columns]
        corr_matrix = df[existing].corr()

        im = axes[1,1].imshow(
            corr_matrix,
            cmap   = "RdYlGn",
            aspect = "auto",
            vmin   = -1,
            vmax   = 1
        )
        axes[1,1].set_xticks(range(len(existing)))
        axes[1,1].set_yticks(range(len(existing)))
        axes[1,1].set_xticklabels(existing, rotation=45,
                                   ha="right", fontsize=8)
        axes[1,1].set_yticklabels(existing, fontsize=8)
        axes[1,1].set_title("Correlation Matrix — Weather vs Solar Output")
        plt.colorbar(im, ax=axes[1,1], shrink=0.8)

        # Add correlation values
        for i in range(len(existing)):
            for j in range(len(existing)):
                axes[1,1].text(
                    j, i,
                    f"{corr_matrix.iloc[i,j]:.2f}",
                    ha="center", va="center",
                    fontsize=7,
                    color="black"
                )

        plt.tight_layout()

        path = self.figures_dir / "cloud_impact.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        self.logger.info(f"Saved: {path}")

        # Save correlation with target for dashboard
        target_corr = df[existing].corr()["ghi_weekly_kwh"].drop(
            "ghi_weekly_kwh"
        ).sort_values()
        target_corr.to_csv(
            self.figures_dir / "weather_correlations.csv"
        )
        self.logger.info("Weather correlations saved to CSV")