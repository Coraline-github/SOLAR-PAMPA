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