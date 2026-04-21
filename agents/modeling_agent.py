# ModelingAgent for SOLAR-PAMPA
# Reads features_weekly.parquet
# Trains XGBoost model to forecast weekly solar output
# Outputs: models/xgboost_solar.pkl + models/predictions.parquet

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from agents import BaseAgent
from config import TARGET_COLUMN, RANDOM_SEED, TEST_SIZE


class ModelingAgent(BaseAgent):

    def __init__(self, input_path: Path, output_path: Path):
        super().__init__(
            name        = "ModelingAgent",
            input_path  = input_path,
            output_path = output_path
        )
        self.input_file       = self.input_path  / "features_weekly.parquet"
        self.model_file       = self.output_path / "xgboost_solar.pkl"
        self.predictions_file = self.output_path / "predictions.parquet"
        self.scaler_file      = self.output_path / "scaler.pkl"

    def validate_inputs(self) -> bool:
        if not self.input_file.exists():
            self.logger.error(f"Input file not found: {self.input_file}")
            return False
        self.logger.info(f"Input file found: {self.input_file}")
        return True

    def run(self) -> bool:
        self.logger.info("Loading features...")
        df = pd.read_parquet(self.input_file)

        self.logger.info("Removing outliers...")
        df = self._remove_outliers(df)

        self.logger.info(f"Dataset: {len(df)} weeks, target: {TARGET_COLUMN}")

        X, y          = self._prepare_features(df)
        X_train, X_test, y_train, y_test, dates_test = self._split_data(df, X, y)

        self.logger.info("Training XGBoost model...")
        model, scaler = self._train_model(X_train, y_train)

        self.logger.info("Generating predictions...")
        predictions   = self._predict(model, scaler, X_test, y_test, dates_test, df)

        self.logger.info("Saving model and predictions...")
        self._save_artifacts(model, scaler, predictions)

        return True

    # ------------------------------------------------------------------
    # Feature Preparation
    # ------------------------------------------------------------------
    def _prepare_features(self, df: pd.DataFrame):
        """Select feature columns and target."""

        # Drop non-feature columns
        drop_cols = [TARGET_COLUMN, "week_start"]
        feature_cols = [c for c in df.columns if c not in drop_cols]

        X = df[feature_cols].values
        y = df[TARGET_COLUMN].values

        self.feature_names = feature_cols
        self.logger.info(f"Features ({len(feature_cols)}): {feature_cols}")
        return X, y

    # ------------------------------------------------------------------
    # Train / Test Split
    # ------------------------------------------------------------------
    def _split_data(self, df, X, y):
        """
        Time series split — never shuffle.
        Last TEST_SIZE% of weeks = test set.
        """
        split_idx = int(len(X) * (1 - TEST_SIZE))

        X_train = X[:split_idx]
        X_test  = X[split_idx:]
        y_train = y[:split_idx]
        y_test  = y[split_idx:]

        dates_test = df["week_start"].values[split_idx:]

        self.logger.info(f"Train: {len(X_train)} weeks | Test: {len(X_test)} weeks")
        return X_train, X_test, y_train, y_test, dates_test

    # ------------------------------------------------------------------
    # Model Training
    # ------------------------------------------------------------------
    def _train_model(self, X_train, y_train):
        """Train XGBoost with cross-validation."""

        scaler  = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        model = xgb.XGBRegressor(
            n_estimators      = 300,
            learning_rate     = 0.05,
            max_depth         = 5,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            random_state      = RANDOM_SEED,
            early_stopping_rounds = 20,
            eval_metric       = "mae",
        )

        # Time series cross-validation
        tscv    = TimeSeriesSplit(n_splits=5)
        cv_maes = []

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            model.fit(
                X_tr, y_tr,
                eval_set        = [(X_val, y_val)],
                verbose         = False,
            )

            preds   = model.predict(X_val)
            fold_mae = mean_absolute_error(y_val, preds)
            cv_maes.append(fold_mae)
            self.logger.info(f"  Fold {fold+1} MAE: {fold_mae:.3f} kWh/m²")

        self.logger.info(f"CV MAE: {np.mean(cv_maes):.3f} ± {np.std(cv_maes):.3f}")

        # Final fit on full training data
        model.fit(
            X_scaled, y_train,
            eval_set  = [(X_scaled, y_train)],
            verbose   = False
        )

        return model, scaler

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------
    def _predict(self, model, scaler, X_test, y_test,
                 dates_test, df) -> pd.DataFrame:
        """Generate predictions with confidence intervals."""

        X_scaled = scaler.transform(X_test)
        y_pred   = model.predict(X_scaled)

        # Simple uncertainty: ±1 std of residuals from training
        X_train_scaled = scaler.transform(
            df[self.feature_names].values[:len(df) - len(X_test)]
        )
        train_preds    = model.predict(X_train_scaled)
        residual_std   = np.std(
            df[TARGET_COLUMN].values[:len(df) - len(X_test)] - train_preds
        )

        predictions = pd.DataFrame({
            "week_start"  : dates_test,
            "actual"      : y_test,
            "predicted"   : y_pred,
            "lower_bound" : y_pred - 1.96 * residual_std,
            "upper_bound" : y_pred + 1.96 * residual_std,
            "residual"    : y_test - y_pred,
        })

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        self.logger.info(f"Test MAE  : {mae:.3f} kWh/m²")
        self.logger.info(f"Test RMSE : {rmse:.3f} kWh/m²")
        self.logger.info(f"Test MAPE : {mape:.2f}%")

        return predictions

    # ------------------------------------------------------------------
    # Save Artifacts
    # ------------------------------------------------------------------
    def _save_artifacts(self, model, scaler, predictions):
        with open(self.model_file, "wb") as f:
            pickle.dump({"model": model,
                         "scaler": scaler,
                         "features": self.feature_names}, f)

        predictions.to_parquet(self.predictions_file, index=False)

        self.logger.info(f"Model saved     : {self.model_file}")
        self.logger.info(f"Predictions saved: {self.predictions_file}")

    # ------------------------------------------------------------------
    # Remove outliers
    # ------------------------------------------------------------------
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove extreme outlier weeks before training.
        Uses IQR method on the target column.
        """
        Q1  = df[TARGET_COLUMN].quantile(0.25)
        Q3  = df[TARGET_COLUMN].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 3.0 * IQR
        upper = Q3 + 3.0 * IQR

        before = len(df)
        df     = df[
            (df[TARGET_COLUMN] >= lower) &
            (df[TARGET_COLUMN] <= upper)
        ].reset_index(drop=True)

        removed = before - len(df)
        if removed > 0:
            self.logger.info(f"Removed {removed} outlier weeks")
        else:
            self.logger.info("No outliers found")

        return df