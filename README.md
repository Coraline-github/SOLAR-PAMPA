# ⚡ SOLAR-PAMPA
## Solar Output Logistics & Atmospheric Response Prediction for La Pampa, Argentina

---

## 🌍 Project Overview

The intermittent nature of solar energy poses challenges for grid integration and energy market operations. **SOLAR-PAMPA** develops a predictive model for solar power generation in the arid and semi-arid regions of La Pampa, Argentina. By analyzing historical solar irradiance and meteorological data, the system forecasts weekly solar energy output through a fully autonomous multi-agent pipeline.

This project demonstrates how data science and software engineering can contribute to the optimization and stability of renewable energy grids — directly addressing a critical challenge in Argentina's energy transition.

Solar and meteorological data is sourced from **PVGIS ERA5** (Photovoltaic Geographical Information System), developed by the **European Commission's Joint Research Centre**. ERA5 is the fifth generation ECMWF atmospheric reanalysis of global climate — one of the most comprehensive and scientifically validated meteorological datasets available, combining model data with worldwide observations into a globally complete dataset. It provides hourly estimates of atmospheric variables with full South America coverage, making it the most reliable freely available source for this region. Data spans **2018–2022** at hourly resolution, aggregated to weekly forecasts.

> **Location:** La Pampa, Argentina | **Coordinates:** -37.5°, -66.5° | **Timezone:** America/Argentina/Buenos_Aires

---

## 🎯 Specific Project Goals

1. **Meteorological & Solar Data Collection:** Gather historical solar irradiance and meteorological data specific to the La Pampa region using real satellite-derived sources.
2. **Feature Engineering:** Create physics-informed features from raw weather data that are highly correlated with solar power generation (clear-sky index, solar geometry, effective solar hours).
3. **Weekly Solar Output Forecasting:** Develop and evaluate time series forecasting models to predict weekly aggregate solar energy generation.
4. **Impact of Atmospheric Conditions:** Quantify the influence of key meteorological variables — particularly cloud cover — on solar energy production.
5. **Forecast Uncertainty Quantification:** Provide not just point forecasts but uncertainty intervals around predictions, critical for grid operators.

---

## 🔬 Hypothesis

We hypothesize that weekly solar energy output in La Pampa is primarily driven by solar irradiance and seasonal patterns, with atmospheric conditions playing a secondary but measurable role. Incorporating physically relevant features derived from these variables enables a robust time-series model to generate accurate weekly forecasts that can inform energy grid management and optimize solar farm operations.

**Result:** R² = 0.999, MAPE = 1.1% on held-out test data — hypothesis confirmed.

---

## 🏗️ Architecture — Multi-Agent Pipeline

The project is built as a **5-agent autonomous pipeline**. Each agent has a single responsibility and passes outputs to the next via Parquet files.

```
python pipeline.py
│
├── [1] DataAgent          → Fetches real solar data from PVGIS ERA5
│         output: solar_raw_validated.parquet
│
├── [2] FeatureAgent       → Physics-informed features via pvlib
│         output: features_weekly.parquet
│
├── [3] ModelingAgent      → XGBoost + future forecast
│         output: xgboost_solar.pkl + predictions.parquet
│                          + future_forecast.parquet
│
├── [4] EvaluationAgent    → Metrics + diagnostic plots
│         output: metrics.json + reports/figures/
│
└── [5] DashboardAgent     → Streamlit interactive dashboard
output: dashboard/app.py
```
---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| R²     | 0.999 |
| MAPE   | 1.1%  |
| MAE    | 0.326 kWh/m² |
| RMSE   | 0.442 kWh/m² |
| Test weeks | 42 |
| CI Coverage | 7.1% |

---

## 🚀 How To Run

### 1. Clone the repository
```bash
git clone https://github.com/Coraline-github/SOLAR-PAMPA.git
cd SOLAR-PAMPA
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment (optional)
Create a `.env` file in the root directory:
NSRDB_API_KEY=your_key_here

> ⚠️ Not required — the pipeline uses PVGIS by default (no API key needed). Only add this if you want to experiment with NSRDB in the future.

### 4. Run the full pipeline
```bash
python pipeline.py
```

This runs all 5 agents in sequence (~3-5 minutes). It will:
- Fetch 5 years of real hourly solar data from PVGIS
- Engineer physics-informed features
- Train and evaluate the XGBoost model
- Generate a 4-week future forecast
- Build the interactive dashboard

### 5. Launch the dashboard

**Windows:**
```bash
python -m streamlit run dashboard/app.py
```

**Mac/Linux:**
```bash
streamlit run dashboard/app.py
```

Open your browser at: `http://localhost:8501`

### 6. Resume from a specific stage
If the pipeline fails at any stage, resume from where it stopped:
```bash
python pipeline.py features     # skip data fetching
python pipeline.py modeling     # skip data + features
python pipeline.py evaluation   # only re-evaluate
python pipeline.py dashboard    # only rebuild dashboard
```

---

## 📁 Project Structure
```
SOLAR-PAMPA/
├── agents/
│   ├── init.py              # BaseAgent abstract class
│   ├── data_agent.py            # PVGIS ERA5 data fetching
│   ├── feature_agent.py         # Solar geometry + feature engineering
│   ├── modeling_agent.py        # XGBoost model + future forecast
│   ├── evaluation_agent.py      # Metrics (MAE, RMSE, MAPE, R²) + plots
│   └── dashboard_agent.py       # Streamlit app generator
├── data/
│   ├── raw/                     # Raw downloaded data
│   └── processed/               # Parquet files passed between agents
├── models/                      # Trained model, predictions, future forecast
├── reports/
│   └── figures/                 # Generated diagnostic plots
├── dashboard/
│   └── app.py                   # Streamlit interactive dashboard
├── pipeline.py                  # Main orchestrator — run this
├── config.py                    # All constants, paths, and settings
├── requirements.txt             # Python dependencies
├── .env                         # API keys (never commit this)
└── .gitignore
```
---

## 🌐 Data Source

| | Details |
|-|---------|
| **Source** | PVGIS ERA5 (EU Commission Joint Research Centre) |
| **Coverage** | Full South America ✅ |
| **Years** | 2018–2022 |
| **Resolution** | Hourly → aggregated to weekly |
| **Variables** | GHI, DNI, DHI, air temperature, wind speed |
| **API key required** | No — completely free |
| **URL** | https://re.jrc.ec.europa.eu/pvg_tools/ |
| **pvlib function** | `pvlib.iotools.get_pvgis_hourly()` |

### ⚠️ Why PVGIS instead of NSRDB?

The original design used NREL's NSRDB API. During implementation we discovered:
- `psm3-download` endpoint: **deprecated** (replaced by v3.2.2)
- `psm3-2-2-download` endpoint: returns **404 for South America**
- `PSM4 (get_nsrdb_psm4_aggregated)`: **CONUS only**, no Argentina coverage
- NREL domain migrating from `developer.nrel.gov` → `developer.nlr.gov` by April 30, 2026

**Solution:** PVGIS ERA5 provides equivalent quality data with full South America coverage and no API key required.

> If NSRDB access is needed in the future: register at https://developer.nrel.gov/signup/ and store the key in `.env` as `NSRDB_API_KEY`.

---

## 🔬 Features Engineered

| Feature | Description | Source |
|---------|-------------|--------|
| `solar_elevation` | Sun angle above horizon | pvlib |
| `solar_azimuth` | Sun compass direction | pvlib |
| `cos_zenith` | Cosine of solar zenith angle | pvlib |
| `ghi_clear_sky` | Theoretical clear-sky GHI | pvlib |
| `clear_sky_index` | GHI / GHI_clearsky (cloud proxy) | Derived |
| `cloud_fraction` | Cloud cover fraction 0–1 | Derived |
| `month_sin/cos` | Cyclical month encoding | Derived |
| `doy_sin/cos` | Cyclical day-of-year encoding | Derived |
| `ghi_lag_1/2/4/8/52w` | Lagged weekly GHI values | Derived |
| `ghi_roll_4/12w` | Rolling mean GHI | Derived |

---

## 🤖 Model Details

- **Algorithm:** XGBoost Regressor
- **Validation:** TimeSeriesSplit (5 folds) — data is never shuffled
- **Train/Test split:** 80% train / 20% test (strictly chronological)
- **Uncertainty:** ±1.96 × residual std → 95% confidence intervals
- **Future forecast:** 4 weeks ahead using learned seasonal patterns
- **Outlier removal:** IQR method (3× IQR threshold)
- **Partial week removal:** First and last weeks of dataset excluded

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `pvlib` | Solar geometry and clear-sky calculations |
| `xgboost` | Time series forecasting model |
| `pandas` / `pyarrow` | Data processing and Parquet I/O |
| `scikit-learn` | Preprocessing, cross-validation |
| `streamlit` | Interactive web dashboard |
| `plotly` | Interactive charts |
| `matplotlib` / `seaborn` | Static diagnostic plots |
| `python-dotenv` | Environment variable management |

---

## 💡 Why This Matters — Real Applications

- **Grid Stability:** Accurate forecasts help grid operators anticipate fluctuations in solar supply, enabling better management of conventional power plants or energy storage to maintain grid balance.
- **Energy Market Bidding:** Solar farm operators can use forecasts to make informed bids in electricity markets, optimizing revenue and avoiding penalties for inaccurate supply predictions.
- **Infrastructure Planning:** Long-term forecasts inform decisions about where to build new solar farms or invest in transmission infrastructure.
- **Hydropower Coordination:** In regions with mixed energy sources, solar forecasts help optimize hydropower dispatch — saving water when solar is abundant.
- **Resource Management:** For remote off-grid systems, knowing future solar availability is crucial for managing battery storage and ensuring continuous power supply.

---

## 🔮 Future Work

- [ ] Add 2023 data when PVGIS releases it
- [ ] Deploy to Streamlit Cloud (public URL)
- [ ] Improve confidence intervals using conformal prediction
- [ ] Add Prophet / LSTM model comparison
- [ ] Extend to multiple locations across La Pampa province
- [ ] Incorporate satellite cloud imagery for finer resolution
- [ ] Connect to real-time data streams for near-real-time forecasting
- [ ] Develop probabilistic forecasts for grid operators

---

## 👤 Target Audience

This project is tailored for **Energy Sector Professionals, Academia** (atmospheric physics, renewable energy research), and **Government Agencies** involved in energy policy and grid management in Argentina and similar regions. It showcases advanced time-series forecasting, real environmental data integration, physics-informed engineering, and a direct application to a pressing challenge in renewable energy.

---

*SOLAR-PAMPA | Solar Output Logistics & Atmospheric Response Prediction for La Pampa, Argentina*
