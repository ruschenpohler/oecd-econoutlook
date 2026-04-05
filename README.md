# GDP Nowcasting with Spark ML — OECD Economic Outlook

A PySpark ML pipeline that nowcasts annual GDP growth across 38 OECD member countries using lagged macroeconomic indicators from the OECD Economic Outlook. The project demonstrates end-to-end command of Spark ML, time-aware model evaluation, and the OECD SDMX REST API.

Loosely inspired by Dorville et al. (2025), "Towards more timely measures of labour productivity growth", *OECD Statistics Working Papers* No. 2025/01 ([doi](https://doi.org/10.1787/436ecbb5-en), [blog](https://oecdstatistics.blog/2025/03/31/nowcasting-labour-productivity-growth-with-machine-learning-and-mixed-frequency-data/)). That paper nowcasts labour productivity using mixed-frequency ML models (GBT, RF, LASSO, Ridge, DFM with MIDAS) on a pooled cross-country panel with time-aware evaluation. This project adapts the same core structure — pooled panel, GBT as primary model, time-based train/test split — to GDP growth using the Economic Outlook, implemented in PySpark. Three deliberate divergences: (1) annual frequency only (no MIDAS), (2) GDP growth as target rather than derived productivity, (3) Spark ML as execution engine. No claims about methodological equivalence are made.

---

## Data

**Source:** OECD Economic Outlook via SDMX REST API (`OECD.ECO.MAD,DSD_EO@DF_EO,`). No authentication required; data pulled programmatically in `src/01_pull_data.py`.

**Panel:** 38 OECD member countries × 1990–2027. Non-OECD partners (ARG, BRA, CHN, IDN, IND, ...), aggregate entities (OECD, EA20, G7, W), and commodity groupings were excluded; the 38-country restriction is applied as an explicit positive list.

| SDMX code | Variable | Role |
|-----------|----------|------|
| `GDPV_ANNPCT` | Real GDP growth (%) | Target |
| `UNR` | Unemployment rate (%) | Feature |
| `CBGDPR` | Current account balance (% of GDP) | Feature |
| `ITV_ANNPCT` | Gross fixed capital formation, volume growth (%) | Feature |
| `XGSV_ANNPCT` | Export volume growth (%) | Feature |
| `MGSV_ANNPCT` | Import volume growth (%) | Feature |

Two variables were dropped after inspection: `CPI_YTYPCT` (headline inflation) is structurally unreported for all Eurozone members, which report only at the aggregate level; `SRATIO` (household saving ratio) is missing for 26 countries including FRA, GBR, GRC, and PRT. Including either would have introduced systematic selection bias into the feature set.

The resulting panel is 94.3% complete across all six variables. Residual missingness (~5.7%) is concentrated in contiguous early years for newer members (CRI, COL, LTU, LVA) entering the panel at their first complete year. This is handled by lag-induced `dropna` in Phase 2 and `VectorAssembler(handleInvalid="skip")` in Phase 3. The pipeline treats this as pragmatic complete-case analysis; a production system would model the missingness mechanism explicitly (MAR/MNAR tests, multiple imputation), but that is outside the scope here.

`UNR` contains 28 observations (2.0% of non-missing values) exceeding 18% — more than 5 SDs above the panel mean of 7.4% (SD 3.8). These are documented structural episodes: Greece 2012–2018 (peak 27.5%, Eurozone debt crisis), Spain 2010–2016 (peak 26.1%), Latvia 1996 and 2010 (post-Soviet transition and credit-bubble collapse), Poland and Slovak Republic 2001–2004 (pre-EU accession restructuring), Costa Rica 2020 (COVID tourism shock). All use the ILO-harmonised definition consistently across countries. GBT is scale-invariant to these observations (tree splits on rank thresholds; `StandardScaler` on lag columns further reduces leverage), though extreme unemployment regimes may have qualitatively different GDP dynamics that a country fixed effect only partially captures.

---

## Pipeline

```
SDMX REST API
    └─ 01_pull_data.py        pandas: pivot wide, restrict to 38 members, save CSV
        └─ 02_feature_engineering.py  PySpark: Window lags (lag-1, lag-2, acceleration)
            └─ 03_pipeline.py         Spark ML Pipeline:
                │   StringIndexer (country_code → country_idx)
                │   VectorAssembler  (14 features → DenseVector)
                │   StandardScaler   (withStd=True, withMean=False)
                │   GBTRegressor     (featuresCol="features", labelCol="gdpv_annpct")
                └─ 04_evaluate.py    CrossValidator, RF baseline, AR(1) OLS, robustness cuts
                        └─ output/   predictions.csv, metrics.json, figures
```

Feature vector (14 dimensions): `country_idx` (StringIndexer ordinal) + `gdp_lag1`, `gdp_lag2`, `gdp_accel` + lag-1 and lag-2 of all five macro variables.

The train/test split is time-based: train on years < 2019, test on 2019–2027. Random splits are inappropriate for time-series data because they leak future information into training via lagged features.

`CrossValidator` wraps the entire `Pipeline`, not just the `GBTRegressor`. This means `StandardScaler`'s mean/std is re-computed on each fold's training subset — preventing the scaler from seeing held-out data, which would otherwise constitute a subtle form of data leakage.

---

## Design choices

**GBT as primary model.** Gradient-boosted trees handle heterogeneous cross-country panels well: they are robust to outliers (scale-invariant splits), can capture non-linear interaction effects between macro variables, and do not require explicit structural assumptions about the GDP process. Dorville et al. found GBT the best single model in 35 of 40 countries; the same prior motivated its use here.

**RF as baseline.** Random Forest provides a natural variance-regularised alternative. Where GBT sequentially corrects residuals and can overfit to idiosyncratic training-set patterns, RF's bagging averages over many independent trees, producing a model with higher bias but lower variance. This contrast is informative when evaluating robustness under distribution shift.

**AR(1) + country fixed effects as econometric benchmark.** The standard panel-data benchmark for GDP nowcasting is `gdpv_annpct ~ gdp_lag1 + α_country`, estimated as OLS with country dummies. Implemented in Spark as `StringIndexer → OneHotEncoder → VectorAssembler → LinearRegression`. `StringIndexer` alone produces an ordinal integer per country (AUS=0, AUT=1, ...) which `LinearRegression` treats as a continuous variable — meaningless as a fixed effect. `OneHotEncoder` converts each index to a binary indicator, giving the model a separate intercept per country as intended. Fitted on the training split and evaluated on the same test cuts as GBT and RF.

**Lag structure.** Two lags of each variable capture both first-order persistence of the macro cycle and the second-order correction typically observed in GDP reversals. `gdp_accel = gdp_lag1 − gdp_lag2` encodes the direction of the cycle explicitly, which tree-based models can use without having to discover the difference from two separate features.

---

## Results

All metrics are on the test set (2019–2027, n=342). Best CV hyperparameters for GBT: `maxDepth=3`, `stepSize=0.1`.

**Full test set:**

| Model | RMSE (pp) | MAE (pp) | R² |
|-------|-----------|----------|----|
| GBT (CV) | 3.57 | 2.28 | −0.165 |
| RF | **3.37** | **2.16** | **−0.038** |
| AR(1) OLS | 3.62 | 2.35 | −0.199 |

All three models show negative R² on the full test set. The test period contains 2020 (COVID-19 crash, per-year RMSE 7.2 pp) and 2021 (recovery rebound, RMSE 5.7 pp) — both are exogenous shocks with no signal in annual lagged indicators, and together they dominate the aggregate error. Post-hoc evaluation across progressively cleaner sample cuts isolates how much of the negative R² is COVID-specific versus structural.

**COVID robustness cuts (post-hoc diagnostic — models not retrained):**

| Cut | n | GBT R² | RF R² | AR(1) R² | Winner |
|-----|---|--------|-------|----------|--------|
| Full test (2019–2027) | 342 | −0.165 | −0.038 | −0.199 | RF |
| Excl. 2020 | 304 | −0.292 | −0.090 | −0.403 | RF |
| Excl. 2020–21 | 266 | −0.304 | +0.054 | −0.023 | RF |
| Excl. 2020–23 | 190 | −0.367 | **+0.207** | +0.073 | RF |

Excl. 2020–23 removes the four years where either outcomes or lag features are COVID-contaminated. `gdp_lag1` in 2022 equals the 2021 actual (+6.9 pp rebound); `gdp_lag2` in 2023 equals the same. Contamination clears after 2023. The remaining observations (2019 + 2024–2027, n=190) form the only window where both outcomes and features are from the pre-COVID distribution.

RF is the best model in every cut. GBT's R² deteriorates monotonically as the sample is cleaned — consistent with a bias-variance interpretation. GBT, as a low-bias/high-variance estimator, overfits the variance structure of the 1992–2018 training distribution. Under the distribution shift introduced by COVID and its aftermath, that excess variance becomes prediction error. RF's bagging reduces variance at the cost of slightly higher bias, making it more robust when the test distribution diverges from training. AR(1) recovers to positive R² in the clean window (+0.07), confirming that its poor performance in earlier cuts is driven primarily by the 2021 rebound propagating through the lag channel rather than by fundamental model misspecification.

**Top features (GBT):** GDP growth (t−1) 0.148, GDP growth (t−2) 0.094, Import growth (t−1) 0.086, Current account (t−1) 0.079, GDP acceleration 0.077. The AR(1) persistence term dominates, consistent with GDP growth's strong autocorrelation at annual frequency. Import growth ranks third, likely capturing domestic demand momentum.

A known limitation: Spark's `CrossValidator` uses random k-fold partitioning rather than expanding-window time-series CV. In-fold validation should ideally respect temporal order; correcting this would require a custom CV splitter outside the standard Spark ML API.

---

## Usage

**Local:**
```bash
uv sync
uv run python src/01_pull_data.py
uv run python src/02_feature_engineering.py
uv run python src/03_pipeline.py
uv run python src/04_evaluate.py
```

**Cluster (YARN):**
```bash
spark-submit \
  --master yarn \
  --deploy-mode client \
  --num-executors 4 \
  --executor-memory 4g \
  src/04_evaluate.py
```

Python 3.10+, PySpark 3.5.x, Java 11. All dependencies managed via `uv`; `uv.lock` pins exact versions for reproducibility.

---

## Project structure

```
oecd-econoutlook/
├── README.md
├── pyproject.toml           # PEP 621 project manifest
├── uv.lock                  # Pinned dependency lockfile
├── .python-version
├── .gitignore
├── data/                    # gitignored — regenerated by scripts
├── notebooks/               # gitignored — interactive scratchpad
├── src/
│   ├── 01_pull_data.py      # SDMX API pull, pivot, clean
│   ├── 02_feature_engineering.py  # Spark Window lags
│   ├── 03_pipeline.py       # Spark ML Pipeline assembly and fit
│   ├── 04_evaluate.py       # CV, RF baseline, AR(1), robustness
│   ├── ar1_robustness.py    # Standalone AR(1) robustness table
│   └── labels.py            # Variable labels and figure utilities
└── output/                  # gitignored — regenerated by scripts
```
