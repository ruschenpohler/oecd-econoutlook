"""
Phase 4: Evaluation & Interpretation
======================================
Loads train/test splits, re-fits the GBT pipeline, fits an RF baseline,
runs CrossValidator for hyperparameter tuning, evaluates both models,
extracts feature importances, and saves all outputs.

Outputs (main analysis)
-----------------------
  output/predictions.csv            — country, year, actual, gbt_pred, rf_pred
  output/metrics.json               — RMSE/MAE/R² for all models, feature importances,
                                      best CV hyperparameters, COVID robustness table
  output/prediction_diagnostics.png — publication-quality 4-panel diagnostic figure
  output/feature_importance.png     — all 14 feature importances (GBT)

Post-hoc COVID robustness exercise (Section 9 onwards)
-------------------------------------------------------
  Motivation: the test set (2019–2027) contains 2020, an exogenous shock with no
  signal in lagged annual indicators. Two post-hoc diagnostics separate model
  performance from COVID-specific failure:

    1. Exclude-2020 metrics: re-evaluate all three models on test \ {2020}.
       Not a fix — a diagnostic. Answers "how does the model perform in normal years?"

    2. AR(1) OLS baseline: gdpv_annpct ~ gdp_lag1, country fixed effects via Spark StringIndexer.
       Fit on train, evaluate on test (full + excl. 2020). The standard econometric
       benchmark for GDP nowcasting. Answers "does GBT add anything beyond naive
       persistence?" If GBT beats AR(1) in normal years, that's the signal.

  Both exercises are clearly labelled ex-post throughout. The COVID dummy approach
  (feeding a known-shock indicator to the model) is deliberately excluded: it would
  require ex-ante knowledge of 2020 being COVID, violating the nowcast premise.

  Outputs:
    output/robustness_covid.png     — 3-model × 2-sample metrics table + AR(1) scatter
    output/metrics.json             — updated with 'robustness' key
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from labels import SHORT, DEFS, add_footer

sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# 1. SparkSession
# ---------------------------------------------------------------------------
spark = (
    SparkSession.builder.master("local[*]")
    .appName("OECD-GDP-Nowcast-Phase4")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# ---------------------------------------------------------------------------
# 2. Load data and define pipeline stages
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 1/6: Loading train/test data from CSV")
print("=" * 60)
train = spark.read.csv(str(ROOT / "data/train.csv"), header=True, inferSchema=True)
test = spark.read.csv(str(ROOT / "data/test.csv"), header=True, inferSchema=True)
print(f"Train: {train.count()} rows | Test: {test.count()} rows")

# ---------------------------------------------------------------------------
# 3. Gradient Boosted Tree with CrossValidator
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print(
    "STEP 2/6: Building ML pipeline (StringIndexer → VectorAssembler → StandardScaler → GBTRegressor)"
)
print("=" * 60)
# 3. GBT pipeline + CrossValidator
# ---------------------------------------------------------------------------
gbt = GBTRegressor(
    featuresCol="features", labelCol="gdpv_annpct", maxIter=100, maxDepth=4, seed=42
)
gbt_pipeline = Pipeline(stages=[indexer, assembler, scaler, gbt])

param_grid = (
    ParamGridBuilder()
    .addGrid(gbt.maxDepth, [3, 5])
    .addGrid(gbt.stepSize, [0.05, 0.1])
    .build()
)

evaluator = RegressionEvaluator(
    labelCol="gdpv_annpct", predictionCol="prediction", metricName="rmse"
)

cv = CrossValidator(
    estimator=gbt_pipeline,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=3,
    seed=42,
)

print("Fitting GBT with CrossValidator (4 param combos × 3 folds)...")
cv_model = cv.fit(train)
best_gbt = cv_model.bestModel

print(f"Best maxDepth: {best_gbt.stages[-1].getOrDefault('maxDepth')}")
print(f"Best stepSize: {best_gbt.stages[-1].getOrDefault('stepSize')}")
print(f"CV avg RMSE per combo: {[round(x, 3) for x in cv_model.avgMetrics]}")

# ---------------------------------------------------------------------------
# 4. Random Forest baseline
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3/6: Fitting Random Forest baseline")
print("=" * 60)
rf = RandomForestRegressor(
    featuresCol="features", labelCol="gdpv_annpct", numTrees=100, maxDepth=5, seed=42
)
rf_pipeline = Pipeline(stages=[indexer, assembler, scaler, rf])
print("Fitting Random Forest baseline...")
rf_model = rf_pipeline.fit(train)


# ---------------------------------------------------------------------------
# 5. Predictions and metrics
# ---------------------------------------------------------------------------
def eval_metrics(predictions_df):
    out = {}
    for metric in ["rmse", "mae", "r2"]:
        out[metric] = round(
            RegressionEvaluator(
                labelCol="gdpv_annpct", predictionCol="prediction", metricName=metric
            ).evaluate(predictions_df),
            4,
        )
    return out


gbt_preds = best_gbt.transform(test).withColumnRenamed("prediction", "gbt_pred")
rf_preds = rf_model.transform(test).withColumnRenamed("prediction", "rf_pred")

# Join on (country_code, year)
preds_joined = gbt_preds.select(
    "country_code", "country_name", "year", "gdpv_annpct", "gbt_pred"
).join(rf_preds.select("country_code", "year", "rf_pred"), on=["country_code", "year"])

gbt_metrics = eval_metrics(gbt_preds.withColumnRenamed("gbt_pred", "prediction"))
rf_metrics = eval_metrics(rf_preds.withColumnRenamed("rf_pred", "prediction"))

print(
    f"\nGBT  — RMSE: {gbt_metrics['rmse']}, MAE: {gbt_metrics['mae']}, R²: {gbt_metrics['r2']}"
)
print(
    f"RF   — RMSE: {rf_metrics['rmse']},  MAE: {rf_metrics['mae']},  R²: {rf_metrics['r2']}"
)

# ---------------------------------------------------------------------------
# 6. Feature importances
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4/6: Computing feature importances")
print("=" * 60)
fi_vec = best_gbt.stages[-1].featureImportances
fi_dict = {
    FEATURE_NAMES[i]: round(float(fi_vec[i]), 4) for i in range(len(FEATURE_COLS))
}
fi_sorted = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)

print("\nTop 10 feature importances (GBT):")
for name, imp in fi_sorted[:10]:
    print(f"  {name:30s} {imp:.4f}")

# ---------------------------------------------------------------------------
# 7. Save outputs
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6/6: Saving outputs to CSV and JSON")
print("=" * 60)
os.makedirs(ROOT / "output", exist_ok=True)

pdf = preds_joined.toPandas().sort_values(["country_code", "year"])
pdf.to_csv(ROOT / "output/predictions.csv", index=False)

metrics_out = {
    "gbt": gbt_metrics,
    "rf": rf_metrics,
    "feature_importance": fi_dict,
    "best_hyperparams": {
        "maxDepth": int(best_gbt.stages[-1].getOrDefault("maxDepth")),
        "stepSize": float(best_gbt.stages[-1].getOrDefault("stepSize")),
    },
    "cv_avg_rmse": [round(x, 4) for x in cv_model.avgMetrics],
}
with open(ROOT / "output/metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=2)

print(f"\nSaved output/predictions.csv and output/metrics.json")

# ---------------------------------------------------------------------------
# 8. Publication-quality prediction diagnostic (4 panels)
# ---------------------------------------------------------------------------
# Collect to pandas for plotting
pdf["error"] = pdf["gbt_pred"] - pdf["gdpv_annpct"]
pdf["abs_error"] = pdf["error"].abs()
is_2020 = pdf["year"] == 2020

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

gdp_label = f"{SHORT['gdpv_annpct']} (%)"

# ── Panel A: Predicted vs Actual scatter ────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
ax_a.scatter(
    pdf.loc[~is_2020, "gdpv_annpct"],
    pdf.loc[~is_2020, "gbt_pred"],
    alpha=0.45,
    s=18,
    color="steelblue",
    label="2019–2027 (excl. 2020)",
)
ax_a.scatter(
    pdf.loc[is_2020, "gdpv_annpct"],
    pdf.loc[is_2020, "gbt_pred"],
    alpha=0.85,
    s=40,
    color="crimson",
    label="2020 (COVID)",
    zorder=5,
)

lims = [
    min(pdf["gdpv_annpct"].min(), pdf["gbt_pred"].min()) - 1,
    max(pdf["gdpv_annpct"].max(), pdf["gbt_pred"].max()) + 1,
]
ax_a.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
ax_a.set_xlim(lims)
ax_a.set_ylim(lims)
ax_a.set_xlabel(f"Actual {gdp_label}")
ax_a.set_ylabel(f"Predicted {gdp_label}")
ax_a.set_title(
    f"(A) Predicted vs Actual — GBT\nRMSE={gbt_metrics['rmse']}, R²={gbt_metrics['r2']}"
)
ax_a.legend(fontsize=8)

# ── Panel B: RMSE by year ────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
yearly = (
    pdf.groupby("year")
    .apply(lambda g: np.sqrt((g["error"] ** 2).mean()))
    .reset_index(name="rmse")
)
bar_colors = ["crimson" if y == 2020 else "steelblue" for y in yearly["year"]]
ax_b.bar(yearly["year"], yearly["rmse"], color=bar_colors, edgecolor="white")
ax_b.set_xlabel("Year")
ax_b.set_ylabel("RMSE (pp)")
ax_b.set_title("(B) RMSE by Year\n(red = 2020)")
ax_b.tick_params(axis="x", rotation=45)

# ── Panel C: GBT vs RF scatter ───────────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])
ax_c.scatter(
    pdf.loc[~is_2020, "gdpv_annpct"],
    pdf.loc[~is_2020, "rf_pred"],
    alpha=0.35,
    s=14,
    color="#2ca02c",
    label="RF",
)
ax_c.scatter(
    pdf.loc[~is_2020, "gdpv_annpct"],
    pdf.loc[~is_2020, "gbt_pred"],
    alpha=0.35,
    s=14,
    color="steelblue",
    label="GBT",
)
ax_c.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
ax_c.set_xlim(lims)
ax_c.set_ylim(lims)
ax_c.set_xlabel(f"Actual {gdp_label}")
ax_c.set_ylabel(f"Predicted {gdp_label}")
ax_c.set_title(
    f"(C) GBT vs RF Baseline (excl. 2020)\n"
    f"GBT RMSE={gbt_metrics['rmse']} | RF RMSE={rf_metrics['rmse']}"
)
ax_c.legend(fontsize=8)

# ── Panel D: All 14 feature importances ──────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])
all_names = [x[0] for x in fi_sorted][::-1]  # all 14, ascending for barh
all_vals = [x[1] for x in fi_sorted][::-1]
bars = ax_d.barh(all_names, all_vals, color="steelblue", edgecolor="white", height=0.65)
ax_d.set_xlabel("Importance")
ax_d.set_title("(D) All Feature Importances — GBT")
ax_d.set_xlim(0, max(all_vals) * 1.18)
for bar, val in zip(bars, all_vals):
    ax_d.text(
        val + 0.001,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}",
        va="center",
        fontsize=7,
    )
ax_d.tick_params(axis="y", labelsize=7.5)

plt.suptitle(
    "GDP Nowcast Evaluation — GBT Pipeline\n"
    "OECD Economic Outlook, 38 countries, test period 2019–2027",
    fontsize=13,
    fontweight="bold",
    y=1.01,
)

fig.tight_layout(rect=[0, 0.15, 1, 1])

# Footer: all 6 base variables + lag note + 2020 note
ALL_BASE_VARS = [
    "gdpv_annpct",
    "unr",
    "cbgdpr",
    "itv_annpct",
    "xgsv_annpct",
    "mgsv_annpct",
]
add_footer(
    fig,
    ALL_BASE_VARS,
    extra_notes="For lagged variables, the lag in years is displayed in the figure. "
    "2020 highlighted in red throughout (COVID-19 exogenous shock). "
    "Dashed line = perfect prediction.",
    y_notes=0.03,
)

diag_path = ROOT / "output/prediction_diagnostics.png"
plt.savefig(diag_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {diag_path}")

# ── Feature importance standalone (all 14) ───────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.barh(all_names, all_vals, color="steelblue", edgecolor="white", height=0.65)
ax2.set_xlabel("Importance")
ax2.set_title("Feature Importances — GBT (all features)")
ax2.set_xlim(0, max(all_vals) * 1.18)
for bar, val in zip(ax2.patches, all_vals):
    ax2.text(
        val + 0.001,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}",
        va="center",
        fontsize=8,
    )
ax2.tick_params(axis="y", labelsize=8)
plt.tight_layout(rect=[0, 0.13, 1, 1])
add_footer(
    fig2,
    ALL_BASE_VARS,
    extra_notes="For lagged variables, the lag in years is displayed in the figure.",
    y_notes=0.10,
)
fi_path = ROOT / "output/feature_importance.png"
plt.savefig(fi_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {fi_path}")

# ===========================================================================
# 9. POST-HOC COVID ROBUSTNESS EXERCISE
# ===========================================================================
# This section is an explicit post-hoc diagnostic, not part of the primary
# nowcasting pipeline. It answers two questions:
#   Q1: How much of the negative R² is driven by 2020 alone?
#       → Re-evaluate all models on test \ {2020}.
#   Q2: Does GBT add value over a naive AR(1) benchmark?
#       → Fit OLS: gdpv_annpct ~ gdp_lag1 + country dummies on train,
#         evaluate on test (full and excl. 2020).
# The COVID dummy approach is excluded: it would require knowing ex ante
# that 2020 is a pandemic year, which violates the nowcast premise.
# ===========================================================================

print("\n" + "=" * 70)
print("POST-HOC COVID ROBUSTNESS")
print("=" * 70)

# --- 9a. Collect predictions to pandas (already done: pdf) ------------------
# pdf has columns: country_code, country_name, year, gdpv_annpct,
#                  gbt_pred, rf_pred, error, abs_error
pdf_no2020 = pdf[pdf["year"] != 2020].copy()


def ols_metrics(y_true, y_pred):
    """RMSE, MAE, R² from numpy arrays."""
    resid = y_pred - y_true
    rmse = round(float(np.sqrt((resid**2).mean())), 4)
    mae = round(float(np.abs(resid).mean()), 4)
    ss_res = (resid**2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = round(float(1 - ss_res / ss_tot), 4)
    return {"rmse": rmse, "mae": mae, "r2": r2}


# GBT and RF on test \ {2020}
gbt_no2020 = ols_metrics(
    pdf_no2020["gdpv_annpct"].values, pdf_no2020["gbt_pred"].values
)
rf_no2020 = ols_metrics(pdf_no2020["gdpv_annpct"].values, pdf_no2020["rf_pred"].values)

# --- 9b. AR(1) OLS baseline -------------------------------------------------
# Model: gdpv_annpct_t = α_country + β · gdp_lag1_t + ε
# Country fixed effects via StringIndexer (Spark handles one-hot internally).
# Fit on train Spark DF, evaluate on test Spark DF.

print("\n" + "=" * 60)
print("STEP 5/6: Fitting AR(1) OLS baseline")
print("=" * 60)

train_ar = train.select("gdpv_annpct", "gdp_lag1", "country_code", "year").dropna()
test_ar = test.select("gdpv_annpct", "gdp_lag1", "country_code", "year").dropna()

country_indexer = StringIndexer(
    inputCol="country_code", outputCol="country_idx", handleInvalid="keep"
)
ar_assembler = VectorAssembler(
    inputCols=["gdp_lag1", "country_idx"], outputCol="features"
)

ar_lr = LinearRegression(featuresCol="features", labelCol="gdpv_annpct")
ar_pipeline = Pipeline(stages=[country_indexer, ar_assembler, ar_lr])

print("Fitting AR(1) OLS baseline (Spark)...")
ar_model = ar_pipeline.fit(train_ar)

ar_preds_full = ar_model.transform(test_ar).withColumnRenamed("prediction", "ar1_pred")

ar1_full = eval_metrics(ar_preds_full.withColumnRenamed("ar1_pred", "prediction"))

ar_preds_no2020 = ar_preds_full.filter(F.col("year") != 2020)
ar1_no2020 = eval_metrics(ar_preds_no2020.withColumnRenamed("ar1_pred", "prediction"))

print(f"\nAR(1) OLS — full test:      RMSE={ar1_full['rmse']},  R²={ar1_full['r2']}")
print(f"AR(1) OLS — excl. 2020:     RMSE={ar1_no2020['rmse']}, R²={ar1_no2020['r2']}")
print(f"GBT       — excl. 2020:     RMSE={gbt_no2020['rmse']}, R²={gbt_no2020['r2']}")
print(f"RF        — excl. 2020:     RMSE={rf_no2020['rmse']},  R²={rf_no2020['r2']}")

# --- 9c. Save robustness metrics to metrics.json ----------------------------
robustness = {
    "note": (
        "Post-hoc exercise. Excl-2020 is a diagnostic, not a correction. "
        "AR(1) is OLS with gdp_lag1 + country dummies, fit on train set."
    ),
    "full_test": {"gbt": gbt_metrics, "rf": rf_metrics, "ar1": ar1_full},
    "excl_2020": {"gbt": gbt_no2020, "rf": rf_no2020, "ar1": ar1_no2020},
}
metrics_out["robustness"] = robustness
with open(ROOT / "output/metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=2)
print("\nUpdated output/metrics.json with robustness key.")

# --- 9d. Robustness figure: 2-panel -----------------------------------------
# Panel A: metrics table (3 models × 2 samples) as a styled heatmap-table
# Panel B: AR(1) predicted vs actual scatter (full test, 2020 in red)
# ─────────────────────────────────────────────────────────────────────────────

fig_r, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(14, 6))

# ── Panel A: metrics table ───────────────────────────────────────────────────
# Build a tidy DataFrame for display
table_data = {
    ("Full test\n(2019–2027)", "RMSE"): [
        gbt_metrics["rmse"],
        rf_metrics["rmse"],
        ar1_full["rmse"],
    ],
    ("Full test\n(2019–2027)", "R²"): [
        gbt_metrics["r2"],
        rf_metrics["r2"],
        ar1_full["r2"],
    ],
    ("Excl. 2020\n(post-hoc)", "RMSE"): [
        gbt_no2020["rmse"],
        rf_no2020["rmse"],
        ar1_no2020["rmse"],
    ],
    ("Excl. 2020\n(post-hoc)", "R²"): [
        gbt_no2020["r2"],
        rf_no2020["r2"],
        ar1_no2020["r2"],
    ],
}
tdf = pd.DataFrame(table_data, index=["GBT (CV)", "RF", "AR(1) OLS"])
tdf.columns = pd.MultiIndex.from_tuples(tdf.columns)

# Render as a matplotlib table (no seaborn needed)
ax_t.axis("off")
col_labels = ["Full test RMSE", "Full test R²", "Excl. 2020 RMSE", "Excl. 2020 R²"]
cell_text = [[f"{v:.4f}" for v in row] for row in tdf.values]

tbl = ax_t.table(
    cellText=cell_text,
    rowLabels=tdf.index.tolist(),
    colLabels=col_labels,
    cellLoc="center",
    rowLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.3, 2.2)

# Colour header row
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2c5f8a")
        cell.set_text_props(color="white", fontweight="bold")
    elif c == -1:
        cell.set_facecolor("#e8f0f7")
        cell.set_text_props(fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#f5f9fc")

ax_t.set_title(
    "(A) Model Comparison: Full Test vs Excl. 2020\n"
    "(Post-hoc diagnostic — excl. 2020 not a corrected result)",
    fontsize=10,
    pad=12,
)

# ── Panel B: AR(1) scatter ───────────────────────────────────────────────────
ar_pdf = ar_preds_full.select("year", "gdpv_annpct", "ar1_pred").toPandas()
is_2020_ar = ar_pdf["year"] == 2020

ax_s.scatter(
    ar_pdf.loc[~is_2020_ar, "gdpv_annpct"],
    ar_pdf.loc[~is_2020_ar, "ar1_pred"],
    alpha=0.5,
    s=18,
    color="steelblue",
    label="2019–2027 (excl. 2020)",
)
ax_s.scatter(
    ar_pdf.loc[is_2020_ar, "gdpv_annpct"],
    ar_pdf.loc[is_2020_ar, "ar1_pred"],
    alpha=0.9,
    s=50,
    color="crimson",
    label="2020 (COVID)",
    zorder=5,
)

lims_ar = [
    min(ar_pdf["gdpv_annpct"].min(), ar_pdf["ar1_pred"].min()) - 1,
    max(ar_pdf["gdpv_annpct"].max(), ar_pdf["ar1_pred"].max()) + 1,
]
ax_s.plot(lims_ar, lims_ar, "k--", linewidth=0.8, alpha=0.5)
ax_s.set_xlim(lims_ar)
ax_s.set_ylim(lims_ar)
ax_s.set_xlabel(f"Actual {SHORT['gdpv_annpct']} (%)")
ax_s.set_ylabel(f"Predicted {SHORT['gdpv_annpct']} (%)")
ax_s.set_title(
    f"(B) AR(1) OLS: Predicted vs Actual\n"
    f"RMSE={ar1_full['rmse']}, R²={ar1_full['r2']} (full test)"
)
ax_s.legend(fontsize=8)

plt.suptitle(
    "Post-hoc COVID Robustness — GDP Nowcasting\n"
    "OECD Economic Outlook, 38 countries, test period 2019–2027",
    fontsize=12,
    fontweight="bold",
    y=1.02,
)

fig_r.tight_layout(rect=[0, 0.13, 1, 1])
add_footer(
    fig_r,
    ["gdpv_annpct", "gdp_lag1"],
    extra_notes=(
        "AR(1) OLS: gdpv_annpct ~ gdp_lag1 + country fixed effects "
        "(one-hot dummies, trained on pre-2019 data). "
        "Excl. 2020 rows are removed from evaluation only — "
        "model was not retrained. 2020 in red throughout."
    ),
    y_notes=0.10,
)

rob_path = ROOT / "output/robustness_covid.png"
plt.savefig(rob_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {rob_path}")

spark.stop()
print("\nPhase 4 complete.")
