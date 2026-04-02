"""
Phase 4: Evaluation & Interpretation
======================================
Loads train/test splits, re-fits the GBT pipeline, fits an RF baseline,
runs CrossValidator for hyperparameter tuning, evaluates both models,
extracts feature importances, and saves all outputs.

Outputs
-------
  output/predictions.csv        — country, year, actual, gbt_pred, rf_pred
  output/metrics.json           — RMSE/MAE/R² for both models, feature importances,
                                  best CV hyperparameters
  output/prediction_diagnostics.png — publication-quality 4-panel diagnostic figure
  output/feature_importance.png      — top-10 feature importances (GBT)
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
spark = (SparkSession.builder
    .master("local[*]")
    .appName("OECD-GDP-Nowcast-Phase4")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate())

spark.sparkContext.setLogLevel("WARN")

# ---------------------------------------------------------------------------
# 2. Load data and define pipeline stages
# ---------------------------------------------------------------------------
train = spark.read.csv(str(ROOT / "data/train.csv"), header=True, inferSchema=True)
test  = spark.read.csv(str(ROOT / "data/test.csv"),  header=True, inferSchema=True)

print(f"Train: {train.count()} rows | Test: {test.count()} rows")

MACRO_VARS   = ["unr", "cbgdpr", "itv_annpct", "xgsv_annpct", "mgsv_annpct"]
lag_cols     = [f"{v}_lag{k}" for v in MACRO_VARS for k in [1, 2]]
FEATURE_COLS = ["country_idx"] + ["gdp_lag1", "gdp_lag2", "gdp_accel"] + lag_cols
FEATURE_NAMES = [SHORT.get(c, c) for c in FEATURE_COLS]

indexer   = StringIndexer(inputCol="country_code", outputCol="country_idx", handleInvalid="keep")
assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features_raw", handleInvalid="skip")
scaler    = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=False)

# ---------------------------------------------------------------------------
# 3. GBT pipeline + CrossValidator
# ---------------------------------------------------------------------------
gbt = GBTRegressor(featuresCol="features", labelCol="gdpv_annpct",
                   maxIter=100, maxDepth=4, seed=42)
gbt_pipeline = Pipeline(stages=[indexer, assembler, scaler, gbt])

param_grid = (ParamGridBuilder()
    .addGrid(gbt.maxDepth, [3, 5])
    .addGrid(gbt.stepSize, [0.05, 0.1])
    .build())

evaluator = RegressionEvaluator(labelCol="gdpv_annpct", predictionCol="prediction",
                                metricName="rmse")

cv = CrossValidator(estimator=gbt_pipeline, estimatorParamMaps=param_grid,
                    evaluator=evaluator, numFolds=3, seed=42)

print("Fitting GBT with CrossValidator (4 param combos × 3 folds)...")
cv_model = cv.fit(train)
best_gbt  = cv_model.bestModel

print(f"Best maxDepth: {best_gbt.stages[-1].getOrDefault('maxDepth')}")
print(f"Best stepSize: {best_gbt.stages[-1].getOrDefault('stepSize')}")
print(f"CV avg RMSE per combo: {[round(x, 3) for x in cv_model.avgMetrics]}")

# ---------------------------------------------------------------------------
# 4. Random Forest baseline
# ---------------------------------------------------------------------------
rf = RandomForestRegressor(featuresCol="features", labelCol="gdpv_annpct",
                           numTrees=100, maxDepth=5, seed=42)
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
            RegressionEvaluator(labelCol="gdpv_annpct", predictionCol="prediction",
                                metricName=metric).evaluate(predictions_df), 4)
    return out

gbt_preds = best_gbt.transform(test).withColumnRenamed("prediction", "gbt_pred")
rf_preds  = rf_model.transform(test).withColumnRenamed("prediction", "rf_pred")

# Join on (country_code, year)
preds_joined = (gbt_preds
    .select("country_code", "country_name", "year", "gdpv_annpct", "gbt_pred")
    .join(rf_preds.select("country_code", "year", "rf_pred"), on=["country_code", "year"]))

gbt_metrics = eval_metrics(gbt_preds.withColumnRenamed("gbt_pred", "prediction"))
rf_metrics  = eval_metrics(rf_preds.withColumnRenamed("rf_pred", "prediction"))

print(f"\nGBT  — RMSE: {gbt_metrics['rmse']}, MAE: {gbt_metrics['mae']}, R²: {gbt_metrics['r2']}")
print(f"RF   — RMSE: {rf_metrics['rmse']},  MAE: {rf_metrics['mae']},  R²: {rf_metrics['r2']}")

# ---------------------------------------------------------------------------
# 6. Feature importances
# ---------------------------------------------------------------------------
fi_vec   = best_gbt.stages[-1].featureImportances
fi_dict  = {FEATURE_NAMES[i]: round(float(fi_vec[i]), 4)
            for i in range(len(FEATURE_COLS))}
fi_sorted = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)

print("\nTop 10 feature importances (GBT):")
for name, imp in fi_sorted[:10]:
    print(f"  {name:30s} {imp:.4f}")

# ---------------------------------------------------------------------------
# 7. Save outputs
# ---------------------------------------------------------------------------
os.makedirs(ROOT / "output", exist_ok=True)

pdf = preds_joined.toPandas().sort_values(["country_code", "year"])
pdf.to_csv(ROOT / "output/predictions.csv", index=False)

metrics_out = {
    "gbt": gbt_metrics,
    "rf":  rf_metrics,
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
pdf["error"]    = pdf["gbt_pred"] - pdf["gdpv_annpct"]
pdf["abs_error"] = pdf["error"].abs()
is_2020 = pdf["year"] == 2020

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

gdp_label = f"{SHORT['gdpv_annpct']} (%)"

# ── Panel A: Predicted vs Actual scatter ────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
ax_a.scatter(pdf.loc[~is_2020, "gdpv_annpct"], pdf.loc[~is_2020, "gbt_pred"],
             alpha=0.45, s=18, color="steelblue", label="2019–2027 (excl. 2020)")
ax_a.scatter(pdf.loc[is_2020, "gdpv_annpct"],  pdf.loc[is_2020, "gbt_pred"],
             alpha=0.85, s=40, color="crimson",  label="2020 (COVID)", zorder=5)

lims = [min(pdf["gdpv_annpct"].min(), pdf["gbt_pred"].min()) - 1,
        max(pdf["gdpv_annpct"].max(), pdf["gbt_pred"].max()) + 1]
ax_a.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
ax_a.set_xlim(lims); ax_a.set_ylim(lims)
ax_a.set_xlabel(f"Actual {gdp_label}")
ax_a.set_ylabel(f"Predicted {gdp_label}")
ax_a.set_title(f"(A) Predicted vs Actual — GBT\nRMSE={gbt_metrics['rmse']}, R²={gbt_metrics['r2']}")
ax_a.legend(fontsize=8)

# ── Panel B: RMSE by year ────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
yearly = (pdf.groupby("year")
          .apply(lambda g: np.sqrt((g["error"] ** 2).mean()))
          .reset_index(name="rmse"))
bar_colors = ["crimson" if y == 2020 else "steelblue" for y in yearly["year"]]
ax_b.bar(yearly["year"], yearly["rmse"], color=bar_colors, edgecolor="white")
ax_b.set_xlabel("Year")
ax_b.set_ylabel("RMSE (pp)")
ax_b.set_title("(B) RMSE by Year\n(red = 2020)")
ax_b.tick_params(axis="x", rotation=45)

# ── Panel C: GBT vs RF scatter ───────────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])
ax_c.scatter(pdf.loc[~is_2020, "gdpv_annpct"], pdf.loc[~is_2020, "rf_pred"],
             alpha=0.35, s=14, color="#2ca02c", label="RF")
ax_c.scatter(pdf.loc[~is_2020, "gdpv_annpct"], pdf.loc[~is_2020, "gbt_pred"],
             alpha=0.35, s=14, color="steelblue", label="GBT")
ax_c.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
ax_c.set_xlim(lims); ax_c.set_ylim(lims)
ax_c.set_xlabel(f"Actual {gdp_label}")
ax_c.set_ylabel(f"Predicted {gdp_label}")
ax_c.set_title(f"(C) GBT vs RF Baseline (excl. 2020)\n"
               f"GBT RMSE={gbt_metrics['rmse']} | RF RMSE={rf_metrics['rmse']}")
ax_c.legend(fontsize=8)

# ── Panel D: All 14 feature importances ──────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])
all_names = [x[0] for x in fi_sorted][::-1]   # all 14, ascending for barh
all_vals  = [x[1] for x in fi_sorted][::-1]
bars = ax_d.barh(all_names, all_vals, color="steelblue", edgecolor="white", height=0.65)
ax_d.set_xlabel("Importance")
ax_d.set_title("(D) All Feature Importances — GBT")
ax_d.set_xlim(0, max(all_vals) * 1.18)
for bar, val in zip(bars, all_vals):
    ax_d.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
              f"{val:.3f}", va="center", fontsize=7)
ax_d.tick_params(axis="y", labelsize=7.5)

plt.suptitle("GDP Nowcast Evaluation — GBT Pipeline\n"
             "OECD Economic Outlook, 38 countries, test period 2019–2027",
             fontsize=13, fontweight="bold", y=1.01)

fig.tight_layout(rect=[0, 0.09, 1, 1])

# Footer: all 6 base variables + lag note + 2020 note
ALL_BASE_VARS = ["gdpv_annpct", "unr", "cbgdpr", "itv_annpct", "xgsv_annpct", "mgsv_annpct"]
add_footer(fig, ALL_BASE_VARS,
           extra_notes="For lagged variables, the lag in years is displayed in the figure. "
                       "2020 highlighted in red throughout (COVID-19 exogenous shock). "
                       "Dashed line = perfect prediction.",
           y_notes=0.04, y_source=0.01)

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
    ax2.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", fontsize=8)
ax2.tick_params(axis="y", labelsize=8)
plt.tight_layout(rect=[0, 0.09, 1, 1])
add_footer(fig2, ALL_BASE_VARS,
           extra_notes="For lagged variables, the lag in years is displayed in the figure.",
           y_notes=0.04, y_source=0.01)
fi_path = ROOT / "output/feature_importance.png"
plt.savefig(fi_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {fi_path}")

spark.stop()
print("\nPhase 4 complete.")
