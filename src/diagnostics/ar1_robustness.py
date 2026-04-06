"""
AR(1) robustness — standalone script.
Loads existing predictions.csv (GBT + RF), fits AR(1) with proper country
fixed effects (StringIndexer → OneHotEncoder → LinearRegression) on the
same train/test split used in Phase 4, and prints the full 4-cut × 3-model
robustness table to stdout.

Run: uv run python src/ar1_robustness.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

ROOT = Path(__file__).resolve().parent.parent

import os as _os
_os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

spark = (
    SparkSession.builder.master("local[*]")
    .appName("AR1-Robustness")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.driver.host", "127.0.0.1")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
features_csv = str(ROOT / "data/features.csv")
sdf = spark.read.csv(features_csv, header=True, inferSchema=True)

train_sdf = sdf.filter(F.col("year") < 2019)
test_sdf  = sdf.filter(F.col("year") >= 2019)

print(f"Train: {train_sdf.count()} rows | Test: {test_sdf.count()} rows")

# ---------------------------------------------------------------------------
# AR(1) pipeline: StringIndexer → OneHotEncoder → VectorAssembler → OLS
# OHE is required — StringIndexer alone gives ordinal integers which OLS
# treats as a continuous country variable (wrong). OHE gives proper binary
# indicators = country fixed effects.
# ---------------------------------------------------------------------------
country_indexer = StringIndexer(
    inputCol="country_code", outputCol="country_idx", handleInvalid="keep"
)
country_ohe = OneHotEncoder(
    inputCol="country_idx", outputCol="country_ohe",
    handleInvalid="keep",
    dropLast=True,   # avoid perfect multicollinearity; one country is baseline
)
ar_assembler = VectorAssembler(
    inputCols=["gdp_lag1", "country_ohe"],
    outputCol="features",
    handleInvalid="skip",
)
ar_lr = LinearRegression(
    featuresCol="features",
    labelCol="gdpv_annpct",
    maxIter=200,
)
ar_pipeline = Pipeline(stages=[country_indexer, country_ohe, ar_assembler, ar_lr])

print("\nFitting AR(1) with country fixed effects (OLS)...")
ar_model = ar_pipeline.fit(train_sdf)

ar_preds_sdf = ar_model.transform(test_sdf).withColumnRenamed("prediction", "ar1_pred")

# ---------------------------------------------------------------------------
# Helper: metrics from numpy arrays
# ---------------------------------------------------------------------------
def metrics(y, yhat):
    resid = y - yhat
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {
        "rmse": round(float(np.sqrt(np.mean(resid ** 2))), 4),
        "mae":  round(float(np.mean(np.abs(resid))), 4),
        "r2":   round(float(1 - ss_res / ss_tot), 4),
        "n":    int(len(y)),
    }

# ---------------------------------------------------------------------------
# Load GBT + RF predictions and merge AR(1)
# ---------------------------------------------------------------------------
pred_csv = pd.read_csv(ROOT / "output/predictions.csv")
ar_pdf   = ar_preds_sdf.select("country_code", "year", "ar1_pred").toPandas()

pdf = pred_csv.merge(ar_pdf, on=["country_code", "year"], how="inner")
print(f"Merged prediction frame: {len(pdf)} rows")

# ---------------------------------------------------------------------------
# Robustness cuts
# ---------------------------------------------------------------------------
COVID_YEARS      = [2020]
COVID_21_YEARS   = [2020, 2021]
WEIRDNESS_YEARS  = [2020, 2021, 2022, 2023]

cuts = {
    "full_test":    pdf,
    "excl_2020":    pdf[~pdf["year"].isin(COVID_YEARS)],
    "excl_2020_21": pdf[~pdf["year"].isin(COVID_21_YEARS)],
    "excl_2020_23": pdf[~pdf["year"].isin(WEIRDNESS_YEARS)],
}

LABELS = {
    "full_test":    "Full test (2019–27)",
    "excl_2020":    "Excl. 2020",
    "excl_2020_21": "Excl. 2020–21",
    "excl_2020_23": "Excl. 2020–23",
}

# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"{'Cut':<22}  {'Model':<6}  {'RMSE':<8}  {'MAE':<8}  {'R2':<9}  {'n'}")
print("-" * 80)

results = {}
for cut_key, df in cuts.items():
    results[cut_key] = {}
    for model, col in [("GBT", "gbt_pred"), ("RF", "rf_pred"), ("AR1", "ar1_pred")]:
        m = metrics(df["gdpv_annpct"].values, df[col].values)
        results[cut_key][model] = m
        label = LABELS[cut_key] if model == "GBT" else ""
        print(f"{label:<22}  {model:<6}  {m['rmse']:<8}  {m['mae']:<8}  {m['r2']:<9}  {m['n']}")
    print()

# ---------------------------------------------------------------------------
# Summary: who wins each cut?
# ---------------------------------------------------------------------------
print("=" * 80)
print("WINNER BY CUT (lowest RMSE):")
for cut_key in cuts:
    best_model = min(results[cut_key], key=lambda m: results[cut_key][m]["rmse"])
    best_rmse  = results[cut_key][best_model]["rmse"]
    print(f"  {LABELS[cut_key]:<22}  →  {best_model}  (RMSE={best_rmse})")

spark.stop()
print("\nDone.")
