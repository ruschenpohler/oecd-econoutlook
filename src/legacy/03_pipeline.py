"""
Phase 3: Spark ML Pipeline Assembly
====================================
Builds a full ML pipeline:
  StringIndexer → VectorAssembler → StandardScaler → GBTRegressor

Key design choices:
- Time-aware train/test split (train < 2019, test >= 2019), NOT randomSplit,
  to prevent future data leaking into training via random shuffling.
- Pipeline wraps all preprocessing + model so that fit() never sees test data —
  even the scaler is fitted on train only.
- Country fixed effects encoded via StringIndexer (learns string→index mapping
  from training data; handleInvalid="keep" for unseen countries at test time).
"""

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor
import os
from pathlib import Path

# Resolve paths relative to this script, not the caller's working directory.
# This makes %run from notebooks/ work identically to running from project root.
ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# 1. SparkSession
# ---------------------------------------------------------------------------
import os as _os
_os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

spark = (
    SparkSession.builder.master("local[*]")
    .appName("OECD-GDP-Nowcast-Phase3")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.driver.host", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# ---------------------------------------------------------------------------
# 2. Load feature-engineered data
# ---------------------------------------------------------------------------
df = spark.read.csv(str(ROOT / "data/features.csv"), header=True, inferSchema=True)
print(f"Loaded features: {df.count()} rows × {len(df.columns)} columns")

# ---------------------------------------------------------------------------
# 3. Define feature columns
# ---------------------------------------------------------------------------
# Lag-1 and lag-2 for 5 macro variables
MACRO_VARS = ["unr", "cbgdpr", "itv_annpct", "xgsv_annpct", "mgsv_annpct"]
lag_cols = [f"{v}_lag{k}" for v in MACRO_VARS for k in [1, 2]]

# Autoregressive terms for GDP
ar_cols = ["gdp_lag1", "gdp_lag2", "gdp_accel"]

# Country index (from StringIndexer) — added at pipeline runtime
# Total feature vector: 1 + 3 + 10 = 14 dimensions
FEATURE_COLS = ["country_idx"] + ar_cols + lag_cols

print(f"\nFeature columns ({len(FEATURE_COLS)} total):")
for col in FEATURE_COLS:
    print(f"  {col}")

# ---------------------------------------------------------------------------
# 4. Time-aware train/test split
#
# Why NOT randomSplit?
# GDP is a time series. A random split would place 2021 data in training and
# 2019 data in test — the model would "know" about future macro conditions
# while being evaluated on the past. This inflates R² and is not representative
# of real nowcasting conditions.
#
# We hold out 2019–2027 as test (post-GFC recovery → COVID → post-COVID).
# ---------------------------------------------------------------------------
train = df.filter(F.col("year") < 2019)
test = df.filter(F.col("year") >= 2019)

print(
    f"\nTrain: {train.count()} rows ({train.agg(F.min('year')).first()[0]}–{train.agg(F.max('year')).first()[0]})"
)
print(
    f"Test:  {test.count()} rows ({test.agg(F.min('year')).first()[0]}–{test.agg(F.max('year')).first()[0]})"
)

# ---------------------------------------------------------------------------
# 5. Pipeline stages
#
# Stage 1 — StringIndexer: country_code (string) → country_idx (double)
#   Estimator: learns the mapping from training data.
#   handleInvalid="keep" assigns a catch-all index to unseen countries at test time.
#
# Stage 2 — VectorAssembler: 14 columns → "features_raw" (DenseVector)
#   Transformer: no fitting needed, just concatenation.
#   handleInvalid="skip" drops rows where any input is null (residual missingness).
#
# Stage 3 — StandardScaler: "features_raw" → "features" (z-scored DenseVector)
#   Estimator: computes std on training data only.
#   withMean=False avoids densifying sparse vectors (not critical here, good habit).
#   withStd=True normalises scale — important for GBT splits to be scale-invariant.
#
# Stage 4 — GBTRegressor: fits gradient boosted trees on "features", predicts "gdpv_annpct"
#   Estimator: returns a GBTRegressionModel (Transformer) after fitting.
# ---------------------------------------------------------------------------
indexer = StringIndexer(
    inputCol="country_code", outputCol="country_idx", handleInvalid="keep"
)

assembler = VectorAssembler(
    inputCols=FEATURE_COLS, outputCol="features_raw", handleInvalid="skip"
)

scaler = StandardScaler(
    inputCol="features_raw", outputCol="features", withStd=True, withMean=False
)

gbt = GBTRegressor(
    featuresCol="features", labelCol="gdpv_annpct", maxIter=100, maxDepth=4, seed=42
)

pipeline = Pipeline(stages=[indexer, assembler, scaler, gbt])

# ---------------------------------------------------------------------------
# 6. Fit the pipeline on training data
#    Only train data flows through .fit() — the scaler never sees test years.
# ---------------------------------------------------------------------------
print("\nFitting pipeline on training data...")
model = pipeline.fit(train)
print("Pipeline fitted.")

# ---------------------------------------------------------------------------
# 7. Quick sanity check on test set
# ---------------------------------------------------------------------------
predictions = model.transform(test)
print(f"\nTest set predictions: {predictions.count()} rows")
print("Sample predictions (DEU):")
predictions.filter("country_code = 'DEU'").select(
    "country_code", "year", "gdpv_annpct", F.round("prediction", 2).alias("predicted")
).orderBy("year").show()

# ---------------------------------------------------------------------------
# 8. Persist for Phase 4
#    Save train/test splits as CSV. Model serialization (model.write().save())
#    is not supported in Spark local mode on Windows — Hadoop's filesystem layer
#    requires a Linux environment. Phase 4 re-fits from data/train.csv instead.
# ---------------------------------------------------------------------------
os.makedirs(ROOT / "output", exist_ok=True)

train.toPandas().to_csv(ROOT / "data/train.csv", index=False)
test.toPandas().to_csv(ROOT / "data/test.csv", index=False)

print("\nSaved:")
print(f"  {ROOT / 'data/train.csv'}  - training split")
print(f"  {ROOT / 'data/test.csv'}   - test split")
print("  (model serialization skipped: not supported in Spark local mode on Windows)")

# SparkSession stays alive — notebook reuses it via getOrCreate()
