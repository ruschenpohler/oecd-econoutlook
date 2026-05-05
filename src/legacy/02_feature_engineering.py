"""
02_feature_engineering.py — Create lagged features in PySpark.

Reads the cleaned OECD Economic Outlook panel from Phase 1, creates lag-1
and lag-2 features for all macro variables plus autoregressive GDP terms,
drops rows with null lags, and saves to data/features.csv.

This is where Spark starts: SparkSession, Window functions, F.lag().
"""

import os
from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "oecd_economic_outlook.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "features.csv")

# ---------------------------------------------------------------------------
# Macro variables to lag (everything except the target and identifiers).
# The target (gdpv_annpct) gets its own autoregressive lags separately.
# ---------------------------------------------------------------------------
MACRO_VARS = ["unr", "cbgdpr", "itv_annpct", "xgsv_annpct", "mgsv_annpct"]


def main():
    # ------------------------------------------------------------------
    # 1. Initialize SparkSession
    # ------------------------------------------------------------------
    # local[*] = use all CPU cores as simulated cluster workers.
    # In production this would be .master("yarn") or similar.
    # spark.sql.shuffle.partitions = 8: default is 200, which is overkill
    # for a ~1,400-row dataset and creates unnecessary overhead.
    import os as _os

    _os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("OECD-GDP-Nowcast-FeatureEng")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )

    # ------------------------------------------------------------------
    # 2. Read the cleaned CSV into a Spark DataFrame
    # ------------------------------------------------------------------
    # inferSchema=True tells Spark to scan the data and auto-detect column
    # types (string, integer, double, etc.). On big data you'd specify the
    # schema explicitly to avoid a full scan, but at 1,400 rows it's fine.
    df = spark.read.csv(INPUT_PATH, header=True, inferSchema=True)
    raw_count = df.count()
    n_countries = df.select("country_code").distinct().count()
    print(
        f"Loaded: {raw_count:,} rows, {n_countries} countries, "
        f"{df.select('year').distinct().count()} years"
    )
    print(f"Columns: {df.columns}\n")

    # ------------------------------------------------------------------
    # 3. Define the Window specification
    # ------------------------------------------------------------------
    # partitionBy("country_code"): each country is an independent group.
    # orderBy("year"): within each country, rows are ordered chronologically.
    # This is the Spark equivalent of pandas groupby + shift.
    w = Window.partitionBy("country_code").orderBy("year")

    # ------------------------------------------------------------------
    # 4. Create lag features
    # ------------------------------------------------------------------

    # 4a. Autoregressive terms: lag-1 and lag-2 of the target variable.
    # These let the model use recent GDP growth to predict current GDP growth.
    df = df.withColumn("gdp_lag1", F.lag("gdpv_annpct", 1).over(w))
    df = df.withColumn("gdp_lag2", F.lag("gdpv_annpct", 2).over(w))

    # 4b. GDP acceleration: change in GDP growth rate.
    # Positive = growth is accelerating; negative = decelerating.
    # This captures momentum — is the economy speeding up or slowing down?
    df = df.withColumn("gdp_accel", F.col("gdp_lag1") - F.col("gdp_lag2"))

    # 4c. Lag-1 and lag-2 for each macro variable.
    # We use lagged (not contemporaneous) values because in a nowcasting
    # context, you only have access to past observations at prediction time.
    for var in MACRO_VARS:
        df = df.withColumn(f"{var}_lag1", F.lag(var, 1).over(w))
        df = df.withColumn(f"{var}_lag2", F.lag(var, 2).over(w))

    # ------------------------------------------------------------------
    # 5. Drop rows with null lags
    # ------------------------------------------------------------------
    # The first 2 years of each country's series will have null lag-2 values.
    # Additionally, any residual missingness from the original data (late-start
    # countries) will produce nulls in the lag columns.
    # dropna() removes any row with at least one null in any column.
    df = df.dropna()

    final_count = df.count()
    dropped = raw_count - final_count
    print(
        f"Dropped {dropped} rows with null values "
        f"(expected ≥ {2 * n_countries} from lag-induced nulls)"
    )
    print(f"Final: {final_count:,} rows × {len(df.columns)} columns")

    # Print the schema so we can see all the new columns and their types
    print("\nSchema:")
    df.printSchema()

    # ------------------------------------------------------------------
    # 6. Save to CSV via pandas
    # ------------------------------------------------------------------
    # The dataset is small enough (~1,200 rows) to collect into local memory.
    # On a large dataset you'd use df.write.parquet() instead.
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.toPandas().to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")

    # SparkSession stays alive — notebook reuses it via getOrCreate()


if __name__ == "__main__":
    main()
