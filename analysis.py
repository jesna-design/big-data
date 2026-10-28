"""
advanced_analysis_pyspark.py
----------------------------
PySpark version of the Ola Mobility advanced analytics layer.

  1. Area x Hour demand heatmap                 (Spark pivot -> seaborn)
  2. K-Means segmentation of the pickup areas   (Spark ML KMeans + silhouette + PCA)
  3. Chi-square test: is "peak hour" real?      (Spark aggregation -> scipy)
  4. Cancellation-risk model                    (Spark ML RandomForest, ROC-AUC)

Requirements:  pip install pyspark pandas numpy matplotlib seaborn scipy
               Java 8 / 11 / 17 installed (JAVA_HOME set)
Run:           python advanced_analysis_pyspark.py
Outputs:       heatmap_area_hour.png, area_clusters.png, feature_importance.png,
               area_segments.csv
"""
import os
import sys

# make Spark workers use the same Python interpreter as this script (avoids version errors)
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chisquare

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import (VectorAssembler, StandardScaler, PCA,
                                StringIndexer, OneHotEncoder)
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator, BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "ola_bengaluru.csv")

# ------------------------------------------------------------------ SPARK
spark = (SparkSession.builder
         .appName("OlaMobilityAdvancedAnalysis")
         .master("local[*]")
         .config("spark.sql.shuffle.partitions", "8")
         .config("spark.sql.ansi.enabled", "false")   # bad values -> null instead of crash
         .config("spark.ui.showConsoleProgress", "false")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

# ------------------------------------------------------------------ LOAD
raw = (spark.read
       .option("header", True)
       .option("inferSchema", True)
       .csv(CSV_PATH))

df = (raw
      .withColumn("Date", F.to_date(F.col("Date"), "d/M/yyyy"))
      # Time looks like "6:00:00" -> take the part before the first colon
      .withColumn("Hour", F.split(F.col("Time"), ":").getItem(0).cast("int"))
      # Spark dayofweek: 1=Sunday..7=Saturday  ->  pandas style 0=Monday..6=Sunday
      .withColumn("DayOfWeek", (F.dayofweek("Date") + 5) % 7)
      .withColumn("IsWeekend", F.when(F.col("DayOfWeek") >= 5, 1).otherwise(0))
      .withColumn("Cancelled",
                  F.when(F.col("Booking Status").startswith("Cancelled"), 1).otherwise(0))
      .withColumn("Success",
                  F.when(F.col("Booking Status") == "Success", 1).otherwise(0))
      .dropna(subset=["Date", "Hour", "Pickup Location"])
      .cache())

n_rows = df.count()
dmin, dmax = df.agg(F.min("Date"), F.max("Date")).first()
print(f"Loaded {n_rows:,} bookings from {dmin} to {dmax}")

# ------------------------------------------------- 1. AREA x HOUR HEATMAP
hours = list(range(24))
pivot_sdf = (df.groupBy("Pickup Location")
             .pivot("Hour", hours)
             .count()
             .na.fill(0))
pivot = pivot_sdf.toPandas().set_index("Pickup Location")
pivot.columns = [int(c) for c in pivot.columns]
pivot = pivot[sorted(pivot.columns)]
order = pivot.sum(axis=1).sort_values(ascending=False).index

plt.figure(figsize=(14, 10))
sns.heatmap(pivot.loc[order], cmap="YlOrRd", cbar_kws={"label": "Bookings"})
plt.title("Demand Heatmap: Pickup Area vs Hour of Day")
plt.xlabel("Hour")
plt.tight_layout()
plt.savefig("heatmap_area_hour.png", dpi=150)
plt.close()
print("Saved heatmap_area_hour.png")

# ------------------------------------- 2. K-MEANS SEGMENTATION OF AREAS
area_sdf = (df.groupBy("Pickup Location")
            .agg(F.count("*").alias("Total_Bookings"),
                 F.avg("Booking Value").alias("Avg_Fare"),
                 F.avg("Cancelled").alias("Cancel_Rate"),
                 F.avg("Success").alias("Success_Rate"),
                 F.avg("Ride Distance").alias("Avg_Distance"),
                 F.avg("Driver Ratings").alias("Avg_Rating"))
            .dropna())

features = ["Total_Bookings", "Avg_Fare", "Cancel_Rate", "Avg_Distance", "Avg_Rating"]

assembler = VectorAssembler(inputCols=features, outputCol="features")
assembled = assembler.transform(area_sdf)
scaler_model = StandardScaler(inputCol="features", outputCol="scaled",
                              withMean=True, withStd=True).fit(assembled)
scaled = scaler_model.transform(assembled).cache()

# choose k by silhouette score rather than guessing
evaluator = ClusteringEvaluator(featuresCol="scaled", predictionCol="prediction",
                                metricName="silhouette",
                                distanceMeasure="squaredEuclidean")
scores = {}
for k in range(2, 7):
    preds = KMeans(k=k, seed=42, featuresCol="scaled", maxIter=100).fit(scaled).transform(scaled)
    scores[k] = evaluator.evaluate(preds)
best_k = max(scores, key=scores.get)
print("Silhouette scores:", {k: round(v, 3) for k, v in scores.items()}, "-> k =", best_k)

km_model = KMeans(k=best_k, seed=42, featuresCol="scaled", maxIter=100).fit(scaled)
clustered = km_model.transform(scaled).withColumnRenamed("prediction", "Cluster")

pca_model = PCA(k=2, inputCol="scaled", outputCol="pca").fit(clustered)
clustered = pca_model.transform(clustered)

# collect the (tiny) 50-row result to pandas; convert Spark vectors to plain floats
rows = clustered.select("Pickup Location", *features, "Success_Rate", "Cluster", "pca").collect()
area = pd.DataFrame([{
    "Pickup Location": r["Pickup Location"],
    "Total_Bookings": r["Total_Bookings"],
    "Avg_Fare": r["Avg_Fare"],
    "Cancel_Rate": r["Cancel_Rate"],
    "Success_Rate": r["Success_Rate"],
    "Avg_Distance": r["Avg_Distance"],
    "Avg_Rating": r["Avg_Rating"],
    "Cluster": int(r["Cluster"]),
    "PC1": float(r["pca"][0]),
    "PC2": float(r["pca"][1]),
} for r in rows]).sort_values("Pickup Location").reset_index(drop=True)

# human-readable tier names, ranked by a simple value score (demand + fare - cancellations)
g = area.groupby("Cluster")[["Total_Bookings", "Avg_Fare", "Cancel_Rate"]].mean()
value = (g["Total_Bookings"] / area["Total_Bookings"].mean()
         + g["Avg_Fare"] / area["Avg_Fare"].mean()
         - g["Cancel_Rate"] / area["Cancel_Rate"].mean()).sort_values(ascending=False)
labels = ["Premium Hotspot", "Steady Performer", "Watch List", "Low Priority",
          "Underperformer", "Weak Zone"]
name_map = {c: labels[i] for i, c in enumerate(value.index)}
area["Segment"] = area["Cluster"].map(name_map)

out_cols = ["Pickup Location", "Total_Bookings", "Avg_Fare", "Cancel_Rate", "Success_Rate",
            "Avg_Distance", "Avg_Rating", "Cluster", "Segment"]
area[out_cols].round(3).to_csv("area_segments.csv", index=False)
print(area.groupby("Segment")[features].mean().round(2))

plt.figure(figsize=(9, 6))
sns.scatterplot(x=area["PC1"], y=area["PC2"], hue=area["Segment"], s=110, palette="Set2")
for _, r in area.iterrows():
    plt.annotate(r["Pickup Location"].replace("Area-", ""), (r["PC1"], r["PC2"]),
                 fontsize=7, xytext=(3, 3), textcoords="offset points")
plt.title(f"K-Means Segmentation of Pickup Areas (k={best_k}, PCA view)")
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.tight_layout()
plt.savefig("area_clusters.png", dpi=150)
plt.close()
print("Saved area_clusters.png and area_segments.csv")

# ------------------------- 3. IS THE "PEAK HOUR" STATISTICALLY REAL?
hourly_pd = (df.groupBy("Hour").count().orderBy("Hour").toPandas()
             .set_index("Hour")["count"])
chi2, p = chisquare(hourly_pd.values)  # H0: demand is equal in every hour
spread = (hourly_pd.max() - hourly_pd.min()) / hourly_pd.mean() * 100
print(f"\nChi-square test of uniform hourly demand: chi2={chi2:.1f}, p={p:.4f}")
print(f"Busiest hour: {hourly_pd.idxmax()}:00 ({hourly_pd.max()}), quietest: "
      f"{hourly_pd.idxmin()}:00 ({hourly_pd.min()}); spread = {spread:.1f}% of the mean")
print("-> Demand is "
      + ("NOT uniform (peaks are statistically significant)" if p < 0.05
         else "statistically flat: peak hours are real but small / close to random variation"))

# ------------------------ 4. CANCELLATION-RISK MODEL (Random Forest)
model_sdf = (df.select("Hour", "DayOfWeek", "IsWeekend", "Vehicle Type",
                       "Pickup Location", "Cancelled")
             .dropna()
             .withColumn("Area_ID",
                         F.regexp_replace("Pickup Location", "Area-", "").cast("int"))
             .withColumn("label", F.col("Cancelled").cast("double"))
             .dropna(subset=["Area_ID"]))

# one-hot encode Vehicle Type (fit on all data so no unseen category appears at test time)
idx_model = StringIndexer(inputCol="Vehicle Type", outputCol="vt_idx").fit(model_sdf)
vehicle_labels = list(idx_model.labels)
model_sdf = idx_model.transform(model_sdf)
ohe_model = OneHotEncoder(inputCols=["vt_idx"], outputCols=["vt_vec"],
                          dropLast=False).fit(model_sdf)
model_sdf = ohe_model.transform(model_sdf)

feature_names = ["Hour", "DayOfWeek", "IsWeekend", "Area_ID"] + \
                [f"Vehicle Type_{v}" for v in vehicle_labels]
ml_sdf = (VectorAssembler(inputCols=["Hour", "DayOfWeek", "IsWeekend", "Area_ID", "vt_vec"],
                          outputCol="features")
          .transform(model_sdf)
          .select("features", "label"))

# class weights (equivalent of class_weight="balanced")
total = ml_sdf.count()
cnt = {r["label"]: r["count"] for r in ml_sdf.groupBy("label").count().collect()}
w = {lab: total / (2.0 * c) for lab, c in cnt.items()}
ml_sdf = ml_sdf.withColumn(
    "weight", F.when(F.col("label") == 1.0, w.get(1.0, 1.0)).otherwise(w.get(0.0, 1.0)))

train, test = ml_sdf.randomSplit([0.75, 0.25], seed=42)
train, test = train.cache(), test.cache()

rf = RandomForestClassifier(featuresCol="features", labelCol="label", weightCol="weight",
                            numTrees=200, maxDepth=8, seed=42)
rf_model = rf.fit(train)
pred = rf_model.transform(test).cache()

auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                    metricName="areaUnderROC").evaluate(pred)
print(f"\nCancellation model ROC-AUC on held-out data: {auc:.3f}  (0.5 = random guessing)")

# classification report (precision / recall / f1 per class)
conf = {(r["label"], r["prediction"]): r["count"]
        for r in pred.groupBy("label", "prediction").count().collect()}
print(f"{'class':>8}{'precision':>12}{'recall':>10}{'f1':>10}{'support':>10}")
for cls in (0.0, 1.0):
    tp = conf.get((cls, cls), 0)
    fp = sum(v for (l, p_), v in conf.items() if p_ == cls and l != cls)
    fn = sum(v for (l, p_), v in conf.items() if l == cls and p_ != cls)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"{int(cls):>8}{prec:>12.3f}{rec:>10.3f}{f1:>10.3f}{tp + fn:>10}")

imp = pd.Series(rf_model.featureImportances.toArray(), index=feature_names).sort_values()
plt.figure(figsize=(8, 5))
imp.plot.barh(color="#FFD100", edgecolor="black")
plt.title(f"Cancellation Model - Feature Importance (AUC = {auc:.2f})")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("Saved feature_importance.png")
print("Interpretation: an AUC near 0.5 means time, area and vehicle type alone do not "
      "predict cancellations - they are driven by factors not in the dataset "
      "(driver availability, surge pricing, weather, traffic).")

spark.stop()"""
advanced_analysis_pyspark.py
----------------------------
PySpark version of the Ola Mobility advanced analytics layer.

  1. Area x Hour demand heatmap                 (Spark pivot -> seaborn)
  2. K-Means segmentation of the pickup areas   (Spark ML KMeans + silhouette + PCA)
  3. Chi-square test: is "peak hour" real?      (Spark aggregation -> scipy)
  4. Cancellation-risk model                    (Spark ML RandomForest, ROC-AUC)

Requirements:  pip install pyspark pandas numpy matplotlib seaborn scipy
               Java 8 / 11 / 17 installed (JAVA_HOME set)
Run:           python advanced_analysis_pyspark.py
Outputs:       heatmap_area_hour.png, area_clusters.png, feature_importance.png,
               area_segments.csv
"""
import os
import sys

# make Spark workers use the same Python interpreter as this script (avoids version errors)
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chisquare

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import (VectorAssembler, StandardScaler, PCA,
                                StringIndexer, OneHotEncoder)
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator, BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "ola_bengaluru.csv")

# ------------------------------------------------------------------ SPARK
spark = (SparkSession.builder
         .appName("OlaMobilityAdvancedAnalysis")
         .master("local[*]")
         .config("spark.sql.shuffle.partitions", "8")
         .config("spark.sql.ansi.enabled", "false")   # bad values -> null instead of crash
         .config("spark.ui.showConsoleProgress", "false")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

# ------------------------------------------------------------------ LOAD
raw = (spark.read
       .option("header", True)
       .option("inferSchema", True)
       .csv(CSV_PATH))

df = (raw
      .withColumn("Date", F.to_date(F.col("Date"), "d/M/yyyy"))
      # Time looks like "6:00:00" -> take the part before the first colon
      .withColumn("Hour", F.split(F.col("Time"), ":").getItem(0).cast("int"))
      # Spark dayofweek: 1=Sunday..7=Saturday  ->  pandas style 0=Monday..6=Sunday
      .withColumn("DayOfWeek", (F.dayofweek("Date") + 5) % 7)
      .withColumn("IsWeekend", F.when(F.col("DayOfWeek") >= 5, 1).otherwise(0))
      .withColumn("Cancelled",
                  F.when(F.col("Booking Status").startswith("Cancelled"), 1).otherwise(0))
      .withColumn("Success",
                  F.when(F.col("Booking Status") == "Success", 1).otherwise(0))
      .dropna(subset=["Date", "Hour", "Pickup Location"])
      .cache())

n_rows = df.count()
dmin, dmax = df.agg(F.min("Date"), F.max("Date")).first()
print(f"Loaded {n_rows:,} bookings from {dmin} to {dmax}")

# ------------------------------------------------- 1. AREA x HOUR HEATMAP
hours = list(range(24))
pivot_sdf = (df.groupBy("Pickup Location")
             .pivot("Hour", hours)
             .count()
             .na.fill(0))
pivot = pivot_sdf.toPandas().set_index("Pickup Location")
pivot.columns = [int(c) for c in pivot.columns]
pivot = pivot[sorted(pivot.columns)]
order = pivot.sum(axis=1).sort_values(ascending=False).index

plt.figure(figsize=(14, 10))
sns.heatmap(pivot.loc[order], cmap="YlOrRd", cbar_kws={"label": "Bookings"})
plt.title("Demand Heatmap: Pickup Area vs Hour of Day")
plt.xlabel("Hour")
plt.tight_layout()
plt.savefig("heatmap_area_hour.png", dpi=150)
plt.close()
print("Saved heatmap_area_hour.png")

# ------------------------------------- 2. K-MEANS SEGMENTATION OF AREAS
area_sdf = (df.groupBy("Pickup Location")
            .agg(F.count("*").alias("Total_Bookings"),
                 F.avg("Booking Value").alias("Avg_Fare"),
                 F.avg("Cancelled").alias("Cancel_Rate"),
                 F.avg("Success").alias("Success_Rate"),
                 F.avg("Ride Distance").alias("Avg_Distance"),
                 F.avg("Driver Ratings").alias("Avg_Rating"))
            .dropna())

features = ["Total_Bookings", "Avg_Fare", "Cancel_Rate", "Avg_Distance", "Avg_Rating"]

assembler = VectorAssembler(inputCols=features, outputCol="features")
assembled = assembler.transform(area_sdf)
scaler_model = StandardScaler(inputCol="features", outputCol="scaled",
                              withMean=True, withStd=True).fit(assembled)
scaled = scaler_model.transform(assembled).cache()

# choose k by silhouette score rather than guessing
evaluator = ClusteringEvaluator(featuresCol="scaled", predictionCol="prediction",
                                metricName="silhouette",
                                distanceMeasure="squaredEuclidean")
scores = {}
for k in range(2, 7):
    preds = KMeans(k=k, seed=42, featuresCol="scaled", maxIter=100).fit(scaled).transform(scaled)
    scores[k] = evaluator.evaluate(preds)
best_k = max(scores, key=scores.get)
print("Silhouette scores:", {k: round(v, 3) for k, v in scores.items()}, "-> k =", best_k)

km_model = KMeans(k=best_k, seed=42, featuresCol="scaled", maxIter=100).fit(scaled)
clustered = km_model.transform(scaled).withColumnRenamed("prediction", "Cluster")

pca_model = PCA(k=2, inputCol="scaled", outputCol="pca").fit(clustered)
clustered = pca_model.transform(clustered)

# collect the (tiny) 50-row result to pandas; convert Spark vectors to plain floats
rows = clustered.select("Pickup Location", *features, "Success_Rate", "Cluster", "pca").collect()
area = pd.DataFrame([{
    "Pickup Location": r["Pickup Location"],
    "Total_Bookings": r["Total_Bookings"],
    "Avg_Fare": r["Avg_Fare"],
    "Cancel_Rate": r["Cancel_Rate"],
    "Success_Rate": r["Success_Rate"],
    "Avg_Distance": r["Avg_Distance"],
    "Avg_Rating": r["Avg_Rating"],
    "Cluster": int(r["Cluster"]),
    "PC1": float(r["pca"][0]),
    "PC2": float(r["pca"][1]),
} for r in rows]).sort_values("Pickup Location").reset_index(drop=True)

# human-readable tier names, ranked by a simple value score (demand + fare - cancellations)
g = area.groupby("Cluster")[["Total_Bookings", "Avg_Fare", "Cancel_Rate"]].mean()
value = (g["Total_Bookings"] / area["Total_Bookings"].mean()
         + g["Avg_Fare"] / area["Avg_Fare"].mean()
         - g["Cancel_Rate"] / area["Cancel_Rate"].mean()).sort_values(ascending=False)
labels = ["Premium Hotspot", "Steady Performer", "Watch List", "Low Priority",
          "Underperformer", "Weak Zone"]
name_map = {c: labels[i] for i, c in enumerate(value.index)}
area["Segment"] = area["Cluster"].map(name_map)

out_cols = ["Pickup Location", "Total_Bookings", "Avg_Fare", "Cancel_Rate", "Success_Rate",
            "Avg_Distance", "Avg_Rating", "Cluster", "Segment"]
area[out_cols].round(3).to_csv("area_segments.csv", index=False)
print(area.groupby("Segment")[features].mean().round(2))

plt.figure(figsize=(9, 6))
sns.scatterplot(x=area["PC1"], y=area["PC2"], hue=area["Segment"], s=110, palette="Set2")
for _, r in area.iterrows():
    plt.annotate(r["Pickup Location"].replace("Area-", ""), (r["PC1"], r["PC2"]),
                 fontsize=7, xytext=(3, 3), textcoords="offset points")
plt.title(f"K-Means Segmentation of Pickup Areas (k={best_k}, PCA view)")
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.tight_layout()
plt.savefig("area_clusters.png", dpi=150)
plt.close()
print("Saved area_clusters.png and area_segments.csv")

# ------------------------- 3. IS THE "PEAK HOUR" STATISTICALLY REAL?
hourly_pd = (df.groupBy("Hour").count().orderBy("Hour").toPandas()
             .set_index("Hour")["count"])
chi2, p = chisquare(hourly_pd.values)  # H0: demand is equal in every hour
spread = (hourly_pd.max() - hourly_pd.min()) / hourly_pd.mean() * 100
print(f"\nChi-square test of uniform hourly demand: chi2={chi2:.1f}, p={p:.4f}")
print(f"Busiest hour: {hourly_pd.idxmax()}:00 ({hourly_pd.max()}), quietest: "
      f"{hourly_pd.idxmin()}:00 ({hourly_pd.min()}); spread = {spread:.1f}% of the mean")
print("-> Demand is "
      + ("NOT uniform (peaks are statistically significant)" if p < 0.05
         else "statistically flat: peak hours are real but small / close to random variation"))

# ------------------------ 4. CANCELLATION-RISK MODEL (Random Forest)
model_sdf = (df.select("Hour", "DayOfWeek", "IsWeekend", "Vehicle Type",
                       "Pickup Location", "Cancelled")
             .dropna()
             .withColumn("Area_ID",
                         F.regexp_replace("Pickup Location", "Area-", "").cast("int"))
             .withColumn("label", F.col("Cancelled").cast("double"))
             .dropna(subset=["Area_ID"]))

# one-hot encode Vehicle Type (fit on all data so no unseen category appears at test time)
idx_model = StringIndexer(inputCol="Vehicle Type", outputCol="vt_idx").fit(model_sdf)
vehicle_labels = list(idx_model.labels)
model_sdf = idx_model.transform(model_sdf)
ohe_model = OneHotEncoder(inputCols=["vt_idx"], outputCols=["vt_vec"],
                          dropLast=False).fit(model_sdf)
model_sdf = ohe_model.transform(model_sdf)

feature_names = ["Hour", "DayOfWeek", "IsWeekend", "Area_ID"] + \
                [f"Vehicle Type_{v}" for v in vehicle_labels]
ml_sdf = (VectorAssembler(inputCols=["Hour", "DayOfWeek", "IsWeekend", "Area_ID", "vt_vec"],
                          outputCol="features")
          .transform(model_sdf)
          .select("features", "label"))

# class weights (equivalent of class_weight="balanced")
total = ml_sdf.count()
cnt = {r["label"]: r["count"] for r in ml_sdf.groupBy("label").count().collect()}
w = {lab: total / (2.0 * c) for lab, c in cnt.items()}
ml_sdf = ml_sdf.withColumn(
    "weight", F.when(F.col("label") == 1.0, w.get(1.0, 1.0)).otherwise(w.get(0.0, 1.0)))

train, test = ml_sdf.randomSplit([0.75, 0.25], seed=42)
train, test = train.cache(), test.cache()

rf = RandomForestClassifier(featuresCol="features", labelCol="label", weightCol="weight",
                            numTrees=200, maxDepth=8, seed=42)
rf_model = rf.fit(train)
pred = rf_model.transform(test).cache()

auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                    metricName="areaUnderROC").evaluate(pred)
print(f"\nCancellation model ROC-AUC on held-out data: {auc:.3f}  (0.5 = random guessing)")

# classification report (precision / recall / f1 per class)
conf = {(r["label"], r["prediction"]): r["count"]
        for r in pred.groupBy("label", "prediction").count().collect()}
print(f"{'class':>8}{'precision':>12}{'recall':>10}{'f1':>10}{'support':>10}")
for cls in (0.0, 1.0):
    tp = conf.get((cls, cls), 0)
    fp = sum(v for (l, p_), v in conf.items() if p_ == cls and l != cls)
    fn = sum(v for (l, p_), v in conf.items() if l == cls and p_ != cls)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"{int(cls):>8}{prec:>12.3f}{rec:>10.3f}{f1:>10.3f}{tp + fn:>10}")

imp = pd.Series(rf_model.featureImportances.toArray(), index=feature_names).sort_values()
plt.figure(figsize=(8, 5))
imp.plot.barh(color="#FFD100", edgecolor="black")
plt.title(f"Cancellation Model - Feature Importance (AUC = {auc:.2f})")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("Saved feature_importance.png")
print("Interpretation: an AUC near 0.5 means time, area and vehicle type alone do not "
      "predict cancellations - they are driven by factors not in the dataset "
      "(driver availability, surge pricing, weather, traffic).")

spark.stop()"""
advanced_analysis_pyspark.py
----------------------------
PySpark version of the Ola Mobility advanced analytics layer.

  1. Area x Hour demand heatmap                 (Spark pivot -> seaborn)
  2. K-Means segmentation of the pickup areas   (Spark ML KMeans + silhouette + PCA)
  3. Chi-square test: is "peak hour" real?      (Spark aggregation -> scipy)
  4. Cancellation-risk model                    (Spark ML RandomForest, ROC-AUC)

Requirements:  pip install pyspark pandas numpy matplotlib seaborn scipy
               Java 8 / 11 / 17 installed (JAVA_HOME set)
Run:           python advanced_analysis_pyspark.py
Outputs:       heatmap_area_hour.png, area_clusters.png, feature_importance.png,
               area_segments.csv
"""
import os
import sys

# make Spark workers use the same Python interpreter as this script (avoids version errors)
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chisquare

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import (VectorAssembler, StandardScaler, PCA,
                                StringIndexer, OneHotEncoder)
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator, BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "ola_bengaluru.csv")

# ------------------------------------------------------------------ SPARK
spark = (SparkSession.builder
         .appName("OlaMobilityAdvancedAnalysis")
         .master("local[*]")
         .config("spark.sql.shuffle.partitions", "8")
         .config("spark.sql.ansi.enabled", "false")   # bad values -> null instead of crash
         .config("spark.ui.showConsoleProgress", "false")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

# ------------------------------------------------------------------ LOAD
raw = (spark.read
       .option("header", True)
       .option("inferSchema", True)
       .csv(CSV_PATH))

df = (raw
      .withColumn("Date", F.to_date(F.col("Date"), "d/M/yyyy"))
      # Time looks like "6:00:00" -> take the part before the first colon
      .withColumn("Hour", F.split(F.col("Time"), ":").getItem(0).cast("int"))
      # Spark dayofweek: 1=Sunday..7=Saturday  ->  pandas style 0=Monday..6=Sunday
      .withColumn("DayOfWeek", (F.dayofweek("Date") + 5) % 7)
      .withColumn("IsWeekend", F.when(F.col("DayOfWeek") >= 5, 1).otherwise(0))
      .withColumn("Cancelled",
                  F.when(F.col("Booking Status").startswith("Cancelled"), 1).otherwise(0))
      .withColumn("Success",
                  F.when(F.col("Booking Status") == "Success", 1).otherwise(0))
      .dropna(subset=["Date", "Hour", "Pickup Location"])
      .cache())

n_rows = df.count()
dmin, dmax = df.agg(F.min("Date"), F.max("Date")).first()
print(f"Loaded {n_rows:,} bookings from {dmin} to {dmax}")

# ------------------------------------------------- 1. AREA x HOUR HEATMAP
hours = list(range(24))
pivot_sdf = (df.groupBy("Pickup Location")
             .pivot("Hour", hours)
             .count()
             .na.fill(0))
pivot = pivot_sdf.toPandas().set_index("Pickup Location")
pivot.columns = [int(c) for c in pivot.columns]
pivot = pivot[sorted(pivot.columns)]
order = pivot.sum(axis=1).sort_values(ascending=False).index

plt.figure(figsize=(14, 10))
sns.heatmap(pivot.loc[order], cmap="YlOrRd", cbar_kws={"label": "Bookings"})
plt.title("Demand Heatmap: Pickup Area vs Hour of Day")
plt.xlabel("Hour")
plt.tight_layout()
plt.savefig("heatmap_area_hour.png", dpi=150)
plt.close()
print("Saved heatmap_area_hour.png")

# ------------------------------------- 2. K-MEANS SEGMENTATION OF AREAS
area_sdf = (df.groupBy("Pickup Location")
            .agg(F.count("*").alias("Total_Bookings"),
                 F.avg("Booking Value").alias("Avg_Fare"),
                 F.avg("Cancelled").alias("Cancel_Rate"),
                 F.avg("Success").alias("Success_Rate"),
                 F.avg("Ride Distance").alias("Avg_Distance"),
                 F.avg("Driver Ratings").alias("Avg_Rating"))
            .dropna())

features = ["Total_Bookings", "Avg_Fare", "Cancel_Rate", "Avg_Distance", "Avg_Rating"]

assembler = VectorAssembler(inputCols=features, outputCol="features")
assembled = assembler.transform(area_sdf)
scaler_model = StandardScaler(inputCol="features", outputCol="scaled",
                              withMean=True, withStd=True).fit(assembled)
scaled = scaler_model.transform(assembled).cache()

# choose k by silhouette score rather than guessing
evaluator = ClusteringEvaluator(featuresCol="scaled", predictionCol="prediction",
                                metricName="silhouette",
                                distanceMeasure="squaredEuclidean")
scores = {}
for k in range(2, 7):
    preds = KMeans(k=k, seed=42, featuresCol="scaled", maxIter=100).fit(scaled).transform(scaled)
    scores[k] = evaluator.evaluate(preds)
best_k = max(scores, key=scores.get)
print("Silhouette scores:", {k: round(v, 3) for k, v in scores.items()}, "-> k =", best_k)

km_model = KMeans(k=best_k, seed=42, featuresCol="scaled", maxIter=100).fit(scaled)
clustered = km_model.transform(scaled).withColumnRenamed("prediction", "Cluster")

pca_model = PCA(k=2, inputCol="scaled", outputCol="pca").fit(clustered)
clustered = pca_model.transform(clustered)

# collect the (tiny) 50-row result to pandas; convert Spark vectors to plain floats
rows = clustered.select("Pickup Location", *features, "Success_Rate", "Cluster", "pca").collect()
area = pd.DataFrame([{
    "Pickup Location": r["Pickup Location"],
    "Total_Bookings": r["Total_Bookings"],
    "Avg_Fare": r["Avg_Fare"],
    "Cancel_Rate": r["Cancel_Rate"],
    "Success_Rate": r["Success_Rate"],
    "Avg_Distance": r["Avg_Distance"],
    "Avg_Rating": r["Avg_Rating"],
    "Cluster": int(r["Cluster"]),
    "PC1": float(r["pca"][0]),
    "PC2": float(r["pca"][1]),
} for r in rows]).sort_values("Pickup Location").reset_index(drop=True)

# human-readable tier names, ranked by a simple value score (demand + fare - cancellations)
g = area.groupby("Cluster")[["Total_Bookings", "Avg_Fare", "Cancel_Rate"]].mean()
value = (g["Total_Bookings"] / area["Total_Bookings"].mean()
         + g["Avg_Fare"] / area["Avg_Fare"].mean()
         - g["Cancel_Rate"] / area["Cancel_Rate"].mean()).sort_values(ascending=False)
labels = ["Premium Hotspot", "Steady Performer", "Watch List", "Low Priority",
          "Underperformer", "Weak Zone"]
name_map = {c: labels[i] for i, c in enumerate(value.index)}
area["Segment"] = area["Cluster"].map(name_map)

out_cols = ["Pickup Location", "Total_Bookings", "Avg_Fare", "Cancel_Rate", "Success_Rate",
            "Avg_Distance", "Avg_Rating", "Cluster", "Segment"]
area[out_cols].round(3).to_csv("area_segments.csv", index=False)
print(area.groupby("Segment")[features].mean().round(2))

plt.figure(figsize=(9, 6))
sns.scatterplot(x=area["PC1"], y=area["PC2"], hue=area["Segment"], s=110, palette="Set2")
for _, r in area.iterrows():
    plt.annotate(r["Pickup Location"].replace("Area-", ""), (r["PC1"], r["PC2"]),
                 fontsize=7, xytext=(3, 3), textcoords="offset points")
plt.title(f"K-Means Segmentation of Pickup Areas (k={best_k}, PCA view)")
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.tight_layout()
plt.savefig("area_clusters.png", dpi=150)
plt.close()
print("Saved area_clusters.png and area_segments.csv")

# ------------------------- 3. IS THE "PEAK HOUR" STATISTICALLY REAL?
hourly_pd = (df.groupBy("Hour").count().orderBy("Hour").toPandas()
             .set_index("Hour")["count"])
chi2, p = chisquare(hourly_pd.values)  # H0: demand is equal in every hour
spread = (hourly_pd.max() - hourly_pd.min()) / hourly_pd.mean() * 100
print(f"\nChi-square test of uniform hourly demand: chi2={chi2:.1f}, p={p:.4f}")
print(f"Busiest hour: {hourly_pd.idxmax()}:00 ({hourly_pd.max()}), quietest: "
      f"{hourly_pd.idxmin()}:00 ({hourly_pd.min()}); spread = {spread:.1f}% of the mean")
print("-> Demand is "
      + ("NOT uniform (peaks are statistically significant)" if p < 0.05
         else "statistically flat: peak hours are real but small / close to random variation"))

# ------------------------ 4. CANCELLATION-RISK MODEL (Random Forest)
model_sdf = (df.select("Hour", "DayOfWeek", "IsWeekend", "Vehicle Type",
                       "Pickup Location", "Cancelled")
             .dropna()
             .withColumn("Area_ID",
                         F.regexp_replace("Pickup Location", "Area-", "").cast("int"))
             .withColumn("label", F.col("Cancelled").cast("double"))
             .dropna(subset=["Area_ID"]))

# one-hot encode Vehicle Type (fit on all data so no unseen category appears at test time)
idx_model = StringIndexer(inputCol="Vehicle Type", outputCol="vt_idx").fit(model_sdf)
vehicle_labels = list(idx_model.labels)
model_sdf = idx_model.transform(model_sdf)
ohe_model = OneHotEncoder(inputCols=["vt_idx"], outputCols=["vt_vec"],
                          dropLast=False).fit(model_sdf)
model_sdf = ohe_model.transform(model_sdf)

feature_names = ["Hour", "DayOfWeek", "IsWeekend", "Area_ID"] + \
                [f"Vehicle Type_{v}" for v in vehicle_labels]
ml_sdf = (VectorAssembler(inputCols=["Hour", "DayOfWeek", "IsWeekend", "Area_ID", "vt_vec"],
                          outputCol="features")
          .transform(model_sdf)
          .select("features", "label"))

# class weights (equivalent of class_weight="balanced")
total = ml_sdf.count()
cnt = {r["label"]: r["count"] for r in ml_sdf.groupBy("label").count().collect()}
w = {lab: total / (2.0 * c) for lab, c in cnt.items()}
ml_sdf = ml_sdf.withColumn(
    "weight", F.when(F.col("label") == 1.0, w.get(1.0, 1.0)).otherwise(w.get(0.0, 1.0)))

train, test = ml_sdf.randomSplit([0.75, 0.25], seed=42)
train, test = train.cache(), test.cache()

rf = RandomForestClassifier(featuresCol="features", labelCol="label", weightCol="weight",
                            numTrees=200, maxDepth=8, seed=42)
rf_model = rf.fit(train)
pred = rf_model.transform(test).cache()

auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                    metricName="areaUnderROC").evaluate(pred)
print(f"\nCancellation model ROC-AUC on held-out data: {auc:.3f}  (0.5 = random guessing)")

# classification report (precision / recall / f1 per class)
conf = {(r["label"], r["prediction"]): r["count"]
        for r in pred.groupBy("label", "prediction").count().collect()}
print(f"{'class':>8}{'precision':>12}{'recall':>10}{'f1':>10}{'support':>10}")
for cls in (0.0, 1.0):
    tp = conf.get((cls, cls), 0)
    fp = sum(v for (l, p_), v in conf.items() if p_ == cls and l != cls)
    fn = sum(v for (l, p_), v in conf.items() if l == cls and p_ != cls)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"{int(cls):>8}{prec:>12.3f}{rec:>10.3f}{f1:>10.3f}{tp + fn:>10}")

imp = pd.Series(rf_model.featureImportances.toArray(), index=feature_names).sort_values()
plt.figure(figsize=(8, 5))
imp.plot.barh(color="#FFD100", edgecolor="black")
plt.title(f"Cancellation Model - Feature Importance (AUC = {auc:.2f})")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("Saved feature_importance.png")
print("Interpretation: an AUC near 0.5 means time, area and vehicle type alone do not "
      "predict cancellations - they are driven by factors not in the dataset "
      "(driver availability, surge pricing, weather, traffic).")

spark.stop()"""
advanced_analysis_pyspark.py
----------------------------
PySpark version of the Ola Mobility advanced analytics layer.

  1. Area x Hour demand heatmap                 (Spark pivot -> seaborn)
  2. K-Means segmentation of the pickup areas   (Spark ML KMeans + silhouette + PCA)
  3. Chi-square test: is "peak hour" real?      (Spark aggregation -> scipy)
  4. Cancellation-risk model                    (Spark ML RandomForest, ROC-AUC)

Requirements:  pip install pyspark pandas numpy matplotlib seaborn scipy
               Java 8 / 11 / 17 installed (JAVA_HOME set)
Run:           python advanced_analysis_pyspark.py
Outputs:       heatmap_area_hour.png, area_clusters.png, feature_importance.png,
               area_segments.csv
"""
import os
import sys

# make Spark workers use the same Python interpreter as this script (avoids version errors)
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chisquare

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import (VectorAssembler, StandardScaler, PCA,
                                StringIndexer, OneHotEncoder)
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator, BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "ola_bengaluru.csv")

# ------------------------------------------------------------------ SPARK
spark = (SparkSession.builder
         .appName("OlaMobilityAdvancedAnalysis")
         .master("local[*]")
         .config("spark.sql.shuffle.partitions", "8")
         .config("spark.sql.ansi.enabled", "false")   # bad values -> null instead of crash
         .config("spark.ui.showConsoleProgress", "false")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

# ------------------------------------------------------------------ LOAD
raw = (spark.read
       .option("header", True)
       .option("inferSchema", True)
       .csv(CSV_PATH))

df = (raw
      .withColumn("Date", F.to_date(F.col("Date"), "d/M/yyyy"))
      # Time looks like "6:00:00" -> take the part before the first colon
      .withColumn("Hour", F.split(F.col("Time"), ":").getItem(0).cast("int"))
      # Spark dayofweek: 1=Sunday..7=Saturday  ->  pandas style 0=Monday..6=Sunday
      .withColumn("DayOfWeek", (F.dayofweek("Date") + 5) % 7)
      .withColumn("IsWeekend", F.when(F.col("DayOfWeek") >= 5, 1).otherwise(0))
      .withColumn("Cancelled",
                  F.when(F.col("Booking Status").startswith("Cancelled"), 1).otherwise(0))
      .withColumn("Success",
                  F.when(F.col("Booking Status") == "Success", 1).otherwise(0))
      .dropna(subset=["Date", "Hour", "Pickup Location"])
      .cache())

n_rows = df.count()
dmin, dmax = df.agg(F.min("Date"), F.max("Date")).first()
print(f"Loaded {n_rows:,} bookings from {dmin} to {dmax}")

# ------------------------------------------------- 1. AREA x HOUR HEATMAP
hours = list(range(24))
pivot_sdf = (df.groupBy("Pickup Location")
             .pivot("Hour", hours)
             .count()
             .na.fill(0))
pivot = pivot_sdf.toPandas().set_index("Pickup Location")
pivot.columns = [int(c) for c in pivot.columns]
pivot = pivot[sorted(pivot.columns)]
order = pivot.sum(axis=1).sort_values(ascending=False).index

plt.figure(figsize=(14, 10))
sns.heatmap(pivot.loc[order], cmap="YlOrRd", cbar_kws={"label": "Bookings"})
plt.title("Demand Heatmap: Pickup Area vs Hour of Day")
plt.xlabel("Hour")
plt.tight_layout()
plt.savefig("heatmap_area_hour.png", dpi=150)
plt.close()
print("Saved heatmap_area_hour.png")

# ------------------------------------- 2. K-MEANS SEGMENTATION OF AREAS
area_sdf = (df.groupBy("Pickup Location")
            .agg(F.count("*").alias("Total_Bookings"),
                 F.avg("Booking Value").alias("Avg_Fare"),
                 F.avg("Cancelled").alias("Cancel_Rate"),
                 F.avg("Success").alias("Success_Rate"),
                 F.avg("Ride Distance").alias("Avg_Distance"),
                 F.avg("Driver Ratings").alias("Avg_Rating"))
            .dropna())

features = ["Total_Bookings", "Avg_Fare", "Cancel_Rate", "Avg_Distance", "Avg_Rating"]

assembler = VectorAssembler(inputCols=features, outputCol="features")
assembled = assembler.transform(area_sdf)
scaler_model = StandardScaler(inputCol="features", outputCol="scaled",
                              withMean=True, withStd=True).fit(assembled)
scaled = scaler_model.transform(assembled).cache()

# choose k by silhouette score rather than guessing
evaluator = ClusteringEvaluator(featuresCol="scaled", predictionCol="prediction",
                                metricName="silhouette",
                                distanceMeasure="squaredEuclidean")
scores = {}
for k in range(2, 7):
    preds = KMeans(k=k, seed=42, featuresCol="scaled", maxIter=100).fit(scaled).transform(scaled)
    scores[k] = evaluator.evaluate(preds)
best_k = max(scores, key=scores.get)
print("Silhouette scores:", {k: round(v, 3) for k, v in scores.items()}, "-> k =", best_k)

km_model = KMeans(k=best_k, seed=42, featuresCol="scaled", maxIter=100).fit(scaled)
clustered = km_model.transform(scaled).withColumnRenamed("prediction", "Cluster")

pca_model = PCA(k=2, inputCol="scaled", outputCol="pca").fit(clustered)
clustered = pca_model.transform(clustered)

# collect the (tiny) 50-row result to pandas; convert Spark vectors to plain floats
rows = clustered.select("Pickup Location", *features, "Success_Rate", "Cluster", "pca").collect()
area = pd.DataFrame([{
    "Pickup Location": r["Pickup Location"],
    "Total_Bookings": r["Total_Bookings"],
    "Avg_Fare": r["Avg_Fare"],
    "Cancel_Rate": r["Cancel_Rate"],
    "Success_Rate": r["Success_Rate"],
    "Avg_Distance": r["Avg_Distance"],
    "Avg_Rating": r["Avg_Rating"],
    "Cluster": int(r["Cluster"]),
    "PC1": float(r["pca"][0]),
    "PC2": float(r["pca"][1]),
} for r in rows]).sort_values("Pickup Location").reset_index(drop=True)

# human-readable tier names, ranked by a simple value score (demand + fare - cancellations)
g = area.groupby("Cluster")[["Total_Bookings", "Avg_Fare", "Cancel_Rate"]].mean()
value = (g["Total_Bookings"] / area["Total_Bookings"].mean()
         + g["Avg_Fare"] / area["Avg_Fare"].mean()
         - g["Cancel_Rate"] / area["Cancel_Rate"].mean()).sort_values(ascending=False)
labels = ["Premium Hotspot", "Steady Performer", "Watch List", "Low Priority",
          "Underperformer", "Weak Zone"]
name_map = {c: labels[i] for i, c in enumerate(value.index)}
area["Segment"] = area["Cluster"].map(name_map)

out_cols = ["Pickup Location", "Total_Bookings", "Avg_Fare", "Cancel_Rate", "Success_Rate",
            "Avg_Distance", "Avg_Rating", "Cluster", "Segment"]
area[out_cols].round(3).to_csv("area_segments.csv", index=False)
print(area.groupby("Segment")[features].mean().round(2))

plt.figure(figsize=(9, 6))
sns.scatterplot(x=area["PC1"], y=area["PC2"], hue=area["Segment"], s=110, palette="Set2")
for _, r in area.iterrows():
    plt.annotate(r["Pickup Location"].replace("Area-", ""), (r["PC1"], r["PC2"]),
                 fontsize=7, xytext=(3, 3), textcoords="offset points")
plt.title(f"K-Means Segmentation of Pickup Areas (k={best_k}, PCA view)")
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.tight_layout()
plt.savefig("area_clusters.png", dpi=150)
plt.close()
print("Saved area_clusters.png and area_segments.csv")

# ------------------------- 3. IS THE "PEAK HOUR" STATISTICALLY REAL?
hourly_pd = (df.groupBy("Hour").count().orderBy("Hour").toPandas()
             .set_index("Hour")["count"])
chi2, p = chisquare(hourly_pd.values)  # H0: demand is equal in every hour
spread = (hourly_pd.max() - hourly_pd.min()) / hourly_pd.mean() * 100
print(f"\nChi-square test of uniform hourly demand: chi2={chi2:.1f}, p={p:.4f}")
print(f"Busiest hour: {hourly_pd.idxmax()}:00 ({hourly_pd.max()}), quietest: "
      f"{hourly_pd.idxmin()}:00 ({hourly_pd.min()}); spread = {spread:.1f}% of the mean")
print("-> Demand is "
      + ("NOT uniform (peaks are statistically significant)" if p < 0.05
         else "statistically flat: peak hours are real but small / close to random variation"))

# ------------------------ 4. CANCELLATION-RISK MODEL (Random Forest)
model_sdf = (df.select("Hour", "DayOfWeek", "IsWeekend", "Vehicle Type",
                       "Pickup Location", "Cancelled")
             .dropna()
             .withColumn("Area_ID",
                         F.regexp_replace("Pickup Location", "Area-", "").cast("int"))
             .withColumn("label", F.col("Cancelled").cast("double"))
             .dropna(subset=["Area_ID"]))

# one-hot encode Vehicle Type (fit on all data so no unseen category appears at test time)
idx_model = StringIndexer(inputCol="Vehicle Type", outputCol="vt_idx").fit(model_sdf)
vehicle_labels = list(idx_model.labels)
model_sdf = idx_model.transform(model_sdf)
ohe_model = OneHotEncoder(inputCols=["vt_idx"], outputCols=["vt_vec"],
                          dropLast=False).fit(model_sdf)
model_sdf = ohe_model.transform(model_sdf)

feature_names = ["Hour", "DayOfWeek", "IsWeekend", "Area_ID"] + \
                [f"Vehicle Type_{v}" for v in vehicle_labels]
ml_sdf = (VectorAssembler(inputCols=["Hour", "DayOfWeek", "IsWeekend", "Area_ID", "vt_vec"],
                          outputCol="features")
          .transform(model_sdf)
          .select("features", "label"))

# class weights (equivalent of class_weight="balanced")
total = ml_sdf.count()
cnt = {r["label"]: r["count"] for r in ml_sdf.groupBy("label").count().collect()}
w = {lab: total / (2.0 * c) for lab, c in cnt.items()}
ml_sdf = ml_sdf.withColumn(
    "weight", F.when(F.col("label") == 1.0, w.get(1.0, 1.0)).otherwise(w.get(0.0, 1.0)))

train, test = ml_sdf.randomSplit([0.75, 0.25], seed=42)
train, test = train.cache(), test.cache()

rf = RandomForestClassifier(featuresCol="features", labelCol="label", weightCol="weight",
                            numTrees=200, maxDepth=8, seed=42)
rf_model = rf.fit(train)
pred = rf_model.transform(test).cache()

auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                    metricName="areaUnderROC").evaluate(pred)
print(f"\nCancellation model ROC-AUC on held-out data: {auc:.3f}  (0.5 = random guessing)")

# classification report (precision / recall / f1 per class)
conf = {(r["label"], r["prediction"]): r["count"]
        for r in pred.groupBy("label", "prediction").count().collect()}
print(f"{'class':>8}{'precision':>12}{'recall':>10}{'f1':>10}{'support':>10}")
for cls in (0.0, 1.0):
    tp = conf.get((cls, cls), 0)
    fp = sum(v for (l, p_), v in conf.items() if p_ == cls and l != cls)
    fn = sum(v for (l, p_), v in conf.items() if l == cls and p_ != cls)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"{int(cls):>8}{prec:>12.3f}{rec:>10.3f}{f1:>10.3f}{tp + fn:>10}")

imp = pd.Series(rf_model.featureImportances.toArray(), index=feature_names).sort_values()
plt.figure(figsize=(8, 5))
imp.plot.barh(color="#FFD100", edgecolor="black")
plt.title(f"Cancellation Model - Feature Importance (AUC = {auc:.2f})")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("Saved feature_importance.png")
print("Interpretation: an AUC near 0.5 means time, area and vehicle type alone do not "
      "predict cancellations - they are driven by factors not in the dataset "
      "(driver availability, surge pricing, weather, traffic).")

spark.stop()"""
advanced_analysis_pyspark.py
----------------------------
PySpark version of the Ola Mobility advanced analytics layer.

  1. Area x Hour demand heatmap                 (Spark pivot -> seaborn)
  2. K-Means segmentation of the pickup areas   (Spark ML KMeans + silhouette + PCA)
  3. Chi-square test: is "peak hour" real?      (Spark aggregation -> scipy)
  4. Cancellation-risk model                    (Spark ML RandomForest, ROC-AUC)

Requirements:  pip install pyspark pandas numpy matplotlib seaborn scipy
               Java 8 / 11 / 17 installed (JAVA_HOME set)
Run:           python advanced_analysis_pyspark.py
Outputs:       heatmap_area_hour.png, area_clusters.png, feature_importance.png,
               area_segments.csv
"""
import os
import sys

# make Spark workers use the same Python interpreter as this script (avoids version errors)
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chisquare

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import (VectorAssembler, StandardScaler, PCA,
                                StringIndexer, OneHotEncoder)
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator, BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "ola_bengaluru.csv")

# ------------------------------------------------------------------ SPARK
spark = (SparkSession.builder
         .appName("OlaMobilityAdvancedAnalysis")
         .master("local[*]")
         .config("spark.sql.shuffle.partitions", "8")
         .config("spark.sql.ansi.enabled", "false")   # bad values -> null instead of crash
         .config("spark.ui.showConsoleProgress", "false")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

# ------------------------------------------------------------------ LOAD
raw = (spark.read
       .option("header", True)
       .option("inferSchema", True)
       .csv(CSV_PATH))

df = (raw
      .withColumn("Date", F.to_date(F.col("Date"), "d/M/yyyy"))
      # Time looks like "6:00:00" -> take the part before the first colon
      .withColumn("Hour", F.split(F.col("Time"), ":").getItem(0).cast("int"))
      # Spark dayofweek: 1=Sunday..7=Saturday  ->  pandas style 0=Monday..6=Sunday
      .withColumn("DayOfWeek", (F.dayofweek("Date") + 5) % 7)
      .withColumn("IsWeekend", F.when(F.col("DayOfWeek") >= 5, 1).otherwise(0))
      .withColumn("Cancelled",
                  F.when(F.col("Booking Status").startswith("Cancelled"), 1).otherwise(0))
      .withColumn("Success",
                  F.when(F.col("Booking Status") == "Success", 1).otherwise(0))
      .dropna(subset=["Date", "Hour", "Pickup Location"])
      .cache())

n_rows = df.count()
dmin, dmax = df.agg(F.min("Date"), F.max("Date")).first()
print(f"Loaded {n_rows:,} bookings from {dmin} to {dmax}")

# ------------------------------------------------- 1. AREA x HOUR HEATMAP
hours = list(range(24))
pivot_sdf = (df.groupBy("Pickup Location")
             .pivot("Hour", hours)
             .count()
             .na.fill(0))
pivot = pivot_sdf.toPandas().set_index("Pickup Location")
pivot.columns = [int(c) for c in pivot.columns]
pivot = pivot[sorted(pivot.columns)]
order = pivot.sum(axis=1).sort_values(ascending=False).index

plt.figure(figsize=(14, 10))
sns.heatmap(pivot.loc[order], cmap="YlOrRd", cbar_kws={"label": "Bookings"})
plt.title("Demand Heatmap: Pickup Area vs Hour of Day")
plt.xlabel("Hour")
plt.tight_layout()
plt.savefig("heatmap_area_hour.png", dpi=150)
plt.close()
print("Saved heatmap_area_hour.png")

# ------------------------------------- 2. K-MEANS SEGMENTATION OF AREAS
area_sdf = (df.groupBy("Pickup Location")
            .agg(F.count("*").alias("Total_Bookings"),
                 F.avg("Booking Value").alias("Avg_Fare"),
                 F.avg("Cancelled").alias("Cancel_Rate"),
                 F.avg("Success").alias("Success_Rate"),
                 F.avg("Ride Distance").alias("Avg_Distance"),
                 F.avg("Driver Ratings").alias("Avg_Rating"))
            .dropna())

features = ["Total_Bookings", "Avg_Fare", "Cancel_Rate", "Avg_Distance", "Avg_Rating"]

assembler = VectorAssembler(inputCols=features, outputCol="features")
assembled = assembler.transform(area_sdf)
scaler_model = StandardScaler(inputCol="features", outputCol="scaled",
                              withMean=True, withStd=True).fit(assembled)
scaled = scaler_model.transform(assembled).cache()

# choose k by silhouette score rather than guessing
evaluator = ClusteringEvaluator(featuresCol="scaled", predictionCol="prediction",
                                metricName="silhouette",
                                distanceMeasure="squaredEuclidean")
scores = {}
for k in range(2, 7):
    preds = KMeans(k=k, seed=42, featuresCol="scaled", maxIter=100).fit(scaled).transform(scaled)
    scores[k] = evaluator.evaluate(preds)
best_k = max(scores, key=scores.get)
print("Silhouette scores:", {k: round(v, 3) for k, v in scores.items()}, "-> k =", best_k)

km_model = KMeans(k=best_k, seed=42, featuresCol="scaled", maxIter=100).fit(scaled)
clustered = km_model.transform(scaled).withColumnRenamed("prediction", "Cluster")

pca_model = PCA(k=2, inputCol="scaled", outputCol="pca").fit(clustered)
clustered = pca_model.transform(clustered)

# collect the (tiny) 50-row result to pandas; convert Spark vectors to plain floats
rows = clustered.select("Pickup Location", *features, "Success_Rate", "Cluster", "pca").collect()
area = pd.DataFrame([{
    "Pickup Location": r["Pickup Location"],
    "Total_Bookings": r["Total_Bookings"],
    "Avg_Fare": r["Avg_Fare"],
    "Cancel_Rate": r["Cancel_Rate"],
    "Success_Rate": r["Success_Rate"],
    "Avg_Distance": r["Avg_Distance"],
    "Avg_Rating": r["Avg_Rating"],
    "Cluster": int(r["Cluster"]),
    "PC1": float(r["pca"][0]),
    "PC2": float(r["pca"][1]),
} for r in rows]).sort_values("Pickup Location").reset_index(drop=True)

# human-readable tier names, ranked by a simple value score (demand + fare - cancellations)
g = area.groupby("Cluster")[["Total_Bookings", "Avg_Fare", "Cancel_Rate"]].mean()
value = (g["Total_Bookings"] / area["Total_Bookings"].mean()
         + g["Avg_Fare"] / area["Avg_Fare"].mean()
         - g["Cancel_Rate"] / area["Cancel_Rate"].mean()).sort_values(ascending=False)
labels = ["Premium Hotspot", "Steady Performer", "Watch List", "Low Priority",
          "Underperformer", "Weak Zone"]
name_map = {c: labels[i] for i, c in enumerate(value.index)}
area["Segment"] = area["Cluster"].map(name_map)

out_cols = ["Pickup Location", "Total_Bookings", "Avg_Fare", "Cancel_Rate", "Success_Rate",
            "Avg_Distance", "Avg_Rating", "Cluster", "Segment"]
area[out_cols].round(3).to_csv("area_segments.csv", index=False)
print(area.groupby("Segment")[features].mean().round(2))

plt.figure(figsize=(9, 6))
sns.scatterplot(x=area["PC1"], y=area["PC2"], hue=area["Segment"], s=110, palette="Set2")
for _, r in area.iterrows():
    plt.annotate(r["Pickup Location"].replace("Area-", ""), (r["PC1"], r["PC2"]),
                 fontsize=7, xytext=(3, 3), textcoords="offset points")
plt.title(f"K-Means Segmentation of Pickup Areas (k={best_k}, PCA view)")
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.tight_layout()
plt.savefig("area_clusters.png", dpi=150)
plt.close()
print("Saved area_clusters.png and area_segments.csv")

# ------------------------- 3. IS THE "PEAK HOUR" STATISTICALLY REAL?
hourly_pd = (df.groupBy("Hour").count().orderBy("Hour").toPandas()
             .set_index("Hour")["count"])
chi2, p = chisquare(hourly_pd.values)  # H0: demand is equal in every hour
spread = (hourly_pd.max() - hourly_pd.min()) / hourly_pd.mean() * 100
print(f"\nChi-square test of uniform hourly demand: chi2={chi2:.1f}, p={p:.4f}")
print(f"Busiest hour: {hourly_pd.idxmax()}:00 ({hourly_pd.max()}), quietest: "
      f"{hourly_pd.idxmin()}:00 ({hourly_pd.min()}); spread = {spread:.1f}% of the mean")
print("-> Demand is "
      + ("NOT uniform (peaks are statistically significant)" if p < 0.05
         else "statistically flat: peak hours are real but small / close to random variation"))

# ------------------------ 4. CANCELLATION-RISK MODEL (Random Forest)
model_sdf = (df.select("Hour", "DayOfWeek", "IsWeekend", "Vehicle Type",
                       "Pickup Location", "Cancelled")
             .dropna()
             .withColumn("Area_ID",
                         F.regexp_replace("Pickup Location", "Area-", "").cast("int"))
             .withColumn("label", F.col("Cancelled").cast("double"))
             .dropna(subset=["Area_ID"]))

# one-hot encode Vehicle Type (fit on all data so no unseen category appears at test time)
idx_model = StringIndexer(inputCol="Vehicle Type", outputCol="vt_idx").fit(model_sdf)
vehicle_labels = list(idx_model.labels)
model_sdf = idx_model.transform(model_sdf)
ohe_model = OneHotEncoder(inputCols=["vt_idx"], outputCols=["vt_vec"],
                          dropLast=False).fit(model_sdf)
model_sdf = ohe_model.transform(model_sdf)

feature_names = ["Hour", "DayOfWeek", "IsWeekend", "Area_ID"] + \
                [f"Vehicle Type_{v}" for v in vehicle_labels]
ml_sdf = (VectorAssembler(inputCols=["Hour", "DayOfWeek", "IsWeekend", "Area_ID", "vt_vec"],
                          outputCol="features")
          .transform(model_sdf)
          .select("features", "label"))

# class weights (equivalent of class_weight="balanced")
total = ml_sdf.count()
cnt = {r["label"]: r["count"] for r in ml_sdf.groupBy("label").count().collect()}
w = {lab: total / (2.0 * c) for lab, c in cnt.items()}
ml_sdf = ml_sdf.withColumn(
    "weight", F.when(F.col("label") == 1.0, w.get(1.0, 1.0)).otherwise(w.get(0.0, 1.0)))

train, test = ml_sdf.randomSplit([0.75, 0.25], seed=42)
train, test = train.cache(), test.cache()

rf = RandomForestClassifier(featuresCol="features", labelCol="label", weightCol="weight",
                            numTrees=200, maxDepth=8, seed=42)
rf_model = rf.fit(train)
pred = rf_model.transform(test).cache()

auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                    metricName="areaUnderROC").evaluate(pred)
print(f"\nCancellation model ROC-AUC on held-out data: {auc:.3f}  (0.5 = random guessing)")

# classification report (precision / recall / f1 per class)
conf = {(r["label"], r["prediction"]): r["count"]
        for r in pred.groupBy("label", "prediction").count().collect()}
print(f"{'class':>8}{'precision':>12}{'recall':>10}{'f1':>10}{'support':>10}")
for cls in (0.0, 1.0):
    tp = conf.get((cls, cls), 0)
    fp = sum(v for (l, p_), v in conf.items() if p_ == cls and l != cls)
    fn = sum(v for (l, p_), v in conf.items() if l == cls and p_ != cls)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"{int(cls):>8}{prec:>12.3f}{rec:>10.3f}{f1:>10.3f}{tp + fn:>10}")

imp = pd.Series(rf_model.featureImportances.toArray(), index=feature_names).sort_values()
plt.figure(figsize=(8, 5))
imp.plot.barh(color="#FFD100", edgecolor="black")
plt.title(f"Cancellation Model - Feature Importance (AUC = {auc:.2f})")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("Saved feature_importance.png")
print("Interpretation: an AUC near 0.5 means time, area and vehicle type alone do not "
      "predict cancellations - they are driven by factors not in the dataset "
      "(driver availability, surge pricing, weather, traffic).")

spark.stop()
