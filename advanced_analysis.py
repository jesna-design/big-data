"""
advanced_analysis.py
--------------------
Advanced analytics layer for the Ola Mobility project.

Adds four things on top of analysis.py:
  1. Area x Hour demand heatmap
  2. K-Means segmentation of the 50 pickup areas into business tiers
  3. Statistical test: is "peak hour" a real effect or just noise?
  4. Cancellation-risk ML model (Random Forest) with honest validation

Run:  python advanced_analysis.py
Outputs: heatmap_area_hour.png, area_clusters.png, feature_importance.png,
         area_segments.csv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chisquare
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

# ------------------------------------------------------------------ LOAD
df = pd.read_csv("ola_bengaluru.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
df["Cancelled"] = df["Booking Status"].str.startswith("Cancelled").astype(int)
df["Success"] = (df["Booking Status"] == "Success").astype(int)
print(f"Loaded {len(df):,} bookings from {df['Date'].min().date()} to {df['Date'].max().date()}")

# ------------------------------------------------- 1. AREA x HOUR HEATMAP
pivot = df.pivot_table(index="Pickup Location", columns="Hour",
                       values="Booking ID", aggfunc="count", fill_value=0)
order = pivot.sum(axis=1).sort_values(ascending=False).index
plt.figure(figsize=(14, 10))
sns.heatmap(pivot.loc[order], cmap="YlOrRd", cbar_kws={"label": "Bookings"})
plt.title("Demand Heatmap: Pickup Area vs Hour of Day")
plt.tight_layout()
plt.savefig("heatmap_area_hour.png", dpi=150)
plt.close()
print("Saved heatmap_area_hour.png")

# ------------------------------------- 2. K-MEANS SEGMENTATION OF AREAS
area = df.groupby("Pickup Location").agg(
    Total_Bookings=("Booking ID", "count"),
    Avg_Fare=("Booking Value", "mean"),
    Cancel_Rate=("Cancelled", "mean"),
    Success_Rate=("Success", "mean"),
    Avg_Distance=("Ride Distance", "mean"),
    Avg_Rating=("Driver Ratings", "mean"),
).reset_index()

features = ["Total_Bookings", "Avg_Fare", "Cancel_Rate", "Avg_Distance", "Avg_Rating"]
X = StandardScaler().fit_transform(area[features])

# choose k with the elbow / silhouette method rather than guessing
from sklearn.metrics import silhouette_score
scores = {k: silhouette_score(X, KMeans(k, n_init=10, random_state=42).fit_predict(X))
          for k in range(2, 7)}
best_k = max(scores, key=scores.get)
print("Silhouette scores:", {k: round(v, 3) for k, v in scores.items()}, "-> k =", best_k)

area["Cluster"] = KMeans(best_k, n_init=10, random_state=42).fit_predict(X)

# human-readable tier names, ranked by a simple value score (demand + fare - cancellations)
value = (area.groupby("Cluster")
         .apply(lambda g: g["Total_Bookings"].mean() / area["Total_Bookings"].mean()
                + g["Avg_Fare"].mean() / area["Avg_Fare"].mean()
                - g["Cancel_Rate"].mean() / area["Cancel_Rate"].mean(),
                include_groups=False)
         .sort_values(ascending=False))
labels = ["Premium Hotspot", "Steady Performer", "Watch List", "Low Priority",
          "Underperformer", "Weak Zone"]
name_map = {c: labels[i] for i, c in enumerate(value.index)}
area["Segment"] = area["Cluster"].map(name_map)
area.round(3).to_csv("area_segments.csv", index=False)
print(area.groupby("Segment")[features].mean().round(2))

coords = PCA(2, random_state=42).fit_transform(X)
plt.figure(figsize=(9, 6))
sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=area["Segment"], s=110, palette="Set2")
for i, name in enumerate(area["Pickup Location"]):
    plt.annotate(name.replace("Area-", ""), (coords[i, 0], coords[i, 1]),
                 fontsize=7, xytext=(3, 3), textcoords="offset points")
plt.title(f"K-Means Segmentation of Pickup Areas (k={best_k}, PCA view)")
plt.xlabel("Principal component 1"); plt.ylabel("Principal component 2")
plt.tight_layout(); plt.savefig("area_clusters.png", dpi=150); plt.close()
print("Saved area_clusters.png and area_segments.csv")

# ------------------------- 3. IS THE "PEAK HOUR" STATISTICALLY REAL?
hourly = df.groupby("Hour").size()
chi2, p = chisquare(hourly.values)  # H0: demand is equal in every hour
spread = (hourly.max() - hourly.min()) / hourly.mean() * 100
print(f"\nChi-square test of uniform hourly demand: chi2={chi2:.1f}, p={p:.4f}")
print(f"Busiest hour: {hourly.idxmax()}:00 ({hourly.max()}), quietest: "
      f"{hourly.idxmin()}:00 ({hourly.min()}); spread = {spread:.1f}% of the mean")
print("-> Demand is "
      + ("NOT uniform (peaks are statistically significant)" if p < 0.05
         else "statistically flat: peak hours are real but small / close to random variation"))

# ------------------------ 4. CANCELLATION-RISK MODEL (Random Forest)
model_df = df[["Hour", "DayOfWeek", "IsWeekend", "Vehicle Type",
               "Pickup Location", "Cancelled"]].copy()
model_df["Area_ID"] = model_df["Pickup Location"].str.replace("Area-", "").astype(int)
X_ml = pd.get_dummies(model_df[["Hour", "DayOfWeek", "IsWeekend", "Area_ID", "Vehicle Type"]],
                      columns=["Vehicle Type"])
y_ml = model_df["Cancelled"]
X_tr, X_te, y_tr, y_te = train_test_split(X_ml, y_ml, test_size=0.25,
                                          random_state=42, stratify=y_ml)
rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced",
                            random_state=42, n_jobs=-1).fit(X_tr, y_tr)
auc = roc_auc_score(y_te, rf.predict_proba(X_te)[:, 1])
print(f"\nCancellation model ROC-AUC on held-out data: {auc:.3f}  (0.5 = random guessing)")
print(classification_report(y_te, rf.predict(X_te), digits=3))

imp = pd.Series(rf.feature_importances_, index=X_ml.columns).sort_values()
plt.figure(figsize=(8, 5))
imp.plot.barh(color="#FFD100", edgecolor="black")
plt.title(f"Cancellation Model - Feature Importance (AUC = {auc:.2f})")
plt.tight_layout(); plt.savefig("feature_importance.png", dpi=150); plt.close()
print("Saved feature_importance.png")
print("Interpretation: an AUC near 0.5 means time, area and vehicle type alone do not "
      "predict cancellations - they are driven by factors not in the dataset "
      "(driver availability, surge pricing, weather, traffic).")
