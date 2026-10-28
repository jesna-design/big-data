"""
app_pyspark.py
--------------
Streamlit dashboard for Ola Mobility Insights, with all data processing done in PySpark.
Spark does the loading, filtering and aggregation; only small aggregated results are
converted to pandas for plotting with Plotly / Streamlit.

Requirements:  pip install streamlit pyspark pandas plotly scipy
               Java 8 / 11 / 17 installed (JAVA_HOME set)
Run:           streamlit run app_pyspark.py
"""
import os
import sys
import shutil
if shutil.which("java") is None:
    import jdk
    os.environ["JAVA_HOME"] = jdk.install("17", jre=True, path="/tmp/jdk")
    os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import chisquare
from pyspark.sql import SparkSession, functions as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "ola_bengaluru.csv")

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Ola Mobility Insights", page_icon="🚕", layout="wide")

# ---------- OLA-STYLE THEME (CSS) ----------
st.markdown("""
<style>
    .stApp { background-color: #121212; color: #F5F5F5; }
    .ola-header {
        background-color: #FFD100; padding: 22px 20px; border-radius: 10px;
        margin-bottom: 25px; display: flex; align-items: center; justify-content: center;
    }
    .ola-header h1 {
        color: #111111; font-family: 'Trebuchet MS', sans-serif; font-weight: 800;
        margin: 0; font-size: 32px; letter-spacing: 1px;
    }
    .ola-header p { color: #222222; margin: 4px 0 0 0; font-size: 14px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E1E1E; border-radius: 8px 8px 0 0; padding: 10px 20px; color: #F5F5F5;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFD100 !important; color: #111111 !important; font-weight: 700;
    }
    label, .stMarkdown p { color: #F5F5F5 !important; }
    .result-card {
        background-color: #1E1E1E; border-left: 6px solid #FFD100; padding: 18px 20px;
        border-radius: 8px; margin-top: 15px; margin-bottom: 15px;
    }
    .result-card h3 { color: #FFD100; margin: 0 0 6px 0; font-size: 16px; }
    .result-card p { color: #FFFFFF !important; font-size: 22px; font-weight: 700; margin: 0; }
    .result-card span { color: #BBBBBB; font-size: 13px; }
    .ola-footer { text-align: center; color: #777777; font-size: 12px; margin-top: 30px; }
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<div class="ola-header">
    <div>
        <h1>OLA MOBILITY INSIGHTS</h1>
        <p style="text-align:center;">Demand, cancellations & driver dispatch analytics — Bengaluru (PySpark)</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------- SPARK SESSION ----------
@st.cache_resource
def get_spark():
    spark = (SparkSession.builder
             .appName("OlaMobilityDashboard")
             .master("local[2]")
             .config("spark.driver.memory", "512m")
             .config("spark.sql.shuffle.partitions", "4")
             .config("spark.ui.enabled", "false")
             .config("spark.sql.ansi.enabled", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


@st.cache_resource
def load_data():
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType
    S, D = StringType(), DoubleType()
    schema = StructType([
        StructField("Date", S), StructField("Time", S), StructField("Booking ID", S),
        StructField("Booking Status", S), StructField("Customer ID", S),
        StructField("Vehicle Type", S), StructField("Pickup Location", S),
        StructField("Drop Location", S), StructField("Avg VTAT", D), StructField("Avg CTAT", D),
        StructField("Cancelled  by Customer", D), StructField("Reason for Cancelling by Customer", S),
        StructField("Cancelled Rides by Driver", D), StructField("Reason for Cancelling by Driver", S),
        StructField("Incomplete Rides", D), StructField("Incomplete Rides Reason", S),
        StructField("Booking Value", D), StructField("Payment Method", S),
        StructField("Ride Distance", D), StructField("Driver Ratings", D),
        StructField("Customer Rating", D),
    ])
    spark = get_spark()
    raw = spark.read.option("header", True).schema(schema).csv(CSV_PATH)
    sdf = (raw
           .dropna(subset=["Pickup Location", "Time"])
           .withColumn("Hour", F.split(F.col("Time"), ":").getItem(0).cast("int"))
           .withColumn("Date", F.to_date(F.col("Date"), "d/M/yyyy"))
           .dropna(subset=["Hour", "Date"])
           .withColumn("DayOfWeek", (F.dayofweek("Date") + 5) % 7)
           .withColumn("Cancelled",
                       F.when(F.col("Booking Status").startswith("Cancelled"), 1).otherwise(0))
           .withColumn("Success",
                       F.when(F.col("Booking Status") == "Success", 1).otherwise(0))
           .cache())
    return sdf


full_df = load_data()


@st.cache_data
def filter_options():
    vehicles = sorted(r[0] for r in full_df.select("Vehicle Type").distinct().collect()
                      if r[0] is not None)
    dmin, dmax = full_df.agg(F.min("Date"), F.max("Date")).first()
    return vehicles, dmin, dmax


# ---------- SIDEBAR FILTERS ----------
st.sidebar.header(" Filters")
vehicles, dmin, dmax = filter_options()
sel_vehicles = st.sidebar.multiselect("Vehicle type", vehicles, default=vehicles)
date_range = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    date_range = (dmin, dmax)

df = (full_df
      .filter(F.col("Vehicle Type").isin(sel_vehicles))
      .filter((F.col("Date") >= F.lit(date_range[0])) & (F.col("Date") <= F.lit(date_range[1]))))

n_rows = df.count()
if n_rows == 0:
    st.warning("No data for the selected filters.")
    st.stop()
st.sidebar.caption(f"{n_rows:,} bookings in view")

df = df.cache()
completed = df.filter(F.col("Success") == 1)

# (Pickup Location, Hour) booking counts - 50 x 24 rows max, so safe to bring to pandas
counts = (df.groupBy("Pickup Location", "Hour").count()
          .withColumnRenamed("count", "Bookings").toPandas())

PLOT = dict(template="plotly_dark", paper_bgcolor="#121212", plot_bgcolor="#1E1E1E")
YELLOW = "#FFD100"

# ---------- TABS ----------
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "OVERVIEW", " AREA → PEAK TIME", " TIME → PEAK AREA",
    "  HEATMAP", " CANCELLATIONS", " DISPATCH PLANNER"])

with tab0:
    m = df.agg(F.avg("Success").alias("succ"), F.avg("Cancelled").alias("canc")).first()
    c = completed.agg(F.sum("Booking Value").alias("rev"),
                      F.avg("Booking Value").alias("fare"),
                      F.avg("Ride Distance").alias("dist"),
                      F.avg("Driver Ratings").alias("rating"),
                      F.avg("Avg VTAT").alias("vtat")).first()
    rev_total = c["rev"] or 0
    fare = c["fare"] or 0
    dist = c["dist"] or 0
    rating = c["rating"] or 0
    vtat = c["vtat"] or 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total bookings", f"{n_rows:,}")
    c2.metric("Completion rate", f"{(m['succ'] or 0) * 100:.1f}%")
    c3.metric("Cancellation rate", f"{(m['canc'] or 0) * 100:.1f}%")
    c4.metric("Total revenue", f"₹{rev_total / 1e6:.1f}M")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg fare", f"₹{fare:,.0f}")
    c6.metric("Avg distance", f"{dist:.1f} km")
    c7.metric("Avg driver rating", f"{rating:.2f} ")
    c8.metric("Avg pickup ETA", f"{vtat:.1f} min")

    colA, colB = st.columns(2)
    with colA:
        daily = (df.groupBy("Date").count().orderBy("Date")
                 .withColumnRenamed("count", "Bookings").toPandas())
        fig = px.line(daily, x="Date", y="Bookings", title="Daily bookings", markers=True,
                      color_discrete_sequence=[YELLOW])
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        rev = (completed.groupBy("Vehicle Type")
               .agg(F.sum("Booking Value").alias("Booking Value")).toPandas())
        fig = px.bar(rev, x="Vehicle Type", y="Booking Value", title="Revenue by vehicle type",
                     color_discrete_sequence=[YELLOW])
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, use_container_width=True)
    colC, colD = st.columns(2)
    with colC:
        status = df.groupBy("Booking Status").count().orderBy(F.desc("count")).toPandas()
        fig = px.pie(status, names="Booking Status", values="count", hole=0.5,
                     title="Booking outcomes",
                     color_discrete_sequence=["#FFD100", "#FF6B35", "#4ECDC4", "#888888"])
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, use_container_width=True)
    with colD:
        pay = (completed.filter(F.col("Payment Method").isNotNull())
               .groupBy("Payment Method").count().orderBy(F.desc("count")).toPandas())
        fig = px.pie(pay, names="Payment Method", values="count", title="Payment methods",
                     color_discrete_sequence=px.colors.sequential.YlOrBr_r)
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, use_container_width=True)

    # statistical honesty check
    hourly = counts.groupby("Hour")["Bookings"].sum()
    _, p = chisquare(hourly.values)
    spread = (hourly.max() - hourly.min()) / hourly.mean() * 100
    verdict = ("statistically significant" if p < 0.05
               else "NOT statistically significant (demand is close to uniform across hours)")
    st.info(f"**Statistical check:** busiest hour {hourly.idxmax()}:00, quietest "
            f"{hourly.idxmin()}:00 — a {spread:.1f}% spread. Chi-square test for uniform "
            f"demand gives p = {p:.3f}, so the hourly peak is {verdict}.")

with tab1:
    areas = sorted(counts["Pickup Location"].unique())
    selected_area = st.selectbox("Select an area", areas)

    area_data = counts[counts["Pickup Location"] == selected_area].sort_values("Hour")

    if not area_data.empty:
        peak_row = area_data.loc[area_data["Bookings"].idxmax()]
        st.markdown(f"""
        <div class="result-card">
            <h3>PEAK HOUR</h3>
            <p>{int(peak_row['Hour'])}:00</p>
            <span>{int(peak_row['Bookings'])} bookings recorded at this hour in {selected_area}</span>
        </div>
        """, unsafe_allow_html=True)
        st.bar_chart(area_data.set_index("Hour")["Bookings"], color="#FFD100")
    else:
        st.warning("No data found for this area.")

with tab2:
    hours = sorted(int(h) for h in counts["Hour"].unique())
    selected_hour = st.selectbox("Select an hour (24h format)", hours)

    hour_data = counts[counts["Hour"] == selected_hour]

    if not hour_data.empty:
        peak_row = hour_data.loc[hour_data["Bookings"].idxmax()]
        st.markdown(f"""
        <div class="result-card">
            <h3>BUSIEST AREA</h3>
            <p>{peak_row['Pickup Location']}</p>
            <span>{int(peak_row['Bookings'])} bookings recorded at {selected_hour}:00</span>
        </div>
        """, unsafe_allow_html=True)
        top10 = hour_data.sort_values("Bookings", ascending=False).head(10)
        st.bar_chart(top10.set_index("Pickup Location")["Bookings"], color="#FFD100")
    else:
        st.warning("No data found for this hour.")

with tab3:
    st.subheader("Demand heatmap: area × hour")
    top_n = st.slider("Number of busiest areas to show", 5, 50, 20)

    hour_list = sorted(int(h) for h in counts["Hour"].unique())
    pivot = (df.groupBy("Pickup Location").pivot("Hour", hour_list).count()
             .na.fill(0).toPandas().set_index("Pickup Location"))
    pivot.columns = [int(c) for c in pivot.columns]
    pivot = pivot[sorted(pivot.columns)]
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index[:top_n]]
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="YlOrRd",
                    labels=dict(x="Hour", y="Area", color="Bookings"))
    fig.update_layout(**PLOT, height=max(400, top_n * 22))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Weekday vs weekend pattern")
    wk = df.withColumn("Type", F.when(F.col("DayOfWeek") >= 5, "Weekend").otherwise("Weekday"))
    days_per_type = {r["Type"]: r["days"] for r in
                     wk.groupBy("Type").agg(F.countDistinct("Date").alias("days")).collect()}
    hr = wk.groupBy("Type", "Hour").count().withColumnRenamed("count", "Bookings").toPandas()
    hr["Bookings per day"] = hr.apply(lambda r: r["Bookings"] / days_per_type[r["Type"]], axis=1)
    hr = hr.sort_values(["Type", "Hour"])
    fig = px.line(hr, x="Hour", y="Bookings per day", color="Type", markers=True,
                  color_discrete_sequence=[YELLOW, "#4ECDC4"])
    fig.update_layout(**PLOT)
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Cancellation analysis")
    col1, col2 = st.columns(2)
    with col1:
        r = (df.filter((F.col("Booking Status") == "Cancelled by Driver")
                       & F.col("Reason for Cancelling by Driver").isNotNull())
             .groupBy("Reason for Cancelling by Driver").count()
             .orderBy(F.desc("count")).toPandas())
        fig = px.bar(r, y="Reason for Cancelling by Driver", x="count", orientation="h",
                     title="Why drivers cancel", color_discrete_sequence=[YELLOW])
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        r = (df.filter((F.col("Booking Status") == "Cancelled by Customer")
                       & F.col("Reason for Cancelling by Customer").isNotNull())
             .groupBy("Reason for Cancelling by Customer").count()
             .orderBy(F.desc("count")).toPandas())
        fig = px.bar(r, y="Reason for Cancelling by Customer", x="count", orientation="h",
                     title="Why customers cancel", color_discrete_sequence=["#FF6B35"])
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, use_container_width=True)

    by_area = (df.groupBy("Pickup Location")
               .agg(F.count("*").alias("Bookings"), F.avg("Cancelled").alias("Cancel_Rate"))
               .toPandas())
    by_area["Cancel_Rate"] = (by_area["Cancel_Rate"] * 100).round(1)
    by_area = by_area.sort_values("Cancel_Rate", ascending=False)
    colX, colY = st.columns(2)
    with colX:
        st.markdown("** Highest cancellation-rate areas**")
        st.dataframe(by_area.head(8).rename(columns={"Cancel_Rate": "Cancel rate %"}),
                     hide_index=True, use_container_width=True)
    with colY:
        st.markdown("** Most reliable areas**")
        st.dataframe(by_area.tail(8).iloc[::-1].rename(columns={"Cancel_Rate": "Cancel rate %"}),
                     hide_index=True, use_container_width=True)

    hc = (df.groupBy("Hour").agg((F.avg("Cancelled") * 100).alias("Cancel rate %"))
          .orderBy("Hour").toPandas())
    fig = px.line(hc, x="Hour", y="Cancel rate %", markers=True,
                  title="Cancellation rate by hour", color_discrete_sequence=["#FF6B35"])
    fig.update_layout(**PLOT)
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader(" Driver dispatch planner")
    st.caption("Where should idle drivers go at a given hour? Areas are scored on demand, "
               "fare value and reliability (low cancellation rate).")
    plan_hour = st.select_slider("Hour of day", options=list(range(24)), value=18)

    h = (df.filter(F.col("Hour") == plan_hour)
         .groupBy("Pickup Location")
         .agg(F.count("*").alias("Demand"), F.avg("Cancelled").alias("Cancel_Rate"))
         .toPandas())
    fare_pd = (completed.groupBy("Pickup Location")
               .agg(F.avg("Booking Value").alias("Avg_Fare")).toPandas())
    overall_fare = completed.agg(F.avg("Booking Value")).first()[0] or 0.0

    if h.empty:
        st.warning("No bookings at this hour for the selected filters.")
    else:
        h = h.merge(fare_pd, on="Pickup Location", how="left").fillna({"Avg_Fare": overall_fare})

        def norm(s):
            return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0

        h["Score"] = (0.5 * norm(h["Demand"]) + 0.3 * norm(h["Avg_Fare"])
                      + 0.2 * (1 - norm(h["Cancel_Rate"]))) * 100
        h = h.sort_values("Score", ascending=False)
        top5 = h.head(5)

        st.markdown(f"""
        <div class="result-card">
            <h3>RECOMMENDED STAGING AREA AT {plan_hour}:00</h3>
            <p>{top5.iloc[0]['Pickup Location']}</p>
            <span>Score {top5.iloc[0]['Score']:.0f}/100 · {int(top5.iloc[0]['Demand'])} bookings ·
            avg fare ₹{top5.iloc[0]['Avg_Fare']:,.0f} · {top5.iloc[0]['Cancel_Rate']*100:.1f}% cancellations</span>
        </div>""", unsafe_allow_html=True)

        fig = px.bar(top5, x="Pickup Location", y="Score", title="Top 5 areas to position drivers",
                     hover_data=["Demand", "Avg_Fare", "Cancel_Rate"],
                     color_discrete_sequence=[YELLOW])
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Score = 50% demand + 30% average fare + 20% reliability, each scaled 0–1 "
                   "across areas for the selected hour.")

st.markdown('<div class="ola-footer">Data source: Bengaluru Ola Cabs dataset · Urban Mobility Optimization Project</div>', unsafe_allow_html=True)
