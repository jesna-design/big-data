"""
app_pyspark.py  (save as app.py on Render if that is your start file)
---------------------------------------------------------------------
Ola Mobility Insights dashboard. ALL data processing is done in PySpark.
Spark runs once per filter combination inside one cached function; only small
aggregated tables are converted to pandas for plotting.

requirements.txt:
    streamlit
    pandas
    plotly
    scipy
    pyspark==3.5.3
    install-jdk
Render env var:  PYTHON_VERSION = 3.11.9
"""
import os
import sys
import glob
import shutil

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import chisquare
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

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

if not os.path.exists(CSV_PATH):
    st.error(f"Data file not found: {CSV_PATH}. Put ola_bengaluru.csv in the same folder as this script.")
    st.stop()


# ---------- JAVA SETUP (only if the machine has no Java) ----------
@st.cache_resource(show_spinner="Setting up Java (first start only, about 1 minute)...")
def ensure_java():
    if shutil.which("java"):
        return "system java"
    found = glob.glob("/tmp/jdk/*/bin/java")
    if not found:
        import jdk
        shutil.rmtree("/tmp/jdk", ignore_errors=True)  # remove any broken partial download
        java_home = jdk.install("17", jre=True, path="/tmp/jdk")
    else:
        java_home = os.path.dirname(os.path.dirname(found[0]))
    os.environ["JAVA_HOME"] = java_home
    os.environ["PATH"] = java_home + "/bin:" + os.environ["PATH"]
    return java_home


# ---------- SPARK SESSION (memory-light settings) ----------
@st.cache_resource(show_spinner="Starting Spark...")
def get_spark():
    ensure_java()
    spark = (SparkSession.builder
             .appName("OlaMobilityDashboard")
             .master("local[2]")
             .config("spark.driver.memory", "384m")
             .config("spark.driver.extraJavaOptions", "-XX:+UseSerialGC")
             .config("spark.sql.shuffle.partitions", "2")
             .config("spark.default.parallelism", "2")
             .config("spark.ui.enabled", "false")
             .config("spark.ui.showConsoleProgress", "false")
             .config("spark.sql.ansi.enabled", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


SCHEMA = StructType([
    StructField("Date", StringType()), StructField("Time", StringType()),
    StructField("Booking ID", StringType()), StructField("Booking Status", StringType()),
    StructField("Customer ID", StringType()), StructField("Vehicle Type", StringType()),
    StructField("Pickup Location", StringType()), StructField("Drop Location", StringType()),
    StructField("Avg VTAT", DoubleType()), StructField("Avg CTAT", DoubleType()),
    StructField("Cancelled  by Customer", DoubleType()),
    StructField("Reason for Cancelling by Customer", StringType()),
    StructField("Cancelled Rides by Driver", DoubleType()),
    StructField("Reason for Cancelling by Driver", StringType()),
    StructField("Incomplete Rides", DoubleType()),
    StructField("Incomplete Rides Reason", StringType()),
    StructField("Booking Value", DoubleType()), StructField("Payment Method", StringType()),
    StructField("Ride Distance", DoubleType()), StructField("Driver Ratings", DoubleType()),
    StructField("Customer Rating", DoubleType()),
])


@st.cache_resource(show_spinner="Loading data with PySpark...")
def load_data():
    spark = get_spark()
    raw = spark.read.option("header", True).schema(SCHEMA).csv(CSV_PATH)
    sdf = (raw
           .dropna(subset=["Pickup Location", "Time"])
           .withColumn("Hour", F.split(F.col("Time"), ":").getItem(0).cast("int"))
           .withColumn("Date", F.to_date(F.col("Date"), "d/M/yyyy"))
           .dropna(subset=["Hour", "Date"])
           # Spark dayofweek: 1=Sunday..7=Saturday -> 0=Monday..6=Sunday
           .withColumn("DayOfWeek", (F.dayofweek("Date") + 5) % 7)
           .withColumn("Cancelled",
                       F.when(F.col("Booking Status").startswith("Cancelled"), 1).otherwise(0))
           .withColumn("Success",
                       F.when(F.col("Booking Status") == "Success", 1).otherwise(0))
           .cache())
    return sdf


@st.cache_data(show_spinner="Reading filter options...")
def filter_options():
    sdf = load_data()
    vehicles = sorted(r[0] for r in sdf.select("Vehicle Type").distinct().collect()
                      if r[0] is not None)
    dmin, dmax = sdf.agg(F.min("Date"), F.max("Date")).first()
    return vehicles, dmin, dmax


# ---------- ALL SPARK WORK IN ONE CACHED STEP ----------
@st.cache_data(show_spinner="Crunching the numbers with PySpark (about 20-60 seconds the first time)...")
def compute(vehicles, d0, d1):
    sdf = load_data()
    df = (sdf.filter(F.col("Vehicle Type").isin(list(vehicles)))
             .filter((F.col("Date") >= F.lit(d0)) & (F.col("Date") <= F.lit(d1))))

    # summary numbers in a single Spark job
    m = df.agg(
        F.count("*").alias("n"),
        F.avg("Success").alias("succ"),
        F.avg("Cancelled").alias("canc"),
        F.sum(F.when(F.col("Success") == 1, F.col("Booking Value"))).alias("rev"),
        F.avg(F.when(F.col("Success") == 1, F.col("Booking Value"))).alias("fare"),
        F.avg(F.when(F.col("Success") == 1, F.col("Ride Distance"))).alias("dist"),
        F.avg(F.when(F.col("Success") == 1, F.col("Driver Ratings"))).alias("rating"),
        F.avg(F.when(F.col("Success") == 1, F.col("Avg VTAT"))).alias("vtat"),
    ).first()
    if m["n"] == 0:
        return None

    out = {"summary": {k: (m[k] if m[k] is not None else 0) for k in
                       ["n", "succ", "canc", "rev", "fare", "dist", "rating", "vtat"]}}

    out["daily"] = (df.groupBy("Date").count().orderBy("Date")
                    .withColumnRenamed("count", "Bookings").toPandas())
    out["rev_vehicle"] = (df.groupBy("Vehicle Type")
                          .agg(F.sum(F.when(F.col("Success") == 1, F.col("Booking Value")))
                               .alias("Booking Value"))
                          .toPandas().fillna(0))
    out["status"] = df.groupBy("Booking Status").count().orderBy(F.desc("count")).toPandas()
    out["payment"] = (df.filter((F.col("Success") == 1) & F.col("Payment Method").isNotNull())
                      .groupBy("Payment Method").count().orderBy(F.desc("count")).toPandas())

    # area x hour table: bookings and cancel rate (basis for peak, heatmap, planner)
    out["ah"] = (df.groupBy("Pickup Location", "Hour")
                 .agg(F.count("*").alias("Bookings"), F.avg("Cancelled").alias("Cancel_Rate"))
                 .toPandas())
    out["area_fare"] = (df.filter(F.col("Success") == 1).groupBy("Pickup Location")
                        .agg(F.avg("Booking Value").alias("Avg_Fare")).toPandas())

    # weekday vs weekend
    wk = df.withColumn("Type", F.when(F.col("DayOfWeek") >= 5, "Weekend").otherwise("Weekday"))
    out["wk_hours"] = wk.groupBy("Type", "Hour").count().withColumnRenamed("count", "Bookings").toPandas()
    out["wk_days"] = {r["Type"]: r["days"] for r in
                      wk.groupBy("Type").agg(F.countDistinct("Date").alias("days")).collect()}

    # cancellation reasons
    out["drv_reasons"] = (df.filter((F.col("Booking Status") == "Cancelled by Driver")
                                    & F.col("Reason for Cancelling by Driver").isNotNull())
                          .groupBy("Reason for Cancelling by Driver").count()
                          .orderBy(F.desc("count")).toPandas())
    out["cust_reasons"] = (df.filter((F.col("Booking Status") == "Cancelled by Customer")
                                     & F.col("Reason for Cancelling by Customer").isNotNull())
                           .groupBy("Reason for Cancelling by Customer").count()
                           .orderBy(F.desc("count")).toPandas())
    return out


# ---------- START SPARK SAFELY (show the error instead of spinning forever) ----------
try:
    vehicles, dmin, dmax = filter_options()
except Exception as e:
    st.error("Spark could not start. If this says JAVA_GATEWAY_EXITED, Java is missing or the "
             "server ran out of memory.")
    st.exception(e)
    st.stop()

# ---------- SIDEBAR FILTERS ----------
st.sidebar.header(" Filters")
sel_vehicles = st.sidebar.multiselect("Vehicle type", vehicles, default=vehicles)
date_range = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    date_range = (dmin, dmax)

if not sel_vehicles:
    st.warning("Select at least one vehicle type.")
    st.stop()

try:
    data = compute(tuple(sel_vehicles), date_range[0], date_range[1])
except Exception as e:
    st.error("Spark failed while computing the dashboard data.")
    st.exception(e)
    st.stop()

if data is None:
    st.warning("No data for the selected filters.")
    st.stop()

S = data["summary"]
st.sidebar.caption(f"{int(S['n']):,} bookings in view")

ah = data["ah"]
PLOT = dict(template="plotly_dark", paper_bgcolor="#121212", plot_bgcolor="#1E1E1E")
YELLOW = "#FFD100"

# ---------- TABS ----------
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "OVERVIEW", " AREA → PEAK TIME", " TIME → PEAK AREA",
    "  HEATMAP", " CANCELLATIONS", " DISPATCH PLANNER"])

with tab0:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total bookings", f"{int(S['n']):,}")
    c2.metric("Completion rate", f"{S['succ'] * 100:.1f}%")
    c3.metric("Cancellation rate", f"{S['canc'] * 100:.1f}%")
    c4.metric("Total revenue", f"₹{S['rev'] / 1e6:.1f}M")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg fare", f"₹{S['fare']:,.0f}")
    c6.metric("Avg distance", f"{S['dist']:.1f} km")
    c7.metric("Avg driver rating", f"{S['rating']:.2f} ")
    c8.metric("Avg pickup ETA", f"{S['vtat']:.1f} min")

    colA, colB = st.columns(2)
    with colA:
        fig = px.line(data["daily"], x="Date", y="Bookings", title="Daily bookings",
                      markers=True, color_discrete_sequence=[YELLOW])
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        fig = px.bar(data["rev_vehicle"], x="Vehicle Type", y="Booking Value",
                     title="Revenue by vehicle type", color_discrete_sequence=[YELLOW])
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, use_container_width=True)
    colC, colD = st.columns(2)
    with colC:
        fig = px.pie(data["status"], names="Booking Status", values="count", hole=0.5,
                     title="Booking outcomes",
                     color_discrete_sequence=["#FFD100", "#FF6B35", "#4ECDC4", "#888888"])
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, use_container_width=True)
    with colD:
        fig = px.pie(data["payment"], names="Payment Method", values="count",
                     title="Payment methods",
                     color_discrete_sequence=px.colors.sequential.YlOrBr_r)
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, use_container_width=True)

    # statistical honesty check
    hourly = ah.groupby("Hour")["Bookings"].sum()
    _, p = chisquare(hourly.values)
    spread = (hourly.max() - hourly.min()) / hourly.mean() * 100
    verdict = ("statistically significant" if p < 0.05
               else "NOT statistically significant (demand is close to uniform across hours)")
    st.info(f"**Statistical check:** busiest hour {int(hourly.idxmax())}:00, quietest "
            f"{int(hourly.idxmin())}:00 — a {spread:.1f}% spread. Chi-square test for uniform "
            f"demand gives p = {p:.3f}, so the hourly peak is {verdict}.")

with tab1:
    areas = sorted(ah["Pickup Location"].unique())
    selected_area = st.selectbox("Select an area", areas)
    area_data = ah[ah["Pickup Location"] == selected_area].sort_values("Hour")

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
    hours = sorted(int(h) for h in ah["Hour"].unique())
    selected_hour = st.selectbox("Select an hour (24h format)", hours)
    hour_data = ah[ah["Hour"] == selected_hour]

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
    pivot = ah.pivot_table(index="Pickup Location", columns="Hour", values="Bookings",
                           aggfunc="sum", fill_value=0)
    pivot.columns = [int(c) for c in pivot.columns]
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index[:top_n]]
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="YlOrRd",
                    labels=dict(x="Hour", y="Area", color="Bookings"))
    fig.update_layout(**PLOT, height=max(400, top_n * 22))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Weekday vs weekend pattern")
    hr = data["wk_hours"].copy()
    hr["Bookings per day"] = hr.apply(lambda r: r["Bookings"] / data["wk_days"][r["Type"]], axis=1)
    hr = hr.sort_values(["Type", "Hour"])
    fig = px.line(hr, x="Hour", y="Bookings per day", color="Type", markers=True,
                  color_discrete_sequence=[YELLOW, "#4ECDC4"])
    fig.update_layout(**PLOT)
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Cancellation analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(data["drv_reasons"], y="Reason for Cancelling by Driver", x="count",
                     orientation="h", title="Why drivers cancel",
                     color_discrete_sequence=[YELLOW])
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(data["cust_reasons"], y="Reason for Cancelling by Customer", x="count",
                     orientation="h", title="Why customers cancel",
                     color_discrete_sequence=["#FF6B35"])
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, use_container_width=True)

    # weighted cancel rates computed from the Spark area x hour table
    tmp = ah.assign(Canc=ah["Cancel_Rate"] * ah["Bookings"])
    by_area = tmp.groupby("Pickup Location").agg(
        Bookings=("Bookings", "sum"), Canc=("Canc", "sum")).reset_index()
    by_area["Cancel_Rate"] = (by_area["Canc"] / by_area["Bookings"] * 100).round(1)
    by_area = by_area.drop(columns="Canc").sort_values("Cancel_Rate", ascending=False)
    colX, colY = st.columns(2)
    with colX:
        st.markdown("** Highest cancellation-rate areas**")
        st.dataframe(by_area.head(8).rename(columns={"Cancel_Rate": "Cancel rate %"}),
                     hide_index=True, use_container_width=True)
    with colY:
        st.markdown("** Most reliable areas**")
        st.dataframe(by_area.tail(8).iloc[::-1].rename(columns={"Cancel_Rate": "Cancel rate %"}),
                     hide_index=True, use_container_width=True)

    hc = tmp.groupby("Hour").agg(Bookings=("Bookings", "sum"), Canc=("Canc", "sum")).reset_index()
    hc["Cancel rate %"] = hc["Canc"] / hc["Bookings"] * 100
    fig = px.line(hc.sort_values("Hour"), x="Hour", y="Cancel rate %", markers=True,
                  title="Cancellation rate by hour", color_discrete_sequence=["#FF6B35"])
    fig.update_layout(**PLOT)
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader(" Driver dispatch planner")
    st.caption("Where should idle drivers go at a given hour? Areas are scored on demand, "
               "fare value and reliability (low cancellation rate).")
    plan_hour = st.select_slider("Hour of day", options=list(range(24)), value=18)

    h = (ah[ah["Hour"] == plan_hour]
         .rename(columns={"Bookings": "Demand"})
         [["Pickup Location", "Demand", "Cancel_Rate"]].copy())

    if h.empty:
        st.warning("No bookings at this hour for the selected filters.")
    else:
        fare_pd = data["area_fare"]
        overall_fare = fare_pd["Avg_Fare"].mean() if not fare_pd.empty else 0.0
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
