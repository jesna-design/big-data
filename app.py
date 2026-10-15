import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import chisquare

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Ola Mobility Insights", page_icon="🚕", layout="wide")

# ---------- OLA-STYLE THEME (CSS) ----------
st.markdown("""
<style>
    /* Overall app background */
    .stApp {
        background-color: #121212;
        color: #F5F5F5;
    }

    /* Header bar */
    .ola-header {
        background-color: #FFD100;
        padding: 22px 20px;
        border-radius: 10px;
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .ola-header h1 {
        color: #111111;
        font-family: 'Trebuchet MS', sans-serif;
        font-weight: 800;
        margin: 0;
        font-size: 32px;
        letter-spacing: 1px;
    }
    .ola-header p {
        color: #222222;
        margin: 4px 0 0 0;
        font-size: 14px;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E1E1E;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #F5F5F5;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFD100 !important;
        color: #111111 !important;
        font-weight: 700;
    }

    /* Selectbox label */
    label, .stMarkdown p {
        color: #F5F5F5 !important;
    }

    /* Result card */
    .result-card {
        background-color: #1E1E1E;
        border-left: 6px solid #FFD100;
        padding: 18px 20px;
        border-radius: 8px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .result-card h3 {
        color: #FFD100;
        margin: 0 0 6px 0;
        font-size: 16px;
    }
    .result-card p {
        color: #FFFFFF !important;
        font-size: 22px;
        font-weight: 700;
        margin: 0;
    }
    .result-card span {
        color: #BBBBBB;
        font-size: 13px;
    }

    /* Footer */
    .ola-footer {
        text-align: center;
        color: #777777;
        font-size: 12px;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<div class="ola-header">
    <div>
        <h1>OLA MOBILITY INSIGHTS</h1>
        <p style="text-align:center;">Demand, cancellations & driver dispatch analytics — Bengaluru</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- LOAD & PREPARE DATA ----------
@st.cache_data
def load_data():
    df = pd.read_csv("ola_bengaluru.csv")
    df = df.dropna(subset=["Pickup Location", "Time"])
    df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dt.hour
    df = df.dropna(subset=["Hour"])
    df["Hour"] = df["Hour"].astype(int)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df["Weekday"] = df["Date"].dt.day_name()
    df["Cancelled"] = df["Booking Status"].str.startswith("Cancelled").astype(int)
    df["Success"] = (df["Booking Status"] == "Success").astype(int)
    return df

full_df = load_data()

# ---------- SIDEBAR FILTERS ----------
st.sidebar.header(" Filters")
vehicles = sorted(full_df["Vehicle Type"].unique())
sel_vehicles = st.sidebar.multiselect("Vehicle type", vehicles, default=vehicles)
dmin, dmax = full_df["Date"].min().date(), full_df["Date"].max().date()
date_range = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
if len(date_range) != 2:
    date_range = (dmin, dmax)

df = full_df[
    full_df["Vehicle Type"].isin(sel_vehicles)
    & (full_df["Date"].dt.date >= date_range[0])
    & (full_df["Date"].dt.date <= date_range[1])
]
if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()
st.sidebar.caption(f"{len(df):,} bookings in view")

counts = df.groupby(["Pickup Location", "Hour"]).size().reset_index(name="Bookings")

PLOT = dict(template="plotly_dark", paper_bgcolor="#121212", plot_bgcolor="#1E1E1E")
YELLOW = "#FFD100"

# ---------- TABS ----------
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "OVERVIEW", " AREA → PEAK TIME", " TIME → PEAK AREA",
    "  HEATMAP", " CANCELLATIONS", " DISPATCH PLANNER"])

with tab0:
    completed = df[df["Success"] == 1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total bookings", f"{len(df):,}")
    c2.metric("Completion rate", f"{df['Success'].mean()*100:.1f}%")
    c3.metric("Cancellation rate", f"{df['Cancelled'].mean()*100:.1f}%")
    c4.metric("Total revenue", f"₹{completed['Booking Value'].sum()/1e6:.1f}M")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg fare", f"₹{completed['Booking Value'].mean():,.0f}")
    c6.metric("Avg distance", f"{completed['Ride Distance'].mean():.1f} km")
    c7.metric("Avg driver rating", f"{completed['Driver Ratings'].mean():.2f} ")
    c8.metric("Avg pickup ETA", f"{completed['Avg VTAT'].mean():.1f} min")

    colA, colB = st.columns(2)
    with colA:
        daily = df.groupby(df["Date"].dt.date).size().reset_index(name="Bookings")
        fig = px.line(daily, x="Date", y="Bookings", title="Daily bookings", markers=True,
                      color_discrete_sequence=[YELLOW])
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        rev = completed.groupby("Vehicle Type")["Booking Value"].sum().reset_index()
        fig = px.bar(rev, x="Vehicle Type", y="Booking Value", title="Revenue by vehicle type",
                     color_discrete_sequence=[YELLOW])
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, use_container_width=True)
    colC, colD = st.columns(2)
    with colC:
        status = df["Booking Status"].value_counts().reset_index()
        fig = px.pie(status, names="Booking Status", values="count", hole=0.5,
                     title="Booking outcomes",
                     color_discrete_sequence=["#FFD100", "#FF6B35", "#4ECDC4", "#888888"])
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, use_container_width=True)
    with colD:
        pay = completed["Payment Method"].value_counts().reset_index()
        fig = px.pie(pay, names="Payment Method", values="count", title="Payment methods",
                     color_discrete_sequence=px.colors.sequential.YlOrBr_r)
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, use_container_width=True)

    # statistical honesty check
    hourly = df.groupby("Hour").size()
    _, p = chisquare(hourly.values)
    spread = (hourly.max() - hourly.min()) / hourly.mean() * 100
    verdict = ("statistically significant" if p < 0.05
               else "NOT statistically significant (demand is close to uniform across hours)")
    st.info(f"**Statistical check:** busiest hour {hourly.idxmax()}:00, quietest "
            f"{hourly.idxmin()}:00 — a {spread:.1f}% spread. Chi-square test for uniform "
            f"demand gives p = {p:.3f}, so the hourly peak is {verdict}.")

with tab1:
    areas = sorted(df["Pickup Location"].unique())
    selected_area = st.selectbox("Select an area", areas)

    area_data = counts[counts["Pickup Location"] == selected_area]

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
    hours = sorted(df["Hour"].unique())
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
    pivot = df.pivot_table(index="Pickup Location", columns="Hour", values="Booking ID",
                           aggfunc="count", fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index[:top_n]]
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="YlOrRd",
                    labels=dict(x="Hour", y="Area", color="Bookings"))
    fig.update_layout(**PLOT, height=max(400, top_n * 22))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Weekday vs weekend pattern")
    wk = df.copy()
    wk["Type"] = wk["Date"].dt.dayofweek.map(lambda d: "Weekend" if d >= 5 else "Weekday")
    days_per_type = wk.groupby("Type")["Date"].nunique()
    hr = wk.groupby(["Type", "Hour"]).size().reset_index(name="Bookings")
    hr["Bookings per day"] = hr.apply(lambda r: r["Bookings"] / days_per_type[r["Type"]], axis=1)
    fig = px.line(hr, x="Hour", y="Bookings per day", color="Type", markers=True,
                  color_discrete_sequence=[YELLOW, "#4ECDC4"])
    fig.update_layout(**PLOT)
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Cancellation analysis")
    col1, col2 = st.columns(2)
    with col1:
        r = df[df["Booking Status"] == "Cancelled by Driver"]["Reason for Cancelling by Driver"] \
            .value_counts().reset_index()
        fig = px.bar(r, y="Reason for Cancelling by Driver", x="count", orientation="h",
                     title="Why drivers cancel", color_discrete_sequence=[YELLOW])
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        r = df[df["Booking Status"] == "Cancelled by Customer"]["Reason for Cancelling by Customer"] \
            .value_counts().reset_index()
        fig = px.bar(r, y="Reason for Cancelling by Customer", x="count", orientation="h",
                     title="Why customers cancel", color_discrete_sequence=["#FF6B35"])
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, use_container_width=True)

    by_area = df.groupby("Pickup Location").agg(
        Bookings=("Booking ID", "count"), Cancel_Rate=("Cancelled", "mean")).reset_index()
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

    hc = df.groupby("Hour")["Cancelled"].mean().mul(100).reset_index(name="Cancel rate %")
    fig = px.line(hc, x="Hour", y="Cancel rate %", markers=True,
                  title="Cancellation rate by hour", color_discrete_sequence=["#FF6B35"])
    fig.update_layout(**PLOT)
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader(" Driver dispatch planner")
    st.caption("Where should idle drivers go at a given hour? Areas are scored on demand, "
               "fare value and reliability (low cancellation rate).")
    plan_hour = st.select_slider("Hour of day", options=list(range(24)), value=18)

    comp = df[df["Success"] == 1]
    h = df[df["Hour"] == plan_hour].groupby("Pickup Location").agg(
        Demand=("Booking ID", "count"), Cancel_Rate=("Cancelled", "mean")).reset_index()
    fare = comp.groupby("Pickup Location")["Booking Value"].mean().rename("Avg_Fare").reset_index()
    h = h.merge(fare, on="Pickup Location", how="left").fillna({"Avg_Fare": comp["Booking Value"].mean()})

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
                 hover_data=["Demand", "Avg_Fare", "Cancel_Rate"], color_discrete_sequence=[YELLOW])
    fig.update_layout(**PLOT)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Score = 50% demand + 30% average fare + 20% reliability, each scaled 0–1 "
               "across areas for the selected hour.")

st.markdown('<div class="ola-footer">Data source: Bengaluru Ola Cabs dataset · Urban Mobility Optimization Project</div>', unsafe_allow_html=True)
