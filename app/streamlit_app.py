"""
Streamlit app: прогноз цены за м² по регионам России.
"""

import os, sys, json, pickle, warnings
warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

MODELS_DIR = "models/prophet"
TS_PATH    = "data/processed/price_timeseries_2018_2026.csv"
MACRO_PATH = "data/processed/macro_quarterly.csv"
META_PATH  = f"{MODELS_DIR}/meta.json"

# ── Helpers ───────────────────────────────────────────────────────────────────
QUARTER_NAMES   = {1: "1 кв.", 2: "2 кв.", 3: "3 кв.", 4: "4 кв."}
QUARTER_SEASONS = {1: "Зима", 2: "Весна", 3: "Лето", 4: "Осень"}

def fmt_q(d) -> str:
    return f"{QUARTER_NAMES[d.quarter]} {d.year}"

# Inflation presets: label → % value
INFLATION_PRESETS = {
    "Низкий рост цен (~4%)":        4.0,
    "Умеренный рост цен (~7%)":     7.0,
    "Высокий рост цен (~12%)":     12.0,
    "Очень высокий рост цен (~17%)": 17.0,
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Прогноз цен на жильё",
    page_icon=None,
    layout="wide",
)

# ── Load data ─────────────────────────────────────────────────────────────────
def _full_region_name(name: str, type_: str) -> str:
    t = type_.strip()
    if t == "г":
        return name
    elif t == "обл":
        return f"{name} область"
    elif t == "край":
        return f"{name} край"
    elif t == "Респ":
        return f"Республика {name}"
    elif t == "АО":
        return f"{name} автономный округ"
    elif t == "Аобл":
        return f"{name} автономная область"
    return name

@st.cache_data
def load_data():
    ts    = pd.read_csv(TS_PATH, parse_dates=["period"])
    macro = pd.read_csv(MACRO_PATH, parse_dates=["period"])
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    # Build full region names from reference CSV
    try:
        ref = pd.read_csv("data/russia_region_codes.csv", encoding="utf-8")
        ref["region"] = ref["kladr_id"].astype(str).str.zfill(13).str[:2].astype(int)
        full_names = {
            row["region"]: _full_region_name(row["name"], row["type"])
            for _, row in ref.iterrows()
        }
        meta["regions"] = {k: full_names.get(int(k), v) for k, v in meta["regions"].items()}
    except Exception:
        pass
    return ts, macro, meta

@st.cache_resource
def load_model(region_code: int):
    with open(f"{MODELS_DIR}/region_{region_code}.pkl", "rb") as f:
        return pickle.load(f)

ts, macro, meta = load_data()
# Region 14 (Якутия) excluded: prices in source data are unreliable (biased downward)
EXCLUDED_REGIONS = {14}
regions_dict   = {int(k): v for k, v in meta["regions"].items() if int(k) not in EXCLUDED_REGIONS}
regions_sorted = sorted(regions_dict.items(), key=lambda x: x[1])

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Параметры прогноза")

# Region — names only, no codes
region_name_selected = st.sidebar.selectbox(
    "Регион",
    options=[name for _, name in regions_sorted],
    index=next(i for i, (c, _) in enumerate(regions_sorted) if c == 77),
)
region_code = next(c for c, n in regions_sorted if n == region_name_selected)

st.sidebar.markdown("---")
st.sidebar.subheader("Экономический сценарий")

cbr_scenario = st.sidebar.slider(
    "Ключевая ставка ЦБ (%)",
    min_value=4.0, max_value=25.0,
    value=16.0, step=0.5,
    help="Чем ниже ставка, тем дешевле ипотека и выше спрос на жильё.",
)

inflation_label = st.sidebar.radio(
    "Рост цен в стране",
    options=list(INFLATION_PRESETS.keys()),
    index=1,
    help="Общий уровень роста цен влияет на стоимость недвижимости.",
)
inflation_scenario = INFLATION_PRESETS[inflation_label]

st.sidebar.markdown("---")

# Forecast always starts one full quarter after last known macro data
_last_macro     = macro["period"].max()
_forecast_start = _last_macro + pd.DateOffset(months=3)   # unambiguous +1 quarter

# Build options: from forecast_start to Q4 2028
_options = {}
d = _forecast_start
while d <= pd.Timestamp("2028-10-01"):
    label = f"{QUARTER_SEASONS[d.quarter]} {d.year}"
    _options[label] = d
    d += pd.DateOffset(months=3)

# Two-step selection: year → season
_available_years = sorted({d.year for d in _options.values()})
forecast_year = st.sidebar.selectbox("Показать прогноз до — год:", _available_years,
                                      index=min(1, len(_available_years) - 1))

_seasons_for_year = {QUARTER_SEASONS[d.quarter]: d
                     for label, d in _options.items() if d.year == forecast_year}
forecast_season_label = st.sidebar.selectbox("Сезон:", list(_seasons_for_year.keys()))
forecast_end = _seasons_for_year[forecast_season_label]

# ── Compute forecast quarters ─────────────────────────────────────────────────
future_dates = pd.date_range(start=_forecast_start, end=forecast_end, freq="QS")
n_quarters   = len(future_dates)

# ── Build forecast ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_forecast(region_code: int, cbr: float, inflation: float,
                 future_dates_list: list):
    model     = load_model(region_code)
    known     = macro[["period", "cbr_rate", "inflation"]].copy()
    fut_dates = pd.DatetimeIndex(future_dates_list)
    fut_macro = pd.DataFrame({"period": fut_dates, "cbr_rate": cbr, "inflation": inflation})
    full      = pd.concat([known, fut_macro]).drop_duplicates("period").sort_values("period")
    fc        = model.predict(full.rename(columns={"period": "ds"}))
    return fc

with st.spinner("Считаю прогноз..."):
    fc = get_forecast(region_code, cbr_scenario, inflation_scenario,
                      future_dates.tolist())

hist      = ts[ts["region"] == region_code].sort_values("period")
train_end = pd.Timestamp(meta["train_end"])

# ── Header ────────────────────────────────────────────────────────────────────
st.title(f"Прогноз цены за м² — {region_name_selected}")

# Metrics
last_price          = float(hist["price_sqm"].iloc[-1]) if len(hist) else None
last_period_label   = fmt_q(hist["period"].iloc[-1]) if len(hist) else "—"
fc_scenario         = fc[fc["ds"].isin(future_dates)]
forecast_end_price  = float(fc_scenario["yhat"].iloc[-1]) if len(fc_scenario) else None
forecast_end_label  = f"{QUARTER_SEASONS[future_dates[-1].quarter]} {future_dates[-1].year}" if len(future_dates) else "—"

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        "Цена за м² сейчас",
        f"{last_price/1000:.1f} тыс. руб." if last_price else "—",
        delta=f"данные на {last_period_label}",
    )
with col2:
    if last_price and forecast_end_price:
        growth = (forecast_end_price / last_price - 1) * 100
        st.metric(
            f"Ожидаемая цена к {forecast_end_label}",
            f"{forecast_end_price/1000:.1f} тыс. руб.",
            delta=f"{growth:+.1f}% к текущей",
        )
with col3:
    st.metric("Ключевая ставка ЦБ (сценарий)", f"{cbr_scenario:.1f}%")

# ── Chart ─────────────────────────────────────────────────────────────────────
fig = go.Figure()

# Single historical line (merge both sources)
fig.add_trace(go.Scatter(
    x=hist["period"], y=hist["price_sqm"] / 1000,
    name="История цен",
    mode="lines+markers",
    line=dict(color="#2196F3", width=2.5),
    marker=dict(size=5),
    hovertemplate="%{x|%d.%m.%Y}: <b>%{y:.1f} тыс. руб./м²</b><extra></extra>",
))

# Forecast line
fig.add_trace(go.Scatter(
    x=fc_scenario["ds"], y=fc_scenario["yhat"] / 1000,
    name="Прогноз",
    mode="lines+markers",
    line=dict(color="#E91E63", width=3),
    marker=dict(size=7, symbol="diamond"),
    hovertemplate="%{x|%d.%m.%Y}: <b>%{y:.1f} тыс. руб./м²</b><extra></extra>",
))

# Separator line — start of forecast
fig.add_shape(
    type="line",
    x0=str(future_dates[0].date()), x1=str(future_dates[0].date()),
    y0=0, y1=1, yref="paper",
    line=dict(dash="dot", color="#E91E63", width=1.5),
)
fig.add_annotation(
    x=str(future_dates[0].date()), y=0.97, yref="paper",
    text="Начало прогноза",
    showarrow=False, font=dict(color="#E91E63", size=11),
    xanchor="left", bgcolor="rgba(0,0,0,0.35)",
)

fig.update_layout(
    title=f"Динамика цены за м² — {region_name_selected}",
    xaxis_title="",
    yaxis_title="Тыс. руб. / м²",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hovermode="x unified",
    height=500,
    template="plotly_white",
)
st.plotly_chart(fig, use_container_width=True)

# ── Forecast table ────────────────────────────────────────────────────────────
st.subheader("Прогнозные значения")

fc_table = fc_scenario[["ds", "yhat"]].copy()
fc_table["Период"]  = fc_table["ds"].apply(lambda d: f"{QUARTER_SEASONS[d.quarter]} {d.year}")
fc_table["Прогноз"] = (fc_table["yhat"] / 1000).round(1).astype(str) + " тыс. руб./м²"
st.dataframe(fc_table[["Период", "Прогноз"]], use_container_width=True, hide_index=True)

st.caption(
    "Прогноз носит ориентировочный характер. "
    "Данные: 2018–2021 — объявления о продаже, 2022–2026 — индексы цен Росстата."
)
