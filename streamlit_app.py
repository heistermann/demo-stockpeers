# -*- coding: utf-8 -*-
# Copyright 2024-2025 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Soil water viewer",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

"""
# :material/query_stats: Bodenwassermonitor Brandenburg

Die dargestellten Daten beruhen auf Neutronenmessungen und Modellsimulationen (weitere Details folgen...).
"""

""  # Add some space.

cols = st.columns([1, 3])
# Will declare right cell later to avoid showing it when no data.

STOCKS = [
    "OEH",
    "LIN",
    "MQ",
    "PAU",
    "BOO",
    "DED",
    "KH",
    "GOL",
]

DEFAULT_STOCKS = ["OEH","MQ","DED"]


def stocks_to_str(stocks):
    return ",".join(stocks)


if "tickers_input" not in st.session_state:
    st.session_state.tickers_input = st.query_params.get(
        "stocks", stocks_to_str(DEFAULT_STOCKS)
    ).split(",")


# Callback to update query param when input changes
def update_query_param():
    if st.session_state.tickers_input:
        st.query_params["stocks"] = stocks_to_str(st.session_state.tickers_input)
    else:
        st.query_params.pop("stocks", None)


top_left_cell = cols[0].container(
    border=True, height="stretch", vertical_alignment="center"
)

with top_left_cell:
    # Selectbox for stock tickers
    tickers = st.multiselect(
        "Standorte",
        options=sorted(set(STOCKS) | set(st.session_state.tickers_input)),
        default=st.session_state.tickers_input,
        placeholder="Choose stocks to compare. Example: NVDA",
        accept_new_options=True,
    )

# Time horizon selector
horizon_map = {
    "1 months": 31,
    "3 months": 3*31,
    "6 months": 6*31,
    "1 year": 365,
    "2 years": 2*365,
    "3 years": 3*365,
}

with top_left_cell:
    # Buttons for picking time horizon
    horizon = st.pills(
        "Time horizon",
        options=list(horizon_map.keys()),
        default="6 months",
    )

tickers = [t.upper() for t in tickers]

# Update query param when text input changes
if tickers:
    st.query_params["stocks"] = stocks_to_str(tickers)
else:
    # Clear the param if input is empty
    st.query_params.pop("stocks", None)

if not tickers:
    top_left_cell.info("Pick some stocks to compare", icon=":material/info:")
    st.stop()


right_cell = cols[1].container(
    border=True, height="stretch", vertical_alignment="center"
)

def load_data(url):
    df = pd.read_csv(url,
                     sep="\t", na_values="na")
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    df.index.name = 'Date'
    df = df.loc[df.index[-1] - pd.DateOffset(days=horizon_map[horizon]):df.index[-1]]
    df = df.rename(columns={"QUI":"DED", "MQ35":"MQ"})
    if "WUS" in df.columns:
        df = df.drop(columns=["WUS"])
    return df

#data = load_data(dtimes, STOCKS, rho=0.7, seed=42)
data2 = load_data("https://b2drop.eudat.eu/public.php/dav/files/efStHSPAM8HLc92/products/swc-from-crns.txt")
data = data2[tickers]
sim2 = load_data("https://b2drop.eudat.eu/public.php/dav/files/efStHSPAM8HLc92/products/swc-from-swap.txt")
sim = sim2[tickers]


#@st.cache_resource(show_spinner=False, ttl="6h")
#def load_data(tickers, period):
#    tickers_obj = yf.Tickers(tickers)
#    data = tickers_obj.history(period=period)
#    if data is None:
#        raise RuntimeError("YFinance returned no data.")
#    return data["Close"]

# Load the data
#try:
#    data = load_data(tickers, horizon_map[horizon])
#except yf.exceptions.YFRateLimitError as e:
#    st.warning("YFinance is rate-limiting us :(\nTry again later.")
#    load_data.clear()  # Remove the bad cache entry.
#    st.stop()

empty_columns = data.columns[data.isna().all()].tolist()

if empty_columns:
    st.error(f"Error loading data for the tickers: {', '.join(empty_columns)}.")
    st.stop()

mean_theta = data.mean()

bottom_left_cell = cols[0].container(
    border=True, height="stretch", vertical_alignment="center"
)

with bottom_left_cell:
    cols = st.columns(2)
    cols[0].metric(
        "Mittelwert: "+mean_theta.idxmin(),
        round(mean_theta.min(), 2),
        delta=f"{round(mean_theta.min() * 100)}%",
        width="content",
    )
    cols[1].metric(
        "Mittelwert: "+mean_theta.idxmax(),
        round(mean_theta.max(), 2),
        delta=f"{round(mean_theta.max() * 100)}%",
        width="content",
    )

with right_cell:
    fig = px.line(data, x=data.index, y=tickers)
    st.plotly_chart(fig, use_container_width=True)

""
""

# Plot individual stock vs peer average
"""
## Darstellung der einzelnen Monitoringstandorte

Die folgenden Abbildungen zeigen für jeden einzelnen Standort unterschiedliche
Variablen: SWC(CRNS) ist die aus CRNS-Beobachtungen abgeleitete Bodenfeuchte,
SWC(SWAP) ist die mit Hilfe des Bodenwasserhaushaltsmodells SWAP simulierte
Bodenfeuchte.
"""

NUM_COLS = 3
cols = st.columns(NUM_COLS)

for i, ticker in enumerate(data2.columns):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=data2.index, y=data2[ticker], mode="lines", name="SWC(CRNS)")
    )
    fig.add_trace(
        go.Scatter(x=sim2.index, y=sim2[ticker], mode="lines", name="SWC(SWAP)")
    )

    #fig = px.line(data2, x=data.index, y=ticker, title=ticker)
    fig.update_layout(title=ticker)
    
    cell = cols[(i * 1) % NUM_COLS].container(border=True)
    cell.write("")
    cell.plotly_chart(fig, use_container_width=True, key=i)


""
""

"""
## Datendownload
"""

data
