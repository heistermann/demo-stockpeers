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
#import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt

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

DEFAULT_STOCKS = ["OEH","LIN"]


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
        "Stock tickers",
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

def load_data2():
    df = pd.read_csv("https://b2drop.eudat.eu/public.php/dav/files/efStHSPAM8HLc92/products/swc-from-crns.txt",
                     sep="\t", na_values="na")
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    df.index.name = 'Date'
    df = df.loc[df.index[-1] - pd.DateOffset(days=horizon_map[horizon]):df.index[-1]]
    return df

#data = load_data(dtimes, STOCKS, rho=0.7, seed=42)
data = load_data2()[tickers]

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

# Normalize prices (start at 1)
normalized = data.div(data.iloc[0])

latest_norm_values = {normalized[ticker].iat[-1]: ticker for ticker in tickers}
#max_norm_value = max(latest_norm_values.items())
#min_norm_value = min(latest_norm_values.items())
mean_theta = data.loc["2024-09-01":"2025-09-01"].mean()

bottom_left_cell = cols[0].container(
    border=True, height="stretch", vertical_alignment="center"
)

with bottom_left_cell:
    cols = st.columns(2)
    cols[0].metric(
        "Trockenster Standort: "+mean_theta.idxmin(),
        round(mean_theta.min(), 2),
        delta=f"{round(mean_theta.min() * 100)}%",
        width="content",
    )
    cols[1].metric(
        "Feuchtester Standort: "+mean_theta.idxmax(),
        round(mean_theta.max(), 2),
        delta=f"{round(mean_theta.max() * 100)}%",
        width="content",
    )


# Plot normalized prices
#with right_cell:
#    st.altair_chart(
#        alt.Chart(
#            normalized.reset_index().melt(
#                id_vars=["Date"], var_name="Stock", value_name="Normalized price"
#            )
#        )
#        .mark_line()
#        .encode(
#            alt.X("Date:T"),
#            alt.Y("Normalized price:Q").scale(zero=False),
#            alt.Color("Stock:N"),
#        )
#        .properties(height=400)
#    )

with right_cell:
    st.altair_chart(
        alt.Chart(
                data.reset_index().rename(columns={"index": "Date"})  .melt("Date", var_name="series", value_name="value")
            )
            .mark_line()
            .encode(
                x="Date:T",
                y=alt.Y("value:Q", title="Soil water content [m³/m³]", scale=alt.Scale(domain=[0, 0.4])),
                color="series:N",
                tooltip=["Date:T", "series:N", "value:Q"]
        )
        .properties(height=400)
    )

""
""

# Plot individual stock vs peer average
"""
## Individual stocks vs peer average

For the analysis below, the "peer average" when analyzing stock X always
excludes X itself.
"""

if len(tickers) <= 1:
    st.warning("Pick 2 or more tickers to compare them")
    st.stop()

NUM_COLS = 4
cols = st.columns(NUM_COLS)

for i, ticker in enumerate(tickers):
    # Calculate peer average (excluding current stock)
    peers = normalized.drop(columns=[ticker])
    peer_avg = peers.mean(axis=1)

    # Create DataFrame with peer average.
    plot_data = pd.DataFrame(
        {
            "Date": normalized.index,
            ticker: normalized[ticker],
            "Peer average": peer_avg,
        }
    ).melt(id_vars=["Date"], var_name="Series", value_name="Price")

    chart = (
        alt.Chart(plot_data)
        .mark_line()
        .encode(
            alt.X("Date:T"),
            alt.Y("Price:Q").scale(zero=False),
            alt.Color(
                "Series:N",
                scale=alt.Scale(domain=[ticker, "Peer average"], range=["red", "gray"]),
                legend=alt.Legend(orient="bottom"),
            ),
            alt.Tooltip(["Date", "Series", "Price"]),
        )
        .properties(title=f"{ticker} vs peer average", height=300)
    )

    cell = cols[(i * 2) % NUM_COLS].container(border=True)
    cell.write("")
    cell.altair_chart(chart, use_container_width=True)

    # Create Delta chart
    plot_data = pd.DataFrame(
        {
            "Date": normalized.index,
            "Delta": normalized[ticker] - peer_avg,
        }
    )

    chart = (
        alt.Chart(plot_data)
        .mark_area()
        .encode(
            alt.X("Date:T"),
            alt.Y("Delta:Q").scale(zero=False),
        )
        .properties(title=f"{ticker} minus peer average", height=300)
    )

    cell = cols[(i * 2 + 1) % NUM_COLS].container(border=True)
    cell.write("")
    cell.altair_chart(chart, use_container_width=True)

""
""

"""
## Raw data
"""

data
