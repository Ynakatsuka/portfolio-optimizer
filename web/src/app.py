from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import streamlit as st
import yfinance
from scipy.optimize import minimize

INPUT_DIR = Path("../crawler/mufg_funds")
BTC_FUND_CD = "XXXXXX"
DATE_COLUMN = "BASE_DATE"
PRICE_COLUMN = "REINVEST_BASE_PRICE"
FUND_COLUMNS = ["FUND_CD", "FUND_NAME", "FUND_SHORT_NAME", DATE_COLUMN, PRICE_COLUMN]


def load_btc_data() -> pl.DataFrame:
    btc = yfinance.Ticker("BTC-JPY").history(period="max").reset_index()
    btc["FUND_CD"] = BTC_FUND_CD
    btc["FUND_NAME"] = "Bitcoin"
    btc["FUND_SHORT_NAME"] = "BTC"
    btc = btc.rename(columns={"Close": PRICE_COLUMN, "Date": DATE_COLUMN})
    btc = pl.from_pandas(btc[FUND_COLUMNS])
    return btc.with_columns(pl.col(DATE_COLUMN).cast(pl.Date).alias(DATE_COLUMN))


def preprocess(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.explode("ROWS")
        .unnest("ROWS")
        .with_columns(
            pl.col(DATE_COLUMN).str.to_datetime("%Y%m%d").cast(pl.Date).alias(DATE_COLUMN),
        )
    )


@st.cache_data
def get_fund_statistics(
    _df: pl.DataFrame,
    frequency: int = 252,
    max_null_ratio: float = 0.8,
    reference_date: str = "2021-01-01",
) -> tuple[np.ndarray, np.ndarray]:
    reference_date = pl.lit(reference_date).cast(pl.Date)
    filtered_df = _df.filter(pl.col(DATE_COLUMN) >= reference_date).sort("FUND_CD", DATE_COLUMN)
    filtered_df = filtered_df.with_columns(
        daily_return=pl.col(PRICE_COLUMN).over("FUND_CD").pct_change(),
    ).to_pandas()

    returns_pivot = filtered_df.pivot_table(index=DATE_COLUMN, columns="FUND_CD", values="daily_return").dropna(
        how="all",
    )

    # remove funds with too many missing values
    null_ratio = returns_pivot.isna().mean()
    returns_pivot = returns_pivot.loc[:, null_ratio[null_ratio <= max_null_ratio].index]

    annual_returns = (returns_pivot + 1).prod() ** (frequency / len(returns_pivot)) - 1
    assert annual_returns.isna().sum() == 0, annual_returns.isna().sum()

    cov_matrix = returns_pivot.cov() * frequency
    assert cov_matrix.isna().sum().sum() == 0, cov_matrix.isna().sum().sum()

    # fix nonpositive semidefinite
    q, v = np.linalg.eigh(cov_matrix)
    q = np.where(q > 0, q, 0)
    cov_matrix = v @ np.diag(q) @ v.T

    return annual_returns, cov_matrix


@st.cache_data
def optimize_portfolio(
    annual_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_ratio: float = 0.01,
) -> tuple[np.ndarray, float]:
    def neg_sharpe(weights: np.ndarray) -> float:
        port_return = weights.T @ annual_returns
        port_risk = np.sqrt(weights.T @ cov_matrix @ weights)
        return -(port_return - risk_free_ratio) / port_risk

    constraints = {"type": "eq", "fun": lambda x: x.sum() - 1}
    initial_weights = np.repeat(1 / annual_returns.size, annual_returns.size)
    bounds = [(0, 1) for _ in range(annual_returns.size)]

    result = minimize(neg_sharpe, initial_weights, constraints=constraints, bounds=bounds, options={"disp": True})

    if not result.success:
        msg = "Optimization failed: " + result.message
        raise ValueError(msg)

    sharpe_ratio = -result.fun
    return result.x, sharpe_ratio


def main() -> None:
    st.title("Portfolio Optimizer", anchor="title")

    # bitcoin data
    btc = load_btc_data()

    # fund data
    df = pl.read_json(INPUT_DIR / "fundchart.json")
    fund_labels = df.select("FUND_CD", "FUND_NAME", "FUND_SHORT_NAME").to_pandas()
    df = preprocess(df).select(FUND_COLUMNS)

    # concatenate bitcoin data
    df = pl.concat([df, btc])
    fund_labels = pd.concat(
        [fund_labels, pd.DataFrame({"FUND_CD": [BTC_FUND_CD], "FUND_NAME": ["Bitcoin"], "FUND_SHORT_NAME": ["BTC"]})],
    )

    st.write("### Annual Returns")
    reference_date = st.date_input("Reference Date", value=pd.to_datetime("2018-01-01"))
    annual_returns, cov_matrix = get_fund_statistics(df, reference_date=reference_date)
    annual_returns_to_display = (
        annual_returns.to_frame()
        .reset_index()
        .rename(columns={"index": "FUND_CD", 0: "ANNUAL_RETURN"})
        .merge(fund_labels, on="FUND_CD")
        .sort_values("ANNUAL_RETURN", ascending=False)
    )
    st.write(annual_returns_to_display)

    risk_free_ratio = st.slider("Risk Free Ratio", min_value=0.0, max_value=0.2, value=0.05, step=0.01)

    st.write("### Optimal Portfolio")
    with st.spinner("Wait for optimization..."):
        weights, sharpe_ratio = optimize_portfolio(annual_returns, cov_matrix, risk_free_ratio=risk_free_ratio)
        weights_to_display = pd.DataFrame({"FUND_CD": annual_returns.index, "WEIGHT": weights}).merge(
            fund_labels,
            on="FUND_CD",
        )
        weights_to_display = weights_to_display.query("WEIGHT > 0.001").sort_values("WEIGHT", ascending=False)
    st.write(f"Sharpe Ratio: {sharpe_ratio}")
    st.write(weights_to_display)

    fig = px.pie(weights_to_display, values="WEIGHT", names="FUND_SHORT_NAME", title="Portfolio Allocation")
    st.plotly_chart(fig)


if __name__ == "__main__":
    main()
