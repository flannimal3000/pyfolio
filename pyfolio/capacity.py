from __future__ import division
import pandas as pd
import numpy as np
from . import pos
from . import timeseries

DISPLAY_UNIT = 1000000


def get_assets_dollar_volume(positions, market_data,
                             only_held_days=True, last_n_days=None):
    """
    Computes descriptive statsitics for the daily dollar
    volume for traded names over the backtest dates.

    Parameters
    ----------
    positions: pd.DataFrame
        Contains daily position values including cash
        - See full explanation in tears.create_full_tear_sheet
    market_data : pd.Panel
        Contains OHLCV DataFrames for the tickers in the passed
        positions DataFrame
    only_held_days : boolean
        Only compute daily dollar volume statistics on days when
        a name was held in the portfolio.
    last_n_days : integer, optional
        Compute max position allocation and dollar volume for only
        the last N days of the backtest

    Returns
    -------
    vol_analysis : pd.DataFrame
        Index of traded tickers. Columns for 10th percentile, 90th percentile,
        and mean daily traded volume, as well as max position concentration.
    """
    DV = market_data['volume'] * market_data['price']

    positions_alloc = pos.get_percent_alloc(positions)
    positions_alloc = positions_alloc.drop('cash', axis=1)

    if last_n_days is not None:
        positions_alloc = positions_alloc.iloc[-last_n_days:]
        DV = DV.iloc[-last_n_days:]

    if only_held_days:
        for name, ts in positions_alloc.iteritems():
            DV[name] = DV.loc[ts[ts != 0].index, name]

    max_exposure_per_ticker = abs(positions_alloc).max()

    vol_analysis = pd.DataFrame()
    vol_analysis['algo_max_exposure'] = max_exposure_per_ticker
    vol_analysis.loc[:, 'avg_daily_dv_$mm'] = np.round(
        DV.mean() / DISPLAY_UNIT, 2)
    vol_analysis['10th_%_daily_dv_$mm'] = np.round(DV.apply(
        lambda x: np.nanpercentile(x, 10)) / DISPLAY_UNIT, 2)
    vol_analysis['90th_%_daily_dv_$mm'] = np.round(DV.apply(
        lambda x: np.nanpercentile(x, 90)) / DISPLAY_UNIT, 2)

    return vol_analysis


def get_portfolio_size_constraints(vol_analysis, daily_vol_limit=0.2):
    """
    Finds max portfolio size at daily volume limit given at different
    slices of daily dollar volume distributions.

    Parameters
    ----------
    vol_analysis : pd.DataFrame
        See output of get_assets_dollar_volume
    daily_vol_limit : float
        Max proportion of any daily bar that the startegy is allowed to
        consume.

    Returns
    -------
    constraint_tickers : pd.DataFrame
        Algo capacity and constraining ticker at various parts of
        daily volume distribution.
    """
    constraints = pd.DataFrame()
    constraints['max_capacity_at_ADTV'] = np.round(
        (vol_analysis['avg_daily_dv_$mm'] * daily_vol_limit) /
        vol_analysis.algo_max_exposure, 2)
    constraints['max_capacity_at_10th%'] = np.round(
        (vol_analysis['10th_%_daily_dv_$mm'] * daily_vol_limit) /
        vol_analysis.algo_max_exposure, 2)
    constraints['max_capacity_at_90th%'] = np.round(
        (vol_analysis['90th_%_daily_dv_$mm'] * daily_vol_limit) /
        vol_analysis.algo_max_exposure, 2)

    constraints.sort('max_capacity_at_ADTV')

    return constraints


def get_constraining_tickers(constraints):
    """
    Finds most constraining tickers at different dollar volume
    descriptive statsitics

    Parameters
    ----------
    constraints : pd.DataFrame
        See output of get_portfolio_size_constraints

    Returns
    -------
    constraint_tickers : pd.DataFrame
        Algo capacity and constraining ticker at various parts of
        daily volume distribution.
    """

    constraint_tickers = pd.DataFrame()
    for name, d in constraints.dropna(axis=0).iteritems():
        constraint_tickers.loc[name, 'Algo Capacity $ Millions'] = d.min()
        constraint_tickers.loc[name, 'Constraining Ticker'] = d.argmin()

    return constraint_tickers


def daily_txns_with_bar_data(transactions, market_data):
    """
    Sums the absolute value of shares traded in each name on each day.
    Adds columns containing the closing price and total daily volume for
    each day-ticker combination.

    Parameters
    ----------
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
        - See full explanation in tears.create_full_tear_sheet
    market_data : pd.DataFrame

    Returns
    -------
    txn_daily : pd.DataFrame
        Daily totals for transacted shares in each traded name.
        price and volume columns for close price and daily volume for
        the corresponding ticker, respectively.
    """
    transactions.index.name = 'date'
    txn_daily = pd.DataFrame(transactions.assign(
        amount=abs(transactions.amount)).groupby(
        ['symbol', pd.TimeGrouper('D')]).sum()['amount'])
    txn_daily['price'] = market_data['price'].unstack()
    txn_daily['volume'] = market_data['volume'].unstack()

    txn_daily = txn_daily.reset_index().set_index('date')

    return txn_daily


def apply_slippage_penalty(returns, txn_daily, simulate_starting_capital,
                           backtest_starting_capital, impact=0.1):
    """
    Applies quadratic volumeshare slippage model to daily return based
    on the proportion of the observed historical daily bar dollar volume
    consumed by the strategy's trades. Scales the size of trades based
    on the ratio of the starting capital we wish to test to the starting
    capital of the passed backtest data.

    Parameters
    ----------
    returns : pd.Series
        Time series of daily returns.
    txn_daily : pd.Series
        Daily transaciton totals, closing price, and daily volume for
        each traded name. See price_volume_daily_txns for more details.
    simulate_starting_capital : integer
        capital at which we want to test
    backtest_starting_capital: capital base at which backtest was
        origionally run. impact: See Zipline volumeshare slippage model

    Returns
    -------
    adj_returns : pd.Series
        Slippage penalty adjusted daily returns.
    """
    mult = simulate_starting_capital / backtest_starting_capital
    simulate_traded_shares = abs(mult * txn_daily.amount)
    simulate_traded_dollars = txn_daily.price * simulate_traded_shares

    penalties = (simulate_traded_shares / txn_daily.volume)**2 \
        * impact * simulate_traded_dollars

    daily_penalty = penalties.resample('D', how='sum')
    daily_penalty = daily_penalty.reindex(returns.index).fillna(0)

    # Since we are scaling the numerator of the penalties linearly
    # by capital base, it makes the most sense to scale the denominator
    # similarly. In other words, since we aren't applying compounding to
    # simulate_traded_shares, we shouldn't apply compounding to pv.
    pv = timeseries.cum_returns(
        returns, starting_value=backtest_starting_capital) * mult

    adj_returns = returns - (daily_penalty / pv)

    return adj_returns
