from __future__ import division
import pandas as pd
import numpy as np
from . import pos
from . import timeseries

DISPLAY_UNIT = 1000000


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


def days_to_liquidate_positions(positions, market_data, 
                                max_bar_consumption=0.2,
                                capital_base=1e6,
                                mean_volume_window=5):
    """Compute the number of days that would have been required
    to fully liquidate each position on each day based on the
    trailing n day mean daily bar volume and a limit on the proportion
    of a daily bar that we are allowed to consume. 

    This analysis uses portfolio allocations and a provided capital base
    rather than the dollar values in the positions DataFrame to remove the
    effect of compounding on days to liquidate. In other words, this function
    assumes that the net liquidation portfolio value will always remain
    constant at capital_base.

    Parameters
    ----------
    positions: pd.DataFrame
        Contains daily position values including cash
        - See full explanation in tears.create_full_tear_sheet
    market_data : pd.Panel
        Contains OHLCV DataFrames for the tickers in the passed
        positions DataFrame
    max_bar_consumption : float
        Max proportion of a daily bar that can be consumed in the
        process of liquidating a position.
    capital_base : integer
        Capital base multiplied by portfolio allocation to compute
        position value that needs liquidating.
    mean_volume_window : float
        Trailing window to use in mean volume calculation.

    Returns
    -------
    days_to_liquidate : pd.DataFrame
        Number of days required to fully liquidate daily positions.
        Datetime index, symbols as columns.

    """
 
    DV = market_data['volume'] * market_data['price']
    roll_mean_dv = pd.rolling_mean(DV, mean_volume_window)

    positions_alloc = pos.get_percent_alloc(positions)
    positions_alloc = positions_alloc.drop('cash', axis=1)

    days_to_liquidate = (positions_alloc * capital_base) / \
        (max_bar_consumption * roll_mean_dv)

    return days_to_liquidate


def get_max_days_to_liquidate_by_ticker(positions, days_to_liquidate):

    positions_alloc = pos.get_percent_alloc(positions)
    positions_alloc = positions_alloc.drop('cash', axis=1)
    longest_liq_each_ticker_ix = days_to_liquidate.idxmax(axis=0).dropna()

    worst_liq = pd.DataFrame()
    for ticker, date in longest_liq_each_ticker_ix.iteritems():
        worst_liq.loc[ticker, 'date'] = date
        worst_liq.loc[ticker, 'portfolio_allocation_%'] = positions_alloc.loc[
            date, ticker] * 100
        worst_liq.loc[ticker, 'days_to_liquidate'] = days_to_liquidate.loc[
            date, ticker]
    worst_liq.index.name = 'symbol'

    return worst_liq.reset_index()


def get_low_liquidity_transactions(txn_daily, market_data):
    def max_consumption_txn_row(x):
        max_pbc = x['max_pct_bar_consumed'].max()
        return x[x['max_pct_bar_consumed'] == max_pbc].iloc[:1]
    
    max_bar_consumption = txn_daily.assign(
        max_pct_bar_consumed=txn_daily.amount/txn_daily.volume
        ).groupby('symbol').apply(max_consumption_txn_row)

    return max_bar_consumption[['max_pct_bar_consumed']].reset_index()


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
