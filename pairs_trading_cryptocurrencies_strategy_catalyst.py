import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

from catalyst import run_algorithm
from catalyst.api import (record, symbol, order_target_percent, date_rules, time_rules, get_datetime)

def initialize(context):
    context.A = symbol('xmr_usd')
    context.B = symbol('neo_usd')

    context.leverage = 1.0                              # 1.0 - no leverage
    context.n_modelling = 72                            # number of lookback bars for modelling
    context.tf = str(60) + "T"                          # how many minutes in a timeframe; 1 - to get minute data (often errors happen); 60 - to get hourly data
    context.z_signal_in = st.norm.ppf(1 - 0.0001 / 2)   # z-score threshold to open an order
    context.z_signal_out = st.norm.ppf(1 - 0.60 / 2)    # z-score threshold to close an order
    context.min_spread = 0.035                          # Threshold for minimal allowed spread

    
    context.set_commission(maker = 0.001, taker = 0.002)
    context.set_slippage(slippage = 0.0005)

def handle_data(context, data):
    current_time = get_datetime().time()
    
    # Get data
    A = data.history(context.A,
                    'price',
                    bar_count = context.n_modelling,
                    frequency = context.tf,
                    )

    B = data.history(context.B,
                    'price',
                    bar_count = context.n_modelling,
                    frequency = context.tf,
                    )

    # Calc returns and spread
    A_return = A.pct_change()
    B_return = B.pct_change()
    spread = A_return - B_return

    zscore = (spread.iloc[-1] - spread.mean()) / spread.std()

    # Close positions
    if context.portfolio.positions[context.B].amount < 0 and zscore >= -context.z_signal_out:
        order_target_percent(context.A,  0.0)
        order_target_percent(context.B,  0.0)
    if context.portfolio.positions[context.B].amount > 0 and zscore <= context.z_signal_out:
        order_target_percent(context.A,  0.0)
        order_target_percent(context.B,  0.0)

    # Check minimal allowed spread value
    if (abs(spread[-1]) >= context.min_spread):# and np.sign(A_return[-1] * B_return[-1]) < 0:
        # Long and Short positions for assets
        if context.portfolio.positions[context.B].amount == 0 and zscore > context.z_signal_in:
            order_target_percent(context.A,  -0.5 * context.leverage)
            order_target_percent(context.B,  0.5 * context.leverage)

        if context.portfolio.positions[context.B].amount == 0 and zscore < -context.z_signal_in:
            order_target_percent(context.A,  0.5 * context.leverage)
            order_target_percent(context.B,  -0.5  * context.leverage)


    record(
        A_return = A_return[-1],
        B_return = B_return[-1],
        spread = spread[-1],
        zscore = zscore
    )

def analyze(context, perf):
    # Summary output
    print("Total return: " + str(perf.algorithm_period_return[-1]))
    print("Sortino coef: " + str(perf.sortino[-1]))
    print("Max drawdown: " + str(np.min(perf.max_drawdown)))
    
    f = plt.figure(figsize = (7.2, 7.2))

    # Plot 1st A group
    ax1 = f.add_subplot(611)
    ax1.plot(perf.A_return, 'blue')
    ax1.set_title('A return')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Return')

    # Plot 2nd public group
    ax2 = f.add_subplot(612, sharex = ax1)
    ax2.plot(perf.B_return, 'green')
    ax2.set_title('B return')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Return')

    # Plot spread
    ax3 = f.add_subplot(613, sharex = ax1)
    ax3.plot(perf.spread, 'darkmagenta')
    ax3.axhline(context.min_spread, c = 'red')
    ax3.axhline(-context.min_spread, c = 'red')
    ax3.set_title('Spread')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Value')

    # Plot z-score
    ax4 = f.add_subplot(614, sharex = ax1)
    ax4.plot(perf.zscore, 'grey')
    ax4.axhline(context.z_signal_in, c = 'green')
    ax4.axhline(-context.z_signal_in, c = 'green')
    ax4.axhline(context.z_signal_out, c = 'red')
    ax4.axhline(-context.z_signal_out, c = 'red')
    ax4.set_title('z-score')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Value')

    # Plot return
    ax5 = f.add_subplot(615, sharex = ax1)
    ax5.plot(perf.algorithm_period_return, 'red')
    ax5.set_title('Algorithm return')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Value')

    # Plot leverage
    ax6 = f.add_subplot(616, sharex = ax1)
    ax6.plot(perf.gross_leverage, 'yellow')
    ax6.set_title('Leverage')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Value')

    plt.tight_layout()
    plt.show()

run_algorithm(
    capital_base = 10000,
    data_frequency = 'minute',
    initialize = initialize,
    handle_data = handle_data,
    analyze = analyze,
    exchange_name = 'bitfinex',
    quote_currency = 'usd',
    start = pd.to_datetime('2018-10-1', utc = True),
    end = pd.to_datetime('2018-11-30', utc = True))

