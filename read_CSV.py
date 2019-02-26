# -*- coding: utf-8 -*-
"""
Import As Pandas

@author: Frank
"""

import os
import pandas as pd

def symbol_to_path(symbol, base_dir=os.path.join("R-Script", "Stocks")):
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbols, dates, colname = 'Adj.Close'):
    """Read stock data for given symbols from CSV files."""
    
    df = pd.DataFrame(index=dates)

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', colname], na_values=['nan'])
        df_temp = df_temp.rename(columns={colname: symbol})
        df = df.join(df_temp)

    return df