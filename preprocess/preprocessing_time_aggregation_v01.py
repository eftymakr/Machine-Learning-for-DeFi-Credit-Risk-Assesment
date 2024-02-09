import sys
import requests
import pandas as pd
import glob
import os
import numpy as np
import pprint as pp
import datetime
import time
from itertools import chain
from numpy import array
import random as rnd
from datetime import datetime, timedelta

def preprocessing(split_counter):
    # Load CSV files and concatenate them
    df = pd.DataFrame()
    for split_counter_it in range(split_counter+1):
        df_read = pd.read_csv('../temp_files/full_training_set_borrowed_10txns_part' + str(split_counter_it) + '_v02.csv')
        df = pd.concat([df, df_read], axis=0)

    # List of event types
    event_list = ['borrow', 'deposit', 'flashloan', 'liquidation', 'repay', 'withdraw']

    # Group by address, YearMonth, and event; reshape the data
    dfg = df.groupby(['address', 'YearMonth', 'event'], as_index=False).count()
    dfg = dfg.set_index(['address', 'YearMonth', 'event'])['amount_in_usd'].unstack(['event'])
    dfg.reset_index(inplace=True)

    # Rename columns to include event types
    for i in event_list:
        dfg = dfg.rename(columns={i: i + "_total_count"})

    # Compute various statistics for amount features
    for feature in ['amount_in_usd', 'amount_out_usd']:
        for ops in ['sum', 'mean', 'std']:
            temp = getattr(df.groupby(['address', 'YearMonth', 'event'], as_index=False), ops)(numeric_only=False)
            temp = temp.set_index(['address', 'YearMonth', 'event'])[feature].unstack(['event'])
            temp.reset_index(inplace=True)

            for i in event_list:
                temp = temp.rename(columns={i: i + "_" + feature + "_" + ops})

            dfg = pd.merge(dfg, temp)

    # Fill missing values and sort
    dfg = dfg.fillna(0)
    dfg = dfg.sort_values(['address', 'YearMonth'], ascending=[True, True])

    # Compute statistics based on YearMonth
    start_ym, end_ym = 41, 46
    dfg['YearMonth'] = dfg.apply(lambda x: 
                               999 if (x['YearMonth'] >= start_ym and x['YearMonth'] <= end_ym) else 
                               (-999 if x['YearMonth'] > end_ym else x['YearMonth']), axis=1)
    dfg = dfg[dfg.YearMonth != -999]
    dfg = dfg.sort_values(['address', 'YearMonth'], ascending=[True, True])
    address_list = dfg.groupby('address', as_index=False).mean()['address'].to_list()

    # Create empty DataFrame with all combinations of addresses and YearMonths
    nb_steps = 41
    yearmonth_list = list(range(0, nb_steps))
    yearmonth_list.append(999)

    # yearmonth_length = list(chain(*[[ym] * len(address_list) for ym in yearmonth_list]))

    yearmonth_length = []
    for i in range(len(address_list)):
        yearmonth_length.append(yearmonth_list)
    yearmonth_length = list(chain(*yearmonth_length))
    yearmonth_length

    address_length = list(chain(*[[address] * (nb_steps + 1) for address in address_list]))
    empty_df = pd.DataFrame({'address': address_length, 'YearMonth': yearmonth_length})

    # Merge empty DataFrame with original DataFrame
    dfn = pd.merge(empty_df, dfg, how='left', on=['address', 'YearMonth'])
    dfn = dfn.groupby(['address', 'YearMonth'], as_index=False).sum()
    dfn.fillna(0, inplace=True)
    dfn.to_csv('../ML_input_v03.csv', index=False)
    print('done!')
    # dfn.groupby('YearMonth').count()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py split_counter")
        sys.exit(1)
    
    split_counter = int(sys.argv[1])
    preprocessing(split_counter)
    print('Done!')
