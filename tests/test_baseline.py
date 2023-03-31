import pandas as pd
import numpy as np
from mavts import baseline

def test_last():
    series = pd.read_csv('./tests/data/data_siling.csv',
                       index_col=0, parse_dates=True).iloc[:, 0]

    for days_ahead in range(1, 11):
        predictions = baseline.last(series, days_ahead=days_ahead)
        exp_fn = f'./tests/data/last_{days_ahead}.csv'
        expected = pd.read_csv(exp_fn, index_col=0, parse_dates=True).iloc[:,0]
        diff = predictions - expected
        np.testing.assert_((abs(diff) < 1e-7).mean() == 1.0)

def test_ewm():
    series = pd.read_csv('./tests/data/data_siling.csv',
                       index_col=0, parse_dates=True).iloc[:, 0]

    for days_ahead in range(1, 11):
        predictions = baseline.ewm(series, days_ahead=days_ahead)
        exp_fn = f'./tests/data/ewm_{days_ahead}.csv'
        expected = pd.read_csv(exp_fn, index_col=0, parse_dates=True).iloc[:,0]
        diff = predictions - expected
        np.testing.assert_((abs(diff) < 1e-7).mean() == 1.0)

