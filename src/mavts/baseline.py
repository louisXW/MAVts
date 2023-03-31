import pandas as pd


def last(series: pd.Series, days_ahead: int = 1) -> pd.Series:
    """ Return pd.Series of `days_ahead` predictions. The predictions are the 
    last known observation. 
    Example input `series` is: [3, 5, 7] the predictions will be [5, 7]"""

    predictions = series.shift(days_ahead).dropna()

    return predictions


def ewm(series: pd.Series, days_ahead: int = 1, alpha=0.95) -> pd.Series:
    """ Return pd.Series of `days_ahead` predictions. The predictions are the 
    last known observation plus an Exponential Weighted Moving Average with 
    `alpha` as parameter`. See pd.DataFrame.ewm for more details.
    """
    last_obs = last(series=series, days_ahead=days_ahead)
    change_ewm = last_obs.pct_change().ewm(alpha=alpha).mean()
    predictions = (last_obs + change_ewm).dropna()

    return predictions
