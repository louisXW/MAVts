import pandas as pd
import numpy as np
from pathlib import Path
from mavts import mark, vis, baseline


def test_errors_by_period_series():

    obs = pd.read_csv('./tests/data/data_siling.csv', index_col=0,
                      parse_dates=True)
    obs = obs.iloc[:, 0]

    predictions = {}
    predictions["ewm_85-99"] = []
    for alpha in np.linspace(0.85, 0.99, num=15):
        tmp = baseline.ewm(obs, alpha=alpha)
        predictions["ewm_85-99"].append(tmp)

    predictions["ewm_55-79"] = []
    for alpha in np.linspace(0.55, 0.79, num=15):
        tmp = baseline.ewm(obs, alpha=alpha)
        predictions["ewm_55-79"].append(tmp)

    predictions["last"] = [baseline.last(obs)]

    vis.errors_by_period(obs, predictions, './tests/plots/error_by_period/')


def test_models_as_filmroll():
    if Path('./tests/data/data_org_trop.csv').exists():
        data = pd.read_csv('./tests/data/data_org_trop.csv',
                           index_col=0, parse_dates=True).iloc[:, 0]

        model_A = pd.read_csv('./tests/data/gru_trop.csv',
                              index_col=0, parse_dates=True).iloc[:, 0]

        model_B = pd.read_csv('./tests/data/arima_trop.csv',
                              index_col=0, parse_dates=True).iloc[:, 0]

        data = data.loc[model_A.index]

        ups, dns = mark.up_dn_periods(data.values, 20, 5)
        peaks, bottoms = mark.peaks_bottoms(ups, dns)
        interpolated = mark.interpolated(data)

        periods = dict(ups=ups,
                       dns=dns,
                       peaks=peaks,
                       bottoms=bottoms,
                       interpolated=interpolated)

        vis.models_as_filmroll(data,
                               model_A,
                               model_B,
                               './tests/plots/',
                               periods,
                               scale_y=True)


def test_observations_as_filmroll_trop():
    if Path('./tests/data/data_mod_trop.csv').exists():
        data = pd.read_csv('./tests/data/data_mod_trop.csv',
                           index_col=0, parse_dates=True).iloc[:, 0]

        ups, dns = mark.up_dn_periods(data.values)
        peaks, bottoms = mark.peaks_bottoms(ups, dns)
        interpolated = mark.interpolated(data)

        periods = dict(ups=ups,
                       dns=dns,
                       peaks=peaks,
                       bottoms=bottoms,
                       interpolated=interpolated)

        vis.observations_as_filmroll(
            data, './tests/plots/', periods, scale_y=True)


def test_observations_as_filmroll_siling():
    data = pd.read_csv('./tests/data/data_siling.csv',
                       index_col=0, parse_dates=True).iloc[:, 0]

    ups, dns = mark.up_dn_periods(data.values, 0.50, 0.13)
    peaks, bottoms = mark.peaks_bottoms(ups, dns)
    interpolated = mark.interpolated(data)

    periods = dict(ups=ups,
                   dns=dns,
                   peaks=peaks,
                   bottoms=bottoms,
                   interpolated=interpolated)
    vis.observations_as_filmroll(data, './tests/plots_siling/', periods, True)
