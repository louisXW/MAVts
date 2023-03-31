from pathlib import Path
from mavts import mark
import numpy as np
import pandas as pd


def test_all_periods():
    if Path('./tests/data/data_mod_trop.csv').exists():
        data = pd.read_csv('./tests/data/data_mod_trop.csv', index_col=0,
                           parse_dates=True).iloc[:, 0]
        all_data = mark.all_periods(data)
        expected = pd.read_csv('./tests/data/all_data.csv',
                               index_col=0,
                               parse_dates=True)

        diff = all_data - expected
        np.testing.assert_((abs(diff) < 1e-7).mean().mean() == 1.0)


def test_up_dn_periods_siling():
    expect_ups = np.load('./tests/data/ups_siling.npy').tolist()
    expect_dns = np.load('./tests/data/dns_siling.npy').tolist()

    data = pd.read_csv('./tests/data/data_siling.csv', index_col=0,
                       parse_dates=True).iloc[:, 0]
    ups, dns = mark.up_dn_periods(data.values, 0.50, 0.13)

    np.testing.assert_allclose(ups, expect_ups)
    np.testing.assert_allclose(dns, expect_dns)


def test_up_dn_periods_org_trop():
    if Path('./tests/data/data_org_trop.csv').exists():
        expect_ups = np.load('./tests/data/ups_trop.npy').tolist()
        expect_dns = np.load('./tests/data/dns_trop.npy').tolist()

        data = pd.read_csv('./tests/data/data_org_trop.csv', index_col=0,
                           parse_dates=True).iloc[:, 0].values
        ups, dns = mark.up_dn_periods(data, 20, 5)

        np.testing.assert_allclose(ups, expect_ups)
        np.testing.assert_allclose(dns, expect_dns)


def test_up_dn_periods_mod_trop():
    if Path('./tests/data/data_mod_trop.csv').exists():
        data = pd.read_csv('./tests/data/data_mod_trop.csv', index_col=0,
                           parse_dates=True).iloc[:, 0].values
        ups, dns = mark.up_dn_periods(data)

        expect_ups = np.load('./tests/data/ups_trop_mod.npy').tolist()
        expect_dns = np.load('./tests/data/dns_trop_mod.npy').tolist()

        np.testing.assert_allclose(ups, expect_ups)
        np.testing.assert_allclose(dns, expect_dns)


def test_resolve_overlaps():
    ups = np.load('./tests/data/ups.npy').tolist()
    dns = np.load('./tests/data/dns.npy').tolist()
    values = np.random.normal(size=(5000))
    overlaps = np.load('./tests/data/overlaps.npy').tolist()

    expected_overlaps = mark._resolve_overlaps(ups, dns, values)[-1]

    np.testing.assert_allclose(expected_overlaps, overlaps)


def test_peaks_and_bottoms_siling():
    ups = np.load('./tests/data/ups_siling.npy').tolist()
    dns = np.load('./tests/data/dns_siling.npy').tolist()

    expect_peaks = np.load('./tests/data/peaks_siling.npy').tolist()
    expect_bottoms = np.load('./tests/data/bottoms_siling.npy').tolist()

    peaks, bottoms = mark.peaks_bottoms(ups, dns)

    np.testing.assert_allclose(peaks, expect_peaks)
    np.testing.assert_allclose(bottoms, expect_bottoms)


def test_peaks_and_bottoms_trop():
    ups = np.load('./tests/data/ups_trop.npy').tolist()
    dns = np.load('./tests/data/dns_trop.npy').tolist()

    peaks, bottoms = mark.peaks_bottoms(ups, dns)

    expect_peaks = np.load('./tests/data/peaks_trop.npy').tolist()
    expect_bottoms = np.load('./tests/data/bottoms_trop.npy').tolist()

    np.testing.assert_allclose(peaks, expect_peaks)
    np.testing.assert_allclose(bottoms, expect_bottoms)


def test_interpolated_org_trop():
    if Path('./tests/data/data_org_trop.csv').exists():
        data = pd.read_csv('./tests/data/data_org_trop.csv', index_col=0,
                           parse_dates=True).iloc[:, 0]

        interpolated = mark.interpolated(data)
        expect_interpolated = np.load('./tests/data/interpolated_trop_org.npy')

        np.testing.assert_allclose(interpolated, expect_interpolated)


def test_interpolated_mod_trop():
    if Path('./tests/data/data_mod_trop.csv').exists():
        data = pd.read_csv('./tests/data/data_mod_trop.csv', index_col=0,
                           parse_dates=True).iloc[:, 0]

        interpolated = mark.interpolated(data)
        expect_interpolated = np.load('./tests/data/interpolated_trop_org.npy')

        np.testing.assert_allclose(interpolated, expect_interpolated)


def test_interpolated_siling():
    data = pd.read_csv('./tests/data/data_siling.csv', index_col=0,
                       parse_dates=True).iloc[:, 0]

    interpolated = mark.interpolated(data)
    expect_interpolated = np.load('./tests/data/interpolated_siling.npy')

    np.testing.assert_allclose(interpolated, expect_interpolated)
