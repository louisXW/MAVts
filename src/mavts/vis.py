""" Model error analysis and time series visualization methods. """
from functools import partial
import multiprocessing as mp
from pathlib import Path
from typing import Dict

import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
import numpy as np
import pandas as pd

from mavts import mark
from mavts import metrics


matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 14})


def errors_by_period(observations: pd.Series | pd.DataFrame,
                     predictions: Dict,
                     out_path: str):
    """ Make error box plots for each model (the box plot is over multiple
    trials if the model is stochastic) and the following metrics: MSE, RMSE,
    MAE, MARE, and NSE.

    If `observations` is pd.Series or pd.DataFrame with one column then 
    it is assumed that they are the observations of the time series and 
    `mavts.mark.all_periods(observations)` will be run to add: 
    "is_interpolated", "is_up_period", "is_dn_period", "is_top", and "is_bottom"
    which are boolean series, indicating whether the row belongs to a special 
    period or not. Separate charts will be done based on this flags.

    Alternatively, `observations` can be pd.DataFrame in which case, it must
    contain all of the above columns as boolean series and the first column 
    to be the observations of the time series.

    `predictions` is a dict of lists. The keys are the model names and the 
    each element in the list contain a trial of predictions pd.Series (i.e. the
    list contains the predictions of multiple trials.

    `out_path` is path to the directory where to save the charts. One jpg file 
    will be created for each metric, with the metric name as the file name.
    """

    if len(observations.shape) == 1 or observations.shape[1] == 1:
        obs = mark.all_periods(observations)
    else:
        obs = observations
    metrics_dict = {"MSE": metrics.MSE,
                    "RMSE": metrics.RMSE,
                    "MAE": metrics.MAE,
                    "MARE": metrics.MARE,
                    "NSE": metrics.NSE}

    out = Path(out_path)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for name, prediction in predictions.items():
        info = {"model": name,
                "is_clean": False,
                "is_all": False}

        info |= {col: False for col in obs.columns[1:]}
        for e, pr_i in enumerate(prediction):
            info["trial_id"] = e
            for col in obs.filter(regex="is_*").columns:
                mask = obs.loc[pr_i.index][col]
                pred = pr_i[mask].values
                test = obs.loc[pr_i.index][mask].values[:, 0]
                scores = {name: fn(test, pred)
                          for name, fn in metrics_dict.items()}

                scores["NSE"] = metrics_dict["NSE"](test, pred, pr_i.mean())
                results.append(info | scores)
                results[-1][col] = True

            mask = obs.loc[pr_i.index]
            mask = ~mask.is_interpolated & ~mask.is_bottom & ~mask.is_top

            pred = pr_i[mask].values
            test = obs.loc[pr_i.index][mask].values[:, 0]
            scores = {name: fn(test, pred) for name, fn in metrics_dict.items()}
            results.append(info | scores)
            results[-1]["is_clean"] = True

    res_df = pd.DataFrame(results)

    masks = {"up_periods": res_df.is_up_period,
             "down_periods": res_df.is_dn_period,
             "Peaks": res_df.is_top,
             "bottoms": res_df.is_bottom,
             "interpolated": res_df.is_interpolated,
             "clean": res_df.is_clean}

    for metric in metrics_dict.keys():
        fig, axes = plt.subplots(2, 3, figsize=(12.5, 7))
        for ax, (name, mask) in zip(axes.reshape(-1), masks.items()):
            masked = res_df[mask]
            idx = masked.groupby("model").median().sort_values(by=metric).index
            if metric != "NSE":
                idx = idx[::-1]
            groups = masked.groupby("model")[metric]
            boxes = {k: groups.get_group(k) for k in idx}
            labels = boxes.keys()
            ax.boxplot(boxes.values(), labels=labels, vert=False)
            xmn, xmx = ax.get_xlim()
            xrange = xmx - xmn
            ax.set_xlim(xmn-(xrange*0.1), xmx+(xrange*0.1))

            for x, y in enumerate(boxes.values()):
                ax.text(y.median(), x+1, f"{y.median():.2f}",
                        size=10,
                        ha="center",
                        va="center",
                        color="green",
                        bbox={"fc": "#ffffff", "ec": "None", "pad": 0.75})

            ax.set_title(name.replace('_', ' ').title())
            ax.tick_params(axis='both', which='major', labelsize=10, left=False)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        plt.tight_layout()
        fig_fn = out.joinpath(f"{metric}.jpg")
        plt.savefig(fig_fn, dpi=500)
        plt.close(fig)


def models_as_filmroll(data: pd.Series,
                       model_A: pd.Series,
                       model_B: pd.Series | None,
                       out_path: str,
                       periods: dict | None = None,
                       scale_y: bool = False,
                       plot_timespan='13w',
                       plot_timestep='1w',
                       figsize=(16, 9)):
    """
    Plot a film roll of the observations in `data`, with the special `periods`,
    and the predictions of `model_A` and `model_B`. Save the charts at 
    "`out_path`/{model_A.name}_vs_{model_B.name}/filmroll/frame***.jpg"
    Where the model names are the pd.Series names.

    `periods` can be a dict of:
    dict(ups=ups,
         dns=dns,
         peaks=peaks,
         bottoms=bottoms,
         interpolated=interpolated)
    or None, in which case it will be populated by calling the respective 
    methods from `mavts.mark`.

    `scale_y` if True will scale the observations from 0 to 1
    `plot_timespan` is the time span of each frame in pandas timespan format
    `plot_timestep` is the difference between each consecutive frames in pandas
    timespan format
    `figsize` is the figsize parameter passed to plt.subplots
    """

    mn1 = model_A.name
    mn2 = '' if model_B is None else model_B.name

    plot_out = Path(out_path)
    plot_out.mkdir(parents=True, exist_ok=True)
    out = plot_out.joinpath(f"{mn1}_vs_{mn2}/filmroll/")
    out.mkdir(parents=True, exist_ok=True)

    idx = data.index[0]
    edx = idx + pd.Timedelta(plot_timespan)  # type:ignore
    frames = []
    while edx <= data.index[-1]:
        frames.append((idx, edx, out.joinpath(f"frame{len(frames):04d}.jpg")))
        idx += pd.Timedelta(plot_timestep)  # type:ignore
        edx = idx + pd.Timedelta(plot_timespan)

    if periods is None:
        ups, dns = mark.up_dn_periods(data.values)  # type:ignore
        peaks, bottoms = mark.peaks_bottoms(ups, dns)
        interpolated = mark.interpolated(data)

        periods = dict(ups=ups,
                       dns=dns,
                       peaks=peaks,
                       bottoms=bottoms,
                       interpolated=interpolated)

    suptitle = f"Model Comparison Between {mn1} And {mn2}"

    with mp.Pool() as pool:
        pool.map_async(partial(_plot_cmp_frame,
                               suptitle=suptitle,
                               data=data,
                               ups=periods["ups"],
                               dns=periods["dns"],
                               peaks=periods["peaks"],
                               bottoms=periods["bottoms"],
                               interpolated=periods["interpolated"],
                               scale_y=scale_y,
                               ymin=data.min(),
                               ymax=data.max(),
                               figsize=figsize,
                               model_A=model_A,
                               model_B=model_B), frames).get()


def _plot_cmp_frame(arg, data, suptitle, ups, dns, peaks, bottoms, interpolated,
                    model_A, scale_y, ymin, ymax, figsize,
                    model_B=None, d=7, q=.95, show_conformal=False):
    """
    Plot each individual frame from `models_as_filmroll` in a format suitable 
    for multiprocessing.

    `arg` contains as a tuple: the start index, the end index, and the 
    filename of the plot.

    'data' is the all observations data, from which we will get a slice 
    according to data.loc[arg[0]:arg[1]]

    `suptitle` is the title of the plot

    ups, dns, peaks, bottoms, interpolated are the special periods

    `model_A` and `model_B` are pd.Series of predictions

    `scale_y` is a boolean if true it will scale the observations from 0 to 1

    ymin and ymax are the min and max of the yaxis

    `figsize` is the figsize parameter passed to plt.subplots

    d is the number of days as int for the conformal prediction interval
    q is the quantile for the conformal prediction interval
    show_conformal if true will show the conformal prediction intervals

    """
    lb, ub, fig_fn = arg

    bidx = data.index.get_loc(lb)
    eidx = data.index.get_loc(ub) if ub in data.index else data.shape[-1]

    def scale(p):
        if scale_y:
            return (p-ymin)/(ymax-ymin)
        else:
            return p

    if scale_y:
        ylim = (-0.045, 1.045)
        yticks = np.arange(-0.1, 1.1, 0.1)
        ylbl = f"Normalized {data.name}"
    else:
        ylim, yticks = None, None
        ylbl = data.name

    df = scale(data)
    pr1 = scale(model_A)
    if model_B is not None:
        pr2 = scale(model_B)
    else:
        pr2 = None

    mn1 = model_A.name
    mn2 = '' if model_B is None else model_B.name

    fig, axes = plt.subplots(2, 1,
                             gridspec_kw={'height_ratios': [4, 1], 'hspace': 0},
                             figsize=figsize,
                             sharex=True)

    axes[0].yaxis.set_minor_locator(AutoMinorLocator(2))
    axes[1].spines['top'].set_visible(False)

    dev1 = (pr1 - df)
    dev2 = (pr2 - df)
    cp1 = dev1.abs().rolling(d).quantile(q)
    cp2 = dev2.abs().rolling(d).quantile(q)

    ax = axes[0]
    df.iloc[bidx:eidx].plot(ax=ax,
                            c='black',
                            lw=2,
                            alpha=0.6,
                            marker='o',
                            markersize=5.5)

    for array, color in zip([ups, dns], ["tab:green", "tab:red"]):
        for left, right in array:
            if bidx <= left <= eidx or bidx <= right <= eidx:
                ax.axvspan(df.index[left], df.index[right],
                           alpha=0.3, color=color, zorder=-5)

    pr1.iloc[bidx:eidx].plot(ax=ax,
                             c='tab:blue',
                             marker='s',
                             lw=2,
                             markersize=6,
                             alpha=0.8)

    if show_conformal:
        ax.fill_between(pr1.index[bidx:eidx],
                        (pr1-cp1).iloc[bidx:eidx],
                        (pr1+cp1).iloc[bidx:eidx],
                        zorder=-100,
                        edgecolor='tab:blue',
                        ls="--",
                        facecolor="None",
                        lw=2)

    if pr2 is not None:
        pr2.iloc[bidx:eidx].plot(ax=ax,
                                 c='tab:red',
                                 marker='D',
                                 markersize=6,
                                 lw=2,
                                 alpha=0.8)

        if show_conformal:
            ax.fill_between(pr2.index[bidx:eidx],
                            (pr2-cp2).iloc[bidx:eidx],
                            (pr2+cp2).iloc[bidx:eidx],
                            zorder=-100,
                            edgecolor='tab:red',
                            ls="--",
                            facecolor="None",
                            lw=2)

    for array, color in zip([peaks, bottoms], ["lime", "red"]):
        for idx in array:
            if bidx <= idx <= eidx:
                ax.plot([df.index[idx]], [df.iloc[idx]],
                        marker='D', color=color,
                        markersize=7, ls='')

    for left, right in interpolated:
        if bidx <= left <= eidx or bidx <= right <= eidx:
            for i in range(left, right+1):
                if bidx <= i and i < eidx:
                    ax.plot([df.index[i]], [df.iloc[i]],
                            marker='o', c='yellow', ls='', markersize=4)

    handles, labels = [], []
    handles.append(Line2D([], [], color='black', marker='o', markersize=5.5))
    labels.append("Observations")

    handles.append(Patch(color='tab:green', alpha=0.3))
    labels.append("Up periods")

    handles.append(Patch(color='tab:red', alpha=0.3))
    labels.append("Down periods")

    handles.append(Line2D([], [], ls='', color="lime",
                   marker='D', markersize=7))
    labels.append("Peaks")

    handles.append(Line2D([], [], ls='', color="red",
                   marker='D', markersize=7))
    labels.append("Bottoms")

    handles.append(Line2D([], [], ls='', color="yellow",
                   marker='o', markeredgecolor='black',
                          markersize=7, markeredgewidth=1))
    labels.append("Interpolated")
    handles.append(Line2D([], [], color='tab:blue', marker='s', markersize=6.5))
    labels.append(f"{mn1} Forecast")
    if show_conformal:
        handles.append(Line2D([], [], color='tab:blue', ls='--'))
        labels.append(f"{mn1} 95% Conformal Interval")
    if pr2 is not None:
        handles.append(Line2D([], [], color='tab:red',
                       marker='D', markersize=6.5))
        labels.append(f"{mn2} Forecast")
        if show_conformal:
            handles.append(Line2D([], [], color='tab:red', ls='--'))
            labels.append(f"{mn2} 95% Conformal Interval")

    ax.legend(handles, labels, loc='lower right')

    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_ylim(ylim)
    ax.set_ylabel(ylbl)
    ax.grid(zorder=-110, alpha=1, lw=0.4, which='both')

    if pr2 is not None:
        axes[1].bar(dev1.index[bidx:eidx],  # np.arange(0, eidx-bidx)-.15,
                    dev1.iloc[bidx:eidx],
                    width=.80, edgecolor='#0000FFAA', facecolor="#0000FF22",
                    linewidth=2, label=mn1)
        axes[1].bar(dev2.index[bidx:eidx],  # np.arange(0, eidx-bidx)+.25,
                    dev2.iloc[bidx:eidx],
                    width=.80, edgecolor='#FF0000AA', facecolor="#FF000022",
                    linewidth=2, label=mn2)
    else:
        axes[1].bar(dev1.index[bidx:eidx],  # np.arange(0, eidx-bidx),
                    dev1.iloc[bidx:eidx],
                    color='tab:blue')

    axes[1].set_ylabel("Error")
    axes[1].legend(loc='lower right')
    axes[1].grid(zorder=-110, alpha=1, lw=0.2, which='both')
    axes[1].yaxis.set_minor_locator(AutoMinorLocator(2))
    axes[1].yaxis.label.set_color('red')
    axes[1].tick_params(axis='y', colors='red')

    plt.suptitle(suptitle, fontsize="xx-large")
    plt.tight_layout()
    plt.savefig(fig_fn, dpi=300)
    plt.close(fig)


def observations_as_filmroll(data: pd.Series,
                             out_path: str,
                             periods: dict | None = None,
                             scale_y: bool = False,
                             plot_timespan='13w',
                             plot_timestep='1w',
                             figsize=(12.5, 7)):
    """
    Plot a film roll of the observations in `data`, with the special `periods`.
    "`out_path`/filmroll/frame***.jpg"

    `periods` can be a dict of:
    dict(ups=ups,
         dns=dns,
         peaks=peaks,
         bottoms=bottoms,
         interpolated=interpolated)
    or None, in which case it will be populated by calling the respective 
    methods from `mavts.mark`.

    `scale_y` if True will scale the observations from 0 to 1
    `plot_timespan` is the time span of each frame in pandas timespan format
    `plot_timestep` is the difference between each consecutive frames in pandas
    timespan format
    `figsize` is the figsize parameter passed to plt.subplots
    """

    plot_out = Path(out_path)
    plot_out.mkdir(parents=True, exist_ok=True)
    out = plot_out.joinpath("filmroll")
    out.mkdir(parents=True, exist_ok=True)

    idx = data.index[0]
    edx = idx + pd.Timedelta(plot_timespan)  # type:ignore
    frames = []
    while edx <= data.index[-1]:
        frames.append((idx, edx, out.joinpath(f"frame{len(frames):04d}.jpg")))
        idx += pd.Timedelta(plot_timestep)  # type:ignore
        edx = idx + pd.Timedelta(plot_timespan)

    if periods is None:
        ups, dns = mark.up_dn_periods(data.values)  # type:ignore
        peaks, bottoms = mark.peaks_bottoms(ups, dns)
        interpolated = mark.interpolated(data)

        periods = dict(ups=ups,
                       dns=dns,
                       peaks=peaks,
                       bottoms=bottoms,
                       interpolated=interpolated)

    with mp.Pool() as pool:
        i = 0
        for _ in pool.imap_unordered(partial(_plot_obs_frame,
                                             data=data,
                                             ups=periods["ups"],
                                             dns=periods["dns"],
                                             peaks=periods["peaks"],
                                             bottoms=periods["bottoms"],
                                             interpolated=periods["interpolated"],
                                             figsize=figsize,
                                             scale_y=scale_y,
                                             ymin=data.min(),
                                             ymax=data.max()), frames):
            i += 1
            p = i / len(frames)


def _plot_obs_frame(arg, data, ups, dns, peaks, bottoms, interpolated,
                    figsize, scale_y, ymin, ymax):
    """
    Plot each individual frame from `observations_as_filmroll` in a format suitable 
    for multiprocessing.

    `arg` contains as a tuple: the start index, the end index, and the 
    filename of the plot.

    'data' is the all observations data, from which we will get a slice 
    according to data.loc[arg[0]:arg[1]]

    ups, dns, peaks, bottoms, interpolated are the special periods

    `figsize` is the figsize parameter passed to plt.subplots

    `scale_y` is a boolean if true it will scale the observations from 0 to 1

    ymin and ymax are the min and max of the yaxis

    """
    lb, ub, fig_fn = arg

    bidx = data.index.get_loc(lb)
    eidx = data.index.get_loc(ub) if ub in data.index else data.shape[-1]

    def scale(p):
        if scale_y:
            return (p-ymin)/(ymax-ymin)
        else:
            return p

    if scale_y:
        ylim = (-0.045, 1.045)
        yticks = np.arange(-0.1, 1.1, 0.1)
        ylbl = f"Normalized {data.name}"
    else:
        ylim, yticks = None, None
        ylbl = data.name

    fig, ax = plt.subplots(figsize=figsize, squeeze=False)
    ax = ax[0, 0]
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    scale(data.iloc[bidx:eidx]).plot(ax=ax,
                                     c='black',
                                     zorder=11,
                                     lw=2,
                                     alpha=1.0,
                                     marker='o',
                                     markersize=5.5)

    for array, color in zip([ups, dns], ["tab:green", "tab:red"]):
        for left, right in array:
            if bidx <= left <= eidx or bidx <= right <= eidx:
                ax.axvspan(data.index[left], data.index[right],
                           alpha=0.3, color=color, zorder=-1)

    for array, color in zip([peaks, bottoms], ["lime", "red"]):
        for idx in array:
            if bidx <= idx <= eidx:
                ax.plot([data.index[idx]], [scale(data.iloc[idx])],
                        marker='D', color=color,
                        markersize=7, zorder=1000, ls='')

    for left, right in interpolated:
        if bidx <= left <= eidx or bidx <= right <= eidx:
            for i in range(left, right+1):
                ax.plot([data.index[i]], [scale(data.iloc[i])],
                        marker='o', c='yellow', ls='', markersize=4,
                        zorder=1000)

    handles, labels = [], []
    handles.append(Line2D([], [], color='black', marker='o', markersize=5.5))
    labels.append("Observations")

    handles.append(Patch(color='tab:green', alpha=0.3))
    labels.append("Up periods")

    handles.append(Patch(color='tab:red', alpha=0.3))
    labels.append("Down periods")

    handles.append(Line2D([], [], ls='', color="lime",
                   marker='D', markersize=7))
    labels.append("Peaks")

    handles.append(Line2D([], [], ls='', color="red",
                   marker='D', markersize=7))
    labels.append("Bottoms")

    handles.append(Line2D([], [], ls='', color="yellow",
                   marker='o', markeredgecolor='black',
                          markersize=7, markeredgewidth=1))
    labels.append("Interpolated")
    ax.legend(handles, labels, loc='upper left')

    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_ylim(ylim)
    ax.set_ylabel(ylbl)
    ax.grid(zorder=-110, alpha=1, lw=0.4, which='both')

    plt.tight_layout()
    plt.savefig(fig_fn, dpi=300)
    plt.close(fig)
