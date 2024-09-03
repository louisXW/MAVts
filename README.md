# Model performance Analysis and Visualization of time series forecasting (MAVts)
This repository contains the code for ["Enhancing algal bloom forecasting: A novel framework for machine learning performance evaluation during periods of special temporal patterns"](https://doi.org/10.1016/j.envsoft.2024.106164).

MAVts is designed for analysis of models' forecasting performance. MAVts provides simple baseline solutions for time series forecasting (including forecasting using the last observation or a rolling average of past observations). MAVts also provide novel algorithms to identify and visualize speical time series periods of interst and evalaute model's performance on these specifical periods. MAVts could be used to analyze the performances of various time series forecasting models (including but not limit to data-driven machine/deep learning models, process-based forecasting models; stastical time series forecasting models).

Specifically, this python package provides:
- methods to mark periods of interest in time series, such as an up or down period, interpolated or possibly manipulated data period, and peaks and bottoms (Figure 1 as an example) (`./src/mavts/mark.py`)
- methods to conveniently visualize the periods of interest as a filmroll of the time series (Animation below as an example) (`./src/mavts/vis.py`).
- methods to analyze the predictions in each period of interest separately (`./src/mavts/vis.py`) 
- methods to compare side-by-side the predictions and error of two models (Figure 2 as an example) (`./src/mavts/vis.py`)
- methods to use the last observation or moving averages as baseline predictions (`./src/mavts/baseline.py`)

![image](https://user-images.githubusercontent.com/22387034/229267197-aae4e18e-7855-441e-a91e-43be988265ee.png#center) 
<p align="center">Figure 1: An example of identified up/down periods, peaks/bottoms, and interpolated periods from the forecasting target time series.</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/22387034/229269362-ca61f3b4-b04e-49d4-bce5-d934a80586e6.gif" width="714" height="400" />
</p>

<p align="center">
Animation: Filmroll of the time series with marked periods 
</p>

![image](https://user-images.githubusercontent.com/22387034/229267805-4896c37a-93fc-49d5-a9ee-30dd5929bc14.png#center)
<p align="center">Figure 2: An example of Visualization of model performance comparison between Gated Recurrent Units (GRU) and ARIMA </p>

## Install

The package requirements are in `./requirements.txt`. Install them with 
```
pip install -r requirements.txt
```
After that run:
```
pip install mavts
```
to install the package

## Developing

If you wish to modify the source code, after installing the requirements, clone or download the git repository and do 
```
pip install -e .
```

For testing we use `pytest`.

## Usage
After installing as above, the package can be used as:

```python
from mavts import mark, vis, baseline

data = pd.read_csv('observations.csv', index_col=0, parse_dates=True).iloc[:, 0]

# to mark all periods of interest
all_data = mark.all_periods(data)

predictions = {}

# to obtain EWM baseline predictions
predictions["ewm"] = [baseline.ewm(data)]

# to obtain Last observation baseline predictions
predictions["last"] = [baseline.last(data)]

# to visualize the errors in each special period
vis.errors_by_periods(data, predictions, './plots/')

```

## Citing
```
@article{XIA2024106164,
title = {Enhancing algal bloom forecasting: A novel framework for machine learning performance evaluation during periods of special temporal patterns},
journal = {Environmental Modelling & Software},
volume = {180},
pages = {106164},
year = {2024},
issn = {1364-8152},
doi = {https://doi.org/10.1016/j.envsoft.2024.106164},
url = {https://www.sciencedirect.com/science/article/pii/S1364815224002251},
author = {Wei Xia and Ilija Ilievski and Christine Ann Shoemaker},
}
```

