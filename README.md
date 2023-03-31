# Model performance Analysis and Visualization of time series forecasting (MAVts)

This python package provides:
- methods to mark periods of interest in time series, such as an up or down period, interpolated or possibly manipulated data period, and peaks and bottoms (`./src/mavts/mark.py`)
- methods to conveniently visualize the periods of interest as a filmroll of the time series (`./src/mavts/vis.py`).
- methods to analyze the predictions in each period of interest separately (`./src/mavts/vis.py`) 
- methods to compare side-by-side the predictions and error of two model (`./src/mavts/vis.py`)s
- methods to use the last observation or moving averages as baseline predictions (`./src/mavts/baseline.py`)

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
TODO
