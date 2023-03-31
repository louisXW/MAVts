import numpy as np
import numpy.typing as npt


def MSE(y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
    """ Mean Squared Error
        y_true: The ground truth - the target array,
        y_pred: The prediction array
    """
    return np.power(y_true - y_pred, 2).mean()


def RMSE(y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
    """ Root Mean Squared Error (so the errors are in the same scale as the 
    observations)
        y_true: The ground truth - the target array,
        y_pred: The prediction array
    """
    return np.sqrt(MSE(y_true, y_pred))


def MAE(y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
    """ Mean Absolute Error
        y_true: The ground truth - the target array,
        y_pred: The prediction array
    """
    return np.abs(y_true - y_pred).mean()


def MARE(y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
    """Mean Absolute Relative error. 0 obsesrvations will be set to 1e-7.
        y_true: The ground truth - the target array,
        y_pred: The prediction array
    """
    div = y_true.copy()
    div[div == 0] = 1e-7
    return np.abs((y_true - y_pred) / div).mean()


def NSE(y_true: npt.NDArray, y_pred: npt.NDArray, y_true_mean=None) -> float:
    """ Nash–Sutcliffe Efficiency
    if `y_true` is only one element than the denominator will be 0 so it's set
    to 1e-7 to avoid division with zero error.

        y_true: The ground truth - the target array,
        y_pred: The prediction array
    """
    if y_true_mean is None:
        y_true_mean = y_true.mean()
    residuals = np.power(y_true - y_pred, 2).sum()
    totals = np.power(y_true - y_true_mean, 2).sum()

    totals += 1e-7 if totals == 0 else 0

    return 1 - (residuals / totals)
