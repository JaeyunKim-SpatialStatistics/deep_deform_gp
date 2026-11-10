# Gaussian 예측의 RMSE, MAE, CRPS를 계산한다.

import math

import torch


def _as_vector(values, dtype=None):
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(
            values,
            dtype=dtype if dtype is not None else torch.get_default_dtype(),
        )
    else:
        values = values.detach()
        if dtype is not None:
            values = values.to(dtype)

    return values.reshape(-1)


def rmse(y_true, y_pred):
    y_true = _as_vector(y_true)
    y_pred = _as_vector(y_pred, dtype=y_true.dtype)
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()


def mae(y_true, y_pred):
    y_true = _as_vector(y_true)
    y_pred = _as_vector(y_pred, dtype=y_true.dtype)
    return torch.mean(torch.abs(y_true - y_pred)).item()


def _standard_normal_pdf(value):
    return torch.exp(-0.5 * value ** 2) / math.sqrt(2.0 * math.pi)


def _standard_normal_cdf(value):
    return 0.5 * (1.0 + torch.erf(value / math.sqrt(2.0)))


def crps_gaussian(y_true, prediction_mean, prediction_std, epsilon=1e-12):
    y_true = _as_vector(y_true)
    prediction_mean = _as_vector(prediction_mean, dtype=y_true.dtype)
    prediction_std = _as_vector(prediction_std, dtype=y_true.dtype)
    prediction_std = torch.clamp(prediction_std, min=epsilon)

    standardized_error = (y_true - prediction_mean) / prediction_std

    score = prediction_std * (
        standardized_error * (2.0 * _standard_normal_cdf(standardized_error) - 1.0)
        + 2.0 * _standard_normal_pdf(standardized_error)
        - 1.0 / math.sqrt(math.pi)
    )

    return torch.mean(score).item()


def evaluate_gaussian_prediction(y_true, prediction_mean, prediction_std):
    return {
        "rmse": rmse(y_true, prediction_mean),
        "mae": mae(y_true, prediction_mean),
        "crps": crps_gaussian(y_true, prediction_mean, prediction_std),
    }
