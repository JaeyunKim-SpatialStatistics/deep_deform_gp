# SciPy의 Bessel K 계산과 PyTorch의 자동미분을 연결한다.

import atexit
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import torch
from scipy.special import kv as scipy_kv
from scipy.special import kvp as scipy_kvp


# 병렬 실행 설정과 재사용할 executor.
_PARALLEL = os.environ.get("BESSEL_SCIPY_PARALLEL", "1").lower() not in {"0", "false", "no", "off"}
_BACKEND = os.environ.get("BESSEL_SCIPY_BACKEND", "thread").lower().strip()
_WORKERS = int(os.environ.get("BESSEL_SCIPY_WORKERS", "0"))
_CHUNKS = int(os.environ.get("BESSEL_SCIPY_CHUNKS", "0"))
_MINIMUM_SIZE = int(os.environ.get("BESSEL_SCIPY_MIN_SIZE", "20000"))
_EXECUTOR = None


def _worker_count():
    if _WORKERS > 0:
        return max(1, _WORKERS)
    return max(1, int(os.cpu_count() or 1))


def get_bessel_scipy_parallel_config():
    workers = _worker_count()
    chunks = _CHUNKS if _CHUNKS > 0 else workers

    return {
        "parallel": bool(_PARALLEL),
        "backend": _BACKEND,
        "workers": int(workers),
        "chunks": int(chunks),
        "min_size": int(_MINIMUM_SIZE),
    }


def shutdown_bessel_scipy_executor():
    global _EXECUTOR

    if _EXECUTOR is not None:
        _EXECUTOR.shutdown(wait=True, cancel_futures=False)
        _EXECUTOR = None


atexit.register(shutdown_bessel_scipy_executor)


def set_bessel_scipy_parallel(
    workers=None,
    chunks=None,
    min_size=None,
    backend=None,
    parallel=True,
):
    global _PARALLEL, _BACKEND, _WORKERS, _CHUNKS, _MINIMUM_SIZE

    shutdown_bessel_scipy_executor()

    _PARALLEL = bool(parallel)

    if backend is not None:
        backend = str(backend).lower().strip()
        if backend not in {"thread", "process"}:
            raise ValueError("backend must be 'thread' or 'process'.")
        _BACKEND = backend

    if workers is not None:
        _WORKERS = int(workers)
    if chunks is not None:
        _CHUNKS = int(chunks)
    if min_size is not None:
        _MINIMUM_SIZE = int(min_size)

    return get_bessel_scipy_parallel_config()


def _get_executor():
    global _EXECUTOR

    if _EXECUTOR is not None:
        return _EXECUTOR

    if _BACKEND == "process":
        _EXECUTOR = ProcessPoolExecutor(max_workers=_worker_count())
    else:
        _EXECUTOR = ThreadPoolExecutor(max_workers=_worker_count())

    return _EXECUTOR


# 각 worker는 전달받은 배열 조각에 대해 SciPy 계산을 수행한다.
def _kv_worker(arguments):
    nu, values = arguments
    return scipy_kv(float(nu), values)


def _kvp_worker(arguments):
    nu, values = arguments
    return scipy_kvp(float(nu), values, n=1)


def _kv_pair_worker(arguments):
    nu_plus, nu_minus, values = arguments
    return (scipy_kv(float(nu_plus), values), scipy_kv(float(nu_minus), values))


def _use_parallel(values):
    return _PARALLEL and _worker_count() > 1 and (np.asarray(values).size >= _MINIMUM_SIZE)


def _split_values(values):
    flat_values = np.ascontiguousarray(np.asarray(values, dtype=np.float64).reshape(-1))

    number_of_chunks = _CHUNKS if _CHUNKS > 0 else _worker_count()
    number_of_chunks = max(1, min(number_of_chunks, flat_values.size))

    return [chunk for chunk in np.array_split(flat_values, number_of_chunks) if chunk.size > 0]


def _join_results(results, shape):
    flat_result = np.concatenate(
        [np.asarray(result, dtype=np.float64).reshape(-1) for result in results]
    )
    return flat_result.reshape(shape)


def _evaluate_kv(nu, values):
    values = np.asarray(values, dtype=np.float64)

    if not _use_parallel(values):
        return np.asarray(scipy_kv(float(nu), values), dtype=np.float64)

    chunks = _split_values(values)
    results = list(_get_executor().map(_kv_worker, [(nu, chunk) for chunk in chunks]))

    return _join_results(results, values.shape)


def _evaluate_kvp(nu, values):
    values = np.asarray(values, dtype=np.float64)

    if not _use_parallel(values):
        return np.asarray(scipy_kvp(float(nu), values, n=1), dtype=np.float64)

    chunks = _split_values(values)
    results = list(_get_executor().map(_kvp_worker, [(nu, chunk) for chunk in chunks]))

    return _join_results(results, values.shape)


def _evaluate_kv_pair(nu_plus, nu_minus, values):
    values = np.asarray(values, dtype=np.float64)

    if not _use_parallel(values):
        return (
            np.asarray(scipy_kv(float(nu_plus), values), dtype=np.float64),
            np.asarray(scipy_kv(float(nu_minus), values), dtype=np.float64),
        )

    chunks = _split_values(values)
    results = list(
        _get_executor().map(_kv_pair_worker, [(nu_plus, nu_minus, chunk) for chunk in chunks])
    )

    plus = _join_results([result[0] for result in results], values.shape)
    minus = _join_results([result[1] for result in results], values.shape)
    return plus, minus


# Forward는 SciPy로 평가하고, backward에서 입력과 nu의 gradient를 반환한다.
class BesselKFunction(torch.autograd.Function):
    @staticmethod
    def forward(context, values, nu):
        values_numpy = values.detach().cpu().numpy().astype(np.float64)
        nu_value = float(nu.detach().cpu().reshape(-1)[0].item())

        result_numpy = _evaluate_kv(nu_value, values_numpy)
        result = torch.as_tensor(result_numpy, device=values.device, dtype=values.dtype)

        context.save_for_backward(values, nu)
        return result

    @staticmethod
    def backward(context, gradient_output):
        values, nu = context.saved_tensors

        values_numpy = values.detach().cpu().numpy().astype(np.float64)
        nu_value = float(nu.detach().cpu().reshape(-1)[0].item())

        # 입력에 대한 미분은 kvp, nu에 대한 미분은 기존 유한차분식을 사용한다.
        derivative_values = _evaluate_kvp(nu_value, values_numpy)

        step = 1e-4
        nu_plus = nu_value + step
        nu_minus = max(nu_value - step, 1e-6)

        value_plus, value_minus = _evaluate_kv_pair(nu_plus, nu_minus, values_numpy)

        derivative_nu = (value_plus - value_minus) / (nu_plus - nu_minus)

        derivative_values = torch.as_tensor(
            derivative_values,
            device=values.device,
            dtype=values.dtype,
        )

        derivative_nu = torch.as_tensor(derivative_nu, device=values.device, dtype=values.dtype)

        gradient_values = gradient_output * derivative_values
        gradient_nu = torch.sum(gradient_output * derivative_nu).reshape_as(nu)

        return gradient_values, gradient_nu


def bessel_k(values, nu):
    return BesselKFunction.apply(values, nu)
