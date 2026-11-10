# 난수 seed, 입력 검사, Cholesky 분해와 로그가능도 계산.

import math

import numpy as np
import torch


def set_seed(seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_coordinates(coordinates, name="coordinates"):
    if not isinstance(coordinates, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2).")


def check_values(values, name="values"):
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if values.ndim != 1:
        raise ValueError(f"{name} must have shape (n,).")


def symmetrize(matrix):
    return 0.5 * (matrix + matrix.T)


def safe_cholesky(matrix, jitter=1e-6, maximum_tries=6):
    identity = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)

    matrix = symmetrize(matrix)
    current_jitter = float(jitter)

    # 분해가 실패할 때만 jitter를 10배 늘려 다시 시도한다.
    for _ in range(maximum_tries):
        try:
            factor = torch.linalg.cholesky(matrix + current_jitter * identity)
            return factor, current_jitter
        except RuntimeError:
            current_jitter *= 10.0

    raise RuntimeError("Cholesky decomposition failed.")


def multivariate_normal_log_likelihood(values, mean, covariance):
    residual = (values - mean).unsqueeze(1)
    factor, _ = safe_cholesky(covariance)

    solved_residual = torch.cholesky_solve(residual, factor)
    quadratic_term = (residual.T @ solved_residual).squeeze()

    log_determinant = 2.0 * torch.sum(torch.log(torch.diagonal(factor)))

    number_of_values = values.shape[0]

    # 항의 계산 및 합산 순서는 원본과 동일하게 유지한다.
    return (
        -0.5 * quadratic_term
        -0.5 * log_determinant
        -0.5 * number_of_values * math.log(2.0 * math.pi)
    )


def pairwise_distance_quantile(coordinates, quantile=0.25):
    with torch.no_grad():
        distances = torch.pdist(coordinates.detach(), p=2)

        if distances.numel() == 0:
            return 1.0

        # 보간하지 않고 기존 kthvalue 순서통계량을 사용한다.
        number_of_distances = int(distances.numel())
        order = int(math.ceil(float(quantile) * number_of_distances))
        order = max(1, min(number_of_distances, order))

        value = torch.kthvalue(distances, order).values
        value = float(value.detach().cpu().item())

    return value if value > 0.0 else 1.0


def clone_state_dict_to_cpu(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def to_device_index_list(index_list, device):
    return [torch.as_tensor(index, dtype=torch.long, device=device) for index in index_list]


def move_tensor_list(tensors, device):
    if tensors is None:
        return None

    return [tensor.to(device) if torch.is_tensor(tensor) else tensor for tensor in tensors]
