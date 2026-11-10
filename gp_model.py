# 좌표 변환, 공분산, exact/Vecchia 가능도를 연결하는 GP 모형.

import math

import torch
import torch.nn as nn

from .numerical_utils import (
    check_coordinates,
    check_values,
    multivariate_normal_log_likelihood,
)


class SpatialGPModel(nn.Module):
    def __init__(
        self,
        coordinate_scaler,
        covariance_model,
        spatial_warper=None,
        vecchia_approximation=None,
        zero_mean=False,
        use_gpu=True,
        dtype=torch.float32,
    ):
        super().__init__()

        self.coordinate_scaler = coordinate_scaler
        self.covariance_model = covariance_model
        self.spatial_warper = spatial_warper
        self.vecchia = vecchia_approximation

        self.zero_mean = bool(zero_mean)
        self.dtype = dtype
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        self.mean = nn.Parameter(torch.tensor(0.0), requires_grad=not self.zero_mean)
        self.log_tau2 = nn.Parameter(torch.tensor(-4.0))

        self.is_fitted = False
        self.number_of_training_points = None
        self.best_negative_log_likelihood = None
        self.best_epoch = None
        self.stopped_early = False
        self.training_time = None
        self.block_diagnostics = None

        self.to_device(self.device)

    @property
    def tau2(self):
        return torch.exp(self.log_tau2).clamp(min=1e-8, max=1e6)

    @property
    def uses_warping(self):
        return self.spatial_warper is not None

    @property
    def uses_vecchia(self):
        return self.vecchia is not None

    @torch.no_grad()
    def initialize_parameters(self, original_coordinates, values):
        base_coordinates = self.base_coordinates(original_coordinates)
        transformed_coordinates = self.transform_base_coordinates(
            base_coordinates,
            use_saved_bounds=False,
        )

        self.covariance_model.initialize_parameters(
            transformed_coordinates,
            original_coordinates,
            values,
        )

        initial_sigma2 = float(torch.var(values, unbiased=False).item())
        initial_sigma2 = max(initial_sigma2, 1e-2)
        initial_tau2 = max(1e-3, 0.05 * initial_sigma2)
        initial_mean = 0.0 if self.zero_mean else float(values.mean().item())

        self.log_tau2.copy_(self.log_tau2.new_tensor(math.log(initial_tau2)))
        self.mean.copy_(self.mean.new_tensor(initial_mean))

    # original: 입력 좌표 / base: scaling 후 좌표 / transformed: warping 후 좌표.
    def base_coordinates(self, original_coordinates, check_domain=True):
        original_coordinates = original_coordinates.to(device=self.device, dtype=self.dtype)

        return self.coordinate_scaler.transform(original_coordinates, check_domain=check_domain)

    def transform_base_coordinates(self, base_coordinates, use_saved_bounds):
        if self.spatial_warper is None:
            return base_coordinates

        return self.spatial_warper(base_coordinates, use_saved_bounds=use_saved_bounds)

    def transform_coordinates(
        self,
        original_coordinates,
        use_saved_bounds=False,
        check_domain=True,
    ):
        base_coordinates = self.base_coordinates(original_coordinates, check_domain=check_domain)

        return self.transform_base_coordinates(base_coordinates, use_saved_bounds=use_saved_bounds)

    def log_likelihood(self, original_coordinates, values):
        check_coordinates(original_coordinates)
        check_values(values)

        original_coordinates = original_coordinates.to(device=self.device, dtype=self.dtype)
        values = values.to(device=self.device, dtype=self.dtype)

        transformed_coordinates = self.transform_coordinates(
            original_coordinates,
            use_saved_bounds=False,
        )

        # 설정에 따라 Vecchia 근사 또는 exact GP 가능도를 계산한다.
        if self.vecchia is not None:
            return self.vecchia.log_likelihood(
                model=self,
                values=values,
                transformed_coordinates=transformed_coordinates,
                original_coordinates=original_coordinates,
            )

        return self._exact_log_likelihood(values, transformed_coordinates, original_coordinates)

    def _exact_log_likelihood(self, values, transformed_coordinates, original_coordinates):
        process_covariance = self.covariance_model.covariance_matrix(
            transformed_coordinates,
            transformed_coordinates,
            original_coordinates,
            original_coordinates,
        )

        # 관측 오차 분산과 기존의 수치 안정화 상수를 그대로 더한다.
        observation_covariance = process_covariance + (
            self.tau2 + 1e-6
        ) * torch.eye(values.shape[0], device=values.device, dtype=values.dtype)

        mean_vector = self.mean.expand_as(values)

        return multivariate_normal_log_likelihood(values, mean_vector, observation_covariance)

    def gp_parameters(self):
        parameters = list(self.covariance_model.parameters())
        parameters.append(self.log_tau2)

        if self.mean.requires_grad:
            parameters.append(self.mean)

        return [parameter for parameter in parameters if parameter.requires_grad]

    def warping_parameters(self):
        if self.spatial_warper is None:
            return []

        return [
            parameter
            for parameter in self.spatial_warper.parameters()
            if parameter.requires_grad
        ]

    def save_warping_bounds(self, original_training_coordinates):
        if self.spatial_warper is None:
            return

        base_coordinates = self.base_coordinates(original_training_coordinates, check_domain=False)

        self.spatial_warper.save_training_bounds(base_coordinates)
        self.block_diagnostics = self.spatial_warper.diagnostics(base_coordinates)

    @torch.no_grad()
    def warp_coordinates(self, original_coordinates, check_domain=False):
        return self.transform_coordinates(
            original_coordinates,
            use_saved_bounds=True,
            check_domain=check_domain,
        )

    def get_original_coordinate_bounds(self):
        return self.coordinate_scaler.get_original_bounds()

    def to_device(self, device):
        self.device = torch.device(device)
        self.to(device=self.device, dtype=self.dtype)

        if self.vecchia is not None:
            self.vecchia.to(self.device)

        return self

    def get_estimated_parameters(self):
        estimated = self.covariance_model.get_summary()
        estimated.update({
            "tau2": float(self.tau2.detach().cpu()),
            "mu": float(self.mean.detach().cpu()),
            "zero_mean": self.zero_mean,
            "best_negloglik": self.best_negative_log_likelihood,
            "best_epoch": self.best_epoch,
            "stopped_early": self.stopped_early,
            "training_time_sec": self.training_time,
            "block_diagnostics": self.block_diagnostics,
        })

        estimated.update(self.coordinate_scaler.get_summary())

        if self.spatial_warper is not None:
            estimated.update(self.spatial_warper.get_summary())

        if self.vecchia is not None:
            estimated.update(self.vecchia.get_summary())
        else:
            estimated["approximation"] = False

        return estimated

    def summary(self):
        model_type = "Invertible GP" if self.uses_warping else "Stationary GP"

        return {
            "model_type": model_type,
            "status": "fitted" if self.is_fitted else "not fitted",
            "device": str(self.device),
            "dtype": str(self.dtype),
            "n_train": self.number_of_training_points,
            "estimated_params": self.get_estimated_parameters(),
        }
