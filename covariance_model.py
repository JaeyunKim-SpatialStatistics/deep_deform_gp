# 변환 좌표의 거리로 상관을 계산하고, 원본 좌표로 공간 분산을 계산한다.

import math

import torch
import torch.nn as nn

from .bessel_functions import bessel_k
from .numerical_utils import pairwise_distance_quantile


def normalize_covariance_name(name):
    if name is None:
        return "exponential"

    name = str(name).lower().replace("-", "").replace("_", "").replace(" ", "")

    if name in {"exp", "exponential", "matern05", "matern12", "nu05"}:
        return "exponential"
    if name in {"matern15", "matern32", "matern1.5", "matern3/2", "nu15"}:
        return "matern15"
    if name in {"matern25", "matern52", "matern2.5", "matern5/2", "nu25"}:
        return "matern25"
    if name in {
        "matern",
        "materngeneral",
        "maternnu",
        "maternlearnnu",
        "maternestnu",
        "estimatenu",
        "learnnu",
    }:
        return "matern_general"

    raise ValueError("Unsupported covariance model.")


def fixed_nu_from_covariance(name):
    name = normalize_covariance_name(name)

    if name == "exponential":
        return 0.5
    if name == "matern15":
        return 1.5
    if name == "matern25":
        return 2.5
    return None


def _bounded_logit(value, lower, upper, epsilon=1e-6):
    probability = (float(value) - float(lower)) / (float(upper) - float(lower))
    probability = min(max(probability, epsilon), 1.0 - epsilon)
    return math.log(probability / (1.0 - probability))


class MaternCovariance(nn.Module):
    def __init__(
        self,
        covariance_model="exponential",
        estimate_nu=False,
        nu_initial=None,
        nu_lower=0.05,
        nu_upper=5.0,
        use_spatial_variance=False,
        longitude_index=0,
        variance_center=None,
        variance_scale=None,
        use_linear_variance_term=False,
        use_quadratic_variance_term=True,
        linear_variance_initial=0.0,
        quadratic_variance_initial=0.0,
    ):
        super().__init__()

        normalized_name = normalize_covariance_name(covariance_model)

        self.covariance_model = "matern_general" if estimate_nu else normalized_name
        self.estimate_nu = bool(estimate_nu)
        self.nu_lower = float(nu_lower)
        self.nu_upper = float(nu_upper)

        fixed_nu = fixed_nu_from_covariance(normalized_name)
        if nu_initial is None:
            nu_initial = fixed_nu if fixed_nu is not None else 1.5
        self.nu_initial = float(nu_initial)

        self.use_spatial_variance = bool(use_spatial_variance)
        self.longitude_index = int(longitude_index)
        self.variance_center = variance_center
        self.variance_scale = variance_scale
        self.use_linear_variance_term = bool(use_linear_variance_term)
        self.use_quadratic_variance_term = bool(use_quadratic_variance_term)

        self.log_a = nn.Parameter(torch.tensor(0.0))
        self.log_sigma2 = nn.Parameter(torch.tensor(0.0))

        if self.covariance_model == "matern_general":
            raw_nu = _bounded_logit(self.nu_initial, self.nu_lower, self.nu_upper)
            self.raw_nu = nn.Parameter(torch.tensor(raw_nu), requires_grad=self.estimate_nu)
        else:
            self.register_parameter("raw_nu", None)

        if self.use_spatial_variance:
            self.linear_variance_coefficient = nn.Parameter(
                torch.tensor(float(linear_variance_initial)),
                requires_grad=self.use_linear_variance_term,
            )
            self.quadratic_variance_coefficient = nn.Parameter(
                torch.tensor(float(quadratic_variance_initial)),
                requires_grad=self.use_quadratic_variance_term,
            )
        else:
            self.register_parameter("linear_variance_coefficient", None)
            self.register_parameter("quadratic_variance_coefficient", None)

        self.variance_center_ = None
        self.variance_scale_ = None

    @property
    def a(self):
        return torch.exp(self.log_a).clamp(min=1e-6, max=1e6)

    @property
    def sigma2(self):
        return torch.exp(self.log_sigma2).clamp(min=1e-6, max=1e6)

    @property
    def nu(self):
        fixed_nu = fixed_nu_from_covariance(self.covariance_model)

        if fixed_nu is not None:
            return self.log_a.new_tensor(fixed_nu)

        return self.nu_lower + (self.nu_upper - self.nu_lower) * torch.sigmoid(self.raw_nu)

    @torch.no_grad()
    def initialize_parameters(self, transformed_coordinates, original_coordinates, values):
        distance_quantile = pairwise_distance_quantile(transformed_coordinates, quantile=0.25)

        initial_a = 1.0 / distance_quantile
        initial_sigma2 = float(torch.var(values, unbiased=False).item())
        initial_sigma2 = max(initial_sigma2, 1e-2)

        self.log_a.copy_(self.log_a.new_tensor(math.log(initial_a)))
        self.log_sigma2.copy_(self.log_sigma2.new_tensor(math.log(initial_sigma2)))

        if self.use_spatial_variance:
            longitude = original_coordinates[:, self.longitude_index]

            if self.variance_center is None:
                self.variance_center_ = 0.5 * float(torch.amin(longitude) + torch.amax(longitude))
            else:
                self.variance_center_ = float(self.variance_center)

            if self.variance_scale is None:
                self.variance_scale_ = 0.5 * float(torch.amax(longitude) - torch.amin(longitude))
                self.variance_scale_ = max(self.variance_scale_, 1.0)
            else:
                self.variance_scale_ = float(self.variance_scale)

        return self

    def correlation_from_distance(self, distance):
        if self.covariance_model == "exponential":
            return torch.exp(-self.a * distance)

        if self.covariance_model == "matern15":
            scaled_distance = math.sqrt(3.0) * self.a * distance
            return (1.0 + scaled_distance) * torch.exp(-scaled_distance)

        if self.covariance_model == "matern25":
            scaled_distance = math.sqrt(5.0) * self.a * distance
            return (
                1.0 + scaled_distance + scaled_distance ** 2 / 3.0
            ) * torch.exp(-scaled_distance)

        return self._general_matern_correlation(distance)

    def _general_matern_correlation(self, distance, small_distance=1e-5):
        nu = self.nu.to(device=distance.device, dtype=distance.dtype)

        scaled_distance = torch.sqrt(2.0 * nu) * self.a * distance

        # 작은 거리에서는 원본과 동일하게 상관을 1로 처리한다.
        small_mask = scaled_distance <= small_distance
        evaluation_distance = torch.where(
            small_mask,
            torch.ones_like(scaled_distance),
            scaled_distance,
        )

        bessel_value = bessel_k(evaluation_distance, nu)
        tiny = torch.finfo(distance.dtype).tiny

        log_coefficient = (1.0 - nu) * math.log(2.0) - torch.lgamma(nu)

        log_correlation = (
            log_coefficient
            + nu * torch.log(evaluation_distance)
            + torch.log(torch.clamp(bessel_value, min=tiny))
        )

        correlation = torch.exp(log_correlation)
        correlation = torch.where(small_mask, torch.ones_like(correlation), correlation)

        correlation = torch.nan_to_num(correlation, nan=1.0, posinf=1.0, neginf=0.0)

        return torch.clamp(correlation, 0.0, 1.0)

    def spatial_log_variance(self, original_coordinates):
        if not self.use_spatial_variance:
            return torch.zeros(
                original_coordinates.shape[0],
                device=original_coordinates.device,
                dtype=original_coordinates.dtype,
            )

        longitude = original_coordinates[:, self.longitude_index]
        centered_longitude = (longitude - self.variance_center_) / (self.variance_scale_ + 1e-12)

        log_variance = torch.zeros_like(centered_longitude)

        if self.use_linear_variance_term:
            log_variance = log_variance + self.linear_variance_coefficient * centered_longitude

        if self.use_quadratic_variance_term:
            log_variance = (
                log_variance + self.quadratic_variance_coefficient * centered_longitude ** 2
            )

        return torch.clamp(log_variance, min=-30.0, max=30.0)

    def marginal_variance(self, original_coordinates):
        log_variance = self.spatial_log_variance(original_coordinates)
        return self.sigma2 * torch.exp(log_variance)

    def covariance_matrix(
        self,
        transformed_coordinates_1,
        transformed_coordinates_2,
        original_coordinates_1,
        original_coordinates_2,
    ):
        distance = torch.cdist(transformed_coordinates_1, transformed_coordinates_2)

        correlation = self.correlation_from_distance(distance)

        if not self.use_spatial_variance:
            return self.sigma2 * correlation

        # 거리에는 변환 좌표를 쓰지만, 분산의 위치 의존성에는 원본 좌표를 쓴다.
        log_variance_1 = self.spatial_log_variance(original_coordinates_1)
        log_variance_2 = self.spatial_log_variance(original_coordinates_2)

        variance_multiplier = torch.exp(0.5 * (log_variance_1[:, None] + log_variance_2[None, :]))

        return self.sigma2 * correlation * variance_multiplier

    def get_summary(self):
        return {
            "a": float(self.a.detach().cpu()),
            "sigma2": float(self.sigma2.detach().cpu()),
            "nu": float(self.nu.detach().cpu()),
            "covariance_model": self.covariance_model,
            "estimate_nu": self.estimate_nu,
            "nu_lower": self.nu_lower,
            "nu_upper": self.nu_upper,
            "use_spatial_variance": self.use_spatial_variance,
            "longitude_index": self.longitude_index,
            "variance_center": self.variance_center_,
            "variance_scale": self.variance_scale_,
            "use_linear_variance_term": self.use_linear_variance_term,
            "use_quadratic_variance_term": self.use_quadratic_variance_term,
            "linear_variance_coefficient": (
                None
                if self.linear_variance_coefficient is None
                else float(self.linear_variance_coefficient.detach().cpu())
            ),
            "quadratic_variance_coefficient": (
                None
                if self.quadratic_variance_coefficient is None
                else float(self.quadratic_variance_coefficient.detach().cpu())
            ),
        }
