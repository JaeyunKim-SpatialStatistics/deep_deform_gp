# 좌표 scaling과 affine coupling block을 이용한 공간 변형.

import warnings

import torch
import torch.nn as nn

from .numerical_utils import check_coordinates


class CoordinateScaler(nn.Module):
    def __init__(
        self,
        mode="train_minmax",
        domain_min=None,
        domain_max=None,
        out_of_domain_policy="warn",
        clamp=False,
        epsilon=1e-8,
    ):
        super().__init__()

        self.mode = str(mode).lower().strip()
        self.out_of_domain_policy = str(out_of_domain_policy).lower().strip()
        self.clamp = bool(clamp)
        self.epsilon = float(epsilon)
        self._warned_outside = False

        if self.mode not in {"none", "train_minmax", "fixed_domain"}:
            raise ValueError("mode must be 'none', 'train_minmax', or 'fixed_domain'.")

        if domain_min is None:
            domain_min = (0.0, 0.0)
        if domain_max is None:
            domain_max = (1.0, 1.0)

        lower = torch.tensor(domain_min, dtype=torch.get_default_dtype()).reshape(1, 2)
        upper = torch.tensor(domain_max, dtype=torch.get_default_dtype()).reshape(1, 2)

        self.register_buffer("lower", lower)
        self.register_buffer("upper", upper)
        self.register_buffer("is_fitted", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def fit(self, coordinates):
        check_coordinates(coordinates)
        coordinates = coordinates.to(device=self.lower.device, dtype=self.lower.dtype)

        if self.mode in {"none", "train_minmax"}:
            self.lower.copy_(torch.amin(coordinates, dim=0, keepdim=True))
            self.upper.copy_(torch.amax(coordinates, dim=0, keepdim=True))

        self.is_fitted.fill_(True)
        self._warned_outside = False
        return self

    def _check_domain(self, coordinates):
        if self.mode == "none" or self.out_of_domain_policy == "allow":
            return

        outside = (
            (coordinates < self.lower - self.epsilon)
            | (coordinates > self.upper + self.epsilon)
        )

        if not bool(torch.any(outside).item()):
            return

        message = (
            "Some coordinates are outside the fitted scaling domain: "
            f"lower={self.lower.detach().cpu().reshape(-1).tolist()}, "
            f"upper={self.upper.detach().cpu().reshape(-1).tolist()}."
        )

        if self.out_of_domain_policy == "error":
            raise ValueError(message)

        if self.out_of_domain_policy == "warn" and not self._warned_outside:
            warnings.warn(message, RuntimeWarning, stacklevel=3)
            self._warned_outside = True

    def transform(self, coordinates, check_domain=True):
        check_coordinates(coordinates)
        coordinates = coordinates.to(device=self.lower.device, dtype=self.lower.dtype)

        if check_domain:
            self._check_domain(coordinates)

        if self.mode == "none":
            scaled_coordinates = coordinates
        else:
            scaled_coordinates = (coordinates - self.lower) / (
                self.upper - self.lower + self.epsilon
            )

        if self.clamp and self.mode != "none":
            scaled_coordinates = torch.clamp(scaled_coordinates, 0.0, 1.0)

        return scaled_coordinates

    def inverse_transform(self, scaled_coordinates):
        check_coordinates(scaled_coordinates, "scaled_coordinates")
        scaled_coordinates = scaled_coordinates.to(device=self.lower.device, dtype=self.lower.dtype)

        if self.mode == "none":
            return scaled_coordinates

        return scaled_coordinates * (self.upper - self.lower + self.epsilon) + self.lower

    def get_original_bounds(self):
        return self.lower.detach().clone(), self.upper.detach().clone()

    def get_summary(self):
        fitted = bool(self.is_fitted.item())

        return {
            "coord_scaling_mode": self.mode,
            "coord_domain_min": (
                self.lower.detach().cpu().reshape(-1).tolist() if fitted else None
            ),
            "coord_domain_max": (
                self.upper.detach().cpu().reshape(-1).tolist() if fitted else None
            ),
            "coord_out_of_domain_policy": self.out_of_domain_policy,
            "coord_clamp": self.clamp,
            "coord_scaling_epsilon": self.epsilon,
        }


class RBFCouplingNetwork(nn.Module):
    def __init__(self, number_of_kernels, output_scale=0.01, identity_initialization=False):
        super().__init__()

        self.number_of_kernels = int(number_of_kernels)

        # Center와 width는 buffer로 저장하고, 출력층의 weight/bias만 학습한다.
        centers = torch.linspace(0.0, 1.0, self.number_of_kernels).reshape(
            1, self.number_of_kernels
        )

        width = 1.0 / (self.number_of_kernels - 1) if self.number_of_kernels > 1 else 1.0

        widths = torch.full((1, self.number_of_kernels), width)

        self.register_buffer("centers", centers)
        self.register_buffer("widths", widths)

        self.output_layer = nn.Linear(self.number_of_kernels, 2)

        if identity_initialization:
            nn.init.zeros_(self.output_layer.weight)
            nn.init.zeros_(self.output_layer.bias)
        else:
            nn.init.uniform_(self.output_layer.weight, -output_scale, output_scale)
            nn.init.uniform_(self.output_layer.bias, -output_scale, output_scale)

    def forward(self, scalar_coordinate):
        standardized_distance = (scalar_coordinate - self.centers) / self.widths
        rbf_values = torch.exp(-0.5 * standardized_distance ** 2)

        output = self.output_layer(rbf_values)
        log_scale = output[:, 0:1]
        shift = output[:, 1:2]
        return log_scale, shift


def _dense_activation(name):
    name = str(name).lower().strip()

    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    if name == "elu":
        return nn.ELU()
    if name == "softplus":
        return nn.Softplus()

    raise ValueError("Unsupported dense activation.")


class DenseCouplingNetwork(nn.Module):
    def __init__(
        self,
        hidden_dimension=32,
        number_of_hidden_layers=2,
        activation="relu",
        output_scale=0.01,
        identity_initialization=False,
    ):
        super().__init__()

        layers = []
        input_dimension = 1

        for _ in range(int(number_of_hidden_layers)):
            linear_layer = nn.Linear(input_dimension, int(hidden_dimension))
            nn.init.xavier_uniform_(linear_layer.weight)
            nn.init.zeros_(linear_layer.bias)

            layers.append(linear_layer)
            layers.append(_dense_activation(activation))
            input_dimension = int(hidden_dimension)

        self.hidden_layers = nn.Sequential(*layers) if layers else nn.Identity()

        self.output_layer = nn.Linear(input_dimension, 2)

        if identity_initialization:
            nn.init.zeros_(self.output_layer.weight)
            nn.init.zeros_(self.output_layer.bias)
        else:
            nn.init.uniform_(self.output_layer.weight, -output_scale, output_scale)
            nn.init.uniform_(self.output_layer.bias, -output_scale, output_scale)

    def forward(self, scalar_coordinate):
        output = self.output_layer(self.hidden_layers(scalar_coordinate))

        log_scale = output[:, 0:1]
        shift = output[:, 1:2]
        return log_scale, shift


class AffineCouplingBlock(nn.Module):
    def __init__(
        self,
        network_type="rbf",
        number_of_rbf_kernels=3,
        rbf_output_scale=0.01,
        dense_hidden_dimension=32,
        dense_hidden_layers=2,
        dense_activation="relu",
        dense_output_scale=0.01,
        identity_initialization=False,
    ):
        super().__init__()

        self.network_type = str(network_type).lower().strip()

        if self.network_type == "rbf":
            def build_network():
                return RBFCouplingNetwork(
                    number_of_kernels=number_of_rbf_kernels,
                    output_scale=rbf_output_scale,
                    identity_initialization=identity_initialization,
                )
        elif self.network_type == "dense":
            def build_network():
                return DenseCouplingNetwork(
                    hidden_dimension=dense_hidden_dimension,
                    number_of_hidden_layers=dense_hidden_layers,
                    activation=dense_activation,
                    output_scale=dense_output_scale,
                    identity_initialization=identity_initialization,
                )
        else:
            raise ValueError("network_type must be 'rbf' or 'dense'.")

        # 같은 설정으로 서로 독립적인 두 network를 생성한다.
        self.first_network = build_network()
        self.second_network = build_network()

    def forward(self, coordinates):
        x_coordinate = coordinates[:, 0:1]
        y_coordinate = coordinates[:, 1:2]

        # y를 조건으로 x를 변환한 뒤, 변환된 x를 조건으로 y를 변환한다.
        first_log_scale, first_shift = self.first_network(y_coordinate)
        transformed_x = x_coordinate * torch.exp(first_log_scale) + first_shift

        second_log_scale, second_shift = self.second_network(transformed_x)
        transformed_y = y_coordinate * torch.exp(second_log_scale) + second_shift

        return torch.cat([transformed_x, transformed_y], dim=1)


class SpatialWarper(nn.Module):
    def __init__(
        self,
        number_of_blocks=2,
        network_type="rbf",
        number_of_rbf_kernels=3,
        rbf_output_scale=0.01,
        dense_hidden_dimension=32,
        dense_hidden_layers=2,
        dense_activation="relu",
        dense_output_scale=0.01,
        identity_initialization=False,
        rescale_between_blocks=False,
        rescale_final_output=False,
        rescale_source="train",
        reference_grid_size=31,
        detach_rescale_bounds=False,
        rescale_epsilon=1e-8,
    ):
        super().__init__()

        self.number_of_blocks = int(number_of_blocks)
        self.network_type = str(network_type).lower().strip()
        self.number_of_rbf_kernels = number_of_rbf_kernels
        self.dense_hidden_dimension = dense_hidden_dimension
        self.rescale_between_blocks = bool(rescale_between_blocks)
        self.rescale_final_output = bool(rescale_final_output)
        self.rescale_source = str(rescale_source).lower().strip()
        self.detach_rescale_bounds = bool(detach_rescale_bounds)
        self.rescale_epsilon = float(rescale_epsilon)

        if isinstance(number_of_rbf_kernels, int):
            rbf_kernels_per_block = [int(number_of_rbf_kernels)] * self.number_of_blocks
        else:
            rbf_kernels_per_block = [int(value) for value in number_of_rbf_kernels]

        if isinstance(dense_hidden_dimension, int):
            dense_dimension_per_block = [int(dense_hidden_dimension)] * self.number_of_blocks
        else:
            dense_dimension_per_block = [int(value) for value in dense_hidden_dimension]

        self.blocks = nn.ModuleList([
            AffineCouplingBlock(
                network_type=self.network_type,
                number_of_rbf_kernels=rbf_kernels_per_block[index],
                rbf_output_scale=rbf_output_scale,
                dense_hidden_dimension=dense_dimension_per_block[index],
                dense_hidden_layers=dense_hidden_layers,
                dense_activation=dense_activation,
                dense_output_scale=dense_output_scale,
                identity_initialization=identity_initialization,
            )
            for index in range(self.number_of_blocks)
        ])

        axis = torch.linspace(0.0, 1.0, int(reference_grid_size))
        x_grid, y_grid = torch.meshgrid(axis, axis, indexing="xy")
        reference_grid = torch.stack([x_grid.reshape(-1), y_grid.reshape(-1)], dim=1)

        self.register_buffer("reference_grid", reference_grid)
        self.register_buffer("saved_lower_bounds", torch.zeros(self.number_of_blocks, 2))
        self.register_buffer("saved_upper_bounds", torch.ones(self.number_of_blocks, 2))
        self.register_buffer(
            "saved_rescale_flags",
            torch.zeros(self.number_of_blocks, dtype=torch.bool),
        )
        self.register_buffer("saved_bounds_ready", torch.tensor(False, dtype=torch.bool))

        self.identity_initialization = bool(identity_initialization)
        self.rbf_output_scale = float(rbf_output_scale)
        self.dense_hidden_layers = int(dense_hidden_layers)
        self.dense_activation = str(dense_activation)
        self.dense_output_scale = float(dense_output_scale)
        self.reference_grid_size = int(reference_grid_size)

    def _should_rescale(self, block_index):
        is_last_block = block_index == self.number_of_blocks - 1

        return (
            (self.rescale_between_blocks and not is_last_block)
            or (self.rescale_final_output and is_last_block)
        )

    def _compute_bounds(self, values):
        lower = torch.amin(values, dim=0, keepdim=True)
        upper = torch.amax(values, dim=0, keepdim=True)

        if self.detach_rescale_bounds:
            lower = lower.detach()
            upper = upper.detach()

        span = torch.clamp(upper - lower, min=self.rescale_epsilon)

        return lower, upper, span

    def _saved_bounds(self, block_index, coordinates):
        lower = self.saved_lower_bounds[block_index:block_index + 1].to(
            device=coordinates.device,
            dtype=coordinates.dtype,
        )

        upper = self.saved_upper_bounds[block_index:block_index + 1].to(
            device=coordinates.device,
            dtype=coordinates.dtype,
        )

        span = torch.clamp(upper - lower, min=self.rescale_epsilon)

        return lower, upper, span

    def forward(
        self,
        base_coordinates,
        use_saved_bounds=False,
        return_diagnostics=False,
        return_bounds=False,
    ):
        check_coordinates(base_coordinates, "base_coordinates")

        transformed = base_coordinates
        reference = None

        if self.rescale_source == "reference_grid" and not use_saved_bounds:
            reference = self.reference_grid.to(device=transformed.device, dtype=transformed.dtype)

        diagnostics = []
        lower_bounds = []
        upper_bounds = []
        rescale_flags = []

        for block_index, block in enumerate(self.blocks):
            # 1. 현재 block의 좌표 변환.
            raw_output = block(transformed)
            raw_reference = block(reference) if reference is not None else None
            rescale_output = self._should_rescale(block_index)

            # 2. 경계를 선택하고, 설정에 따라 rescaling을 적용한다.
            # 예측 시에는 학습에서 저장한 경계를 재사용한다.
            if rescale_output:
                if use_saved_bounds:
                    lower, upper, span = self._saved_bounds(block_index, raw_output)
                else:
                    bound_values = raw_output if self.rescale_source == "train" else raw_reference
                    lower, upper, span = self._compute_bounds(bound_values)

                next_output = (raw_output - lower) / span
                next_reference = (
                    (raw_reference - lower) / span if raw_reference is not None else None
                )
            else:
                bound_values = raw_output if raw_reference is None else raw_reference
                lower, upper, span = self._compute_bounds(bound_values)
                next_output = raw_output
                next_reference = raw_reference

            lower_bounds.append(lower.reshape(2))
            upper_bounds.append(upper.reshape(2))
            rescale_flags.append(rescale_output)

            # 3. 요청된 경우 block별 진단 정보를 기록한다.
            if return_diagnostics:
                diagnostics.append({
                    "block": block_index + 1,
                    "rescaled": bool(rescale_output),
                    "bounds_source": self.rescale_source,
                    "used_fitted_training_bounds": bool(use_saved_bounds and rescale_output),
                    "data_raw_min": torch.amin(raw_output, dim=0).detach().cpu().tolist(),
                    "data_raw_max": torch.amax(raw_output, dim=0).detach().cpu().tolist(),
                    "rescale_lower": lower.detach().cpu().reshape(-1).tolist(),
                    "rescale_upper": upper.detach().cpu().reshape(-1).tolist(),
                    "data_output_min": torch.amin(next_output, dim=0).detach().cpu().tolist(),
                    "data_output_max": torch.amax(next_output, dim=0).detach().cpu().tolist(),
                    "fraction_below_zero": float((next_output < 0.0).float().mean().detach().cpu()),
                    "fraction_above_one": float((next_output > 1.0).float().mean().detach().cpu()),
                    "minimum_rescale_span": float(torch.amin(span).detach().cpu()),
                })

            # 4. 다음 block으로 전달한다.
            transformed = next_output
            reference = next_reference

        # 기본 반환은 좌표 tensor이며, 옵션에 따라 진단/경계를 tuple에 추가한다.
        output = [transformed]

        if return_diagnostics:
            output.append(diagnostics)

        if return_bounds:
            output.append({
                "lower": torch.stack(lower_bounds),
                "upper": torch.stack(upper_bounds),
                "rescaled": torch.tensor(
                    rescale_flags, device=transformed.device, dtype=torch.bool
                ),
            })

        return output[0] if len(output) == 1 else tuple(output)

    @torch.no_grad()
    def save_training_bounds(self, training_base_coordinates):
        _, bounds = self.forward(
            training_base_coordinates,
            use_saved_bounds=False,
            return_bounds=True,
        )

        self.saved_lower_bounds.copy_(bounds["lower"].to(self.saved_lower_bounds))
        self.saved_upper_bounds.copy_(bounds["upper"].to(self.saved_upper_bounds))
        self.saved_rescale_flags.copy_(bounds["rescaled"].to(self.saved_rescale_flags.device))
        self.saved_bounds_ready.fill_(True)
        return self

    @torch.no_grad()
    def diagnostics(self, base_coordinates):
        _, diagnostics = self.forward(
            base_coordinates,
            use_saved_bounds=bool(self.saved_bounds_ready.item()),
            return_diagnostics=True,
        )
        return diagnostics

    def get_summary(self):
        saved_bounds = None

        if bool(self.saved_bounds_ready.item()):
            saved_bounds = [
                {
                    "block": index + 1,
                    "rescaled": bool(self.saved_rescale_flags[index].item()),
                    "lower": self.saved_lower_bounds[index].detach().cpu().tolist(),
                    "upper": self.saved_upper_bounds[index].detach().cpu().tolist(),
                }
                for index in range(self.number_of_blocks)
            ]

        return {
            "number_of_warping_blocks": self.number_of_blocks,
            "warp_network_type": self.network_type,
            "number_of_rbf_kernels": (
                self.number_of_rbf_kernels if self.network_type == "rbf" else None
            ),
            "dense_hidden_dimension": (
                self.dense_hidden_dimension if self.network_type == "dense" else None
            ),
            "identity_initialization": self.identity_initialization,
            "rescale_between_blocks": self.rescale_between_blocks,
            "rescale_final_output": self.rescale_final_output,
            "rescale_source": self.rescale_source,
            "detach_rescale_bounds": self.detach_rescale_bounds,
            "rescale_epsilon": self.rescale_epsilon,
            "saved_bounds_ready": bool(self.saved_bounds_ready.item()),
            "saved_rescale_bounds": saved_bounds,
        }
