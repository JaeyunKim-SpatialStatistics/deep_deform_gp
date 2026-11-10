# GP 모수와 warping 모수를 음의 로그가능도로 함께 학습한다.

import time

import torch
import torch.optim as optim

from .numerical_utils import (
    check_coordinates,
    check_values,
    clone_state_dict_to_cpu,
)


class GPFitter:
    def __init__(
        self,
        number_of_epochs=500,
        gp_learning_rate=0.001,
        warping_learning_rate=0.001,
        early_stopping=True,
        patience=100,
        minimum_improvement=1e-6,
        maximum_gradient_norm=5.0,
        print_every=100,
    ):
        self.number_of_epochs = int(number_of_epochs)
        self.gp_learning_rate = float(gp_learning_rate)
        self.warping_learning_rate = float(warping_learning_rate)
        self.early_stopping = bool(early_stopping)
        self.patience = int(patience)
        self.minimum_improvement = float(minimum_improvement)
        self.maximum_gradient_norm = maximum_gradient_norm
        self.print_every = int(print_every)

    def _create_optimizer(self, model):
        parameter_groups = [
            {
                "params": model.gp_parameters(),
                "lr": self.gp_learning_rate,
            }
        ]

        warping_parameters = model.warping_parameters()

        if warping_parameters:
            parameter_groups.append({
                "params": warping_parameters,
                "lr": self.warping_learning_rate,
            })

        return optim.Adam(parameter_groups)

    def fit(self, model, training_coordinates, training_values, verbose=True):
        check_coordinates(training_coordinates, "training_coordinates")
        check_values(training_values, "training_values")

        training_coordinates = training_coordinates.to(device=model.device, dtype=model.dtype)
        training_values = training_values.to(device=model.device, dtype=model.dtype)

        start_time = time.time()

        # 1. 좌표 경계와 GP 모수를 초기화한다.
        model.coordinate_scaler.fit(training_coordinates)
        model.initialize_parameters(training_coordinates, training_values)

        base_coordinates = model.base_coordinates(training_coordinates, check_domain=False)
        transformed_coordinates = model.transform_base_coordinates(
            base_coordinates,
            use_saved_bounds=False,
        )

        # 2. Vecchia를 사용하는 경우 block과 이웃 집합을 만든다.
        if model.vecchia is not None:
            model.vecchia.build_training_structure(base_coordinates, transformed_coordinates)

        optimizer = self._create_optimizer(model)

        best_loss = float("inf")
        best_epoch = None
        best_state = None
        epochs_without_improvement = 0
        loss_history = []

        # 3. 기존 순서대로 가능도 평가, 역전파, Adam 업데이트를 반복한다.
        model.train()

        for epoch in range(1, self.number_of_epochs + 1):
            if (
                model.vecchia is not None
                and model.vecchia.neighbor_space == "warped"
                and model.vecchia.update_neighbors_every > 0
                and epoch > 1
                and (epoch - 1) % model.vecchia.update_neighbors_every == 0
            ):
                with torch.no_grad():
                    transformed_coordinates = model.transform_base_coordinates(
                        base_coordinates,
                        use_saved_bounds=False,
                    )
                    model.vecchia.update_training_neighbors(
                        base_coordinates,
                        transformed_coordinates,
                    )
                epochs_without_improvement = 0

            optimizer.zero_grad(set_to_none=True)

            log_likelihood = model.log_likelihood(training_coordinates, training_values)
            loss = -log_likelihood

            if not torch.isfinite(loss):
                break

            loss.backward()

            if self.maximum_gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=self.maximum_gradient_norm,
                )

            optimizer.step()

            # 원본 순서 유지: loss는 업데이트 전 값, best_state는 업데이트 후 모수다.
            current_loss = float(loss.detach().cpu().item())
            loss_history.append(current_loss)

            if current_loss < best_loss - self.minimum_improvement:
                best_loss = current_loss
                best_epoch = epoch
                best_state = clone_state_dict_to_cpu(model)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if verbose and (
                epoch == 1 or epoch % self.print_every == 0 or epoch == self.number_of_epochs
            ):
                self._print_progress(model, epoch, current_loss, best_loss)

            if self.early_stopping and epochs_without_improvement >= self.patience:
                model.stopped_early = True
                break

        if best_state is None:
            raise RuntimeError("No finite likelihood was obtained during fitting.")

        # 4. 선택된 모수를 복원하고 예측에 사용할 warping 경계를 저장한다.
        model.load_state_dict(best_state)
        model.to_device(model.device)
        model.save_warping_bounds(training_coordinates)

        model.is_fitted = True
        model.number_of_training_points = int(training_coordinates.shape[0])
        model.best_negative_log_likelihood = best_loss
        model.best_epoch = best_epoch
        model.training_time = time.time() - start_time

        if verbose:
            print(
                f"Best epoch: {best_epoch} | "
                f"best -loglik: {best_loss:.4f} | "
                f"time: {model.training_time:.2f} sec"
            )

        return {
            "loss_history": loss_history,
            "best_epoch": best_epoch,
            "best_negative_log_likelihood": best_loss,
            "training_time_sec": model.training_time,
        }

    def _print_progress(self, model, epoch, loss, best_loss):
        covariance = model.covariance_model

        message = (
            f"Epoch {epoch:04d} | "
            f"-loglik={loss:.4f} | "
            f"best={best_loss:.4f} | "
            f"a={float(covariance.a.detach()):.4f} | "
            f"sigma2={float(covariance.sigma2.detach()):.4f} | "
            f"tau2={float(model.tau2.detach()):.6f}"
        )

        if covariance.covariance_model == "matern_general":
            message += f" | nu={float(covariance.nu.detach()):.4f}"

        print(message)
