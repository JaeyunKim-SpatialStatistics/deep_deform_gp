# 적합된 GP의 예측 평균과 잠재장/관측값의 예측분산을 계산한다.

import torch

from .numerical_utils import check_coordinates, check_values, safe_cholesky


class GPPredictor:
    def __init__(self, fitted_model, training_coordinates, training_values):
        if not fitted_model.is_fitted:
            raise RuntimeError("The model must be fitted before prediction.")

        check_coordinates(training_coordinates, "training_coordinates")
        check_values(training_values, "training_values")

        self.model = fitted_model
        self.training_coordinates = training_coordinates.to(
            device=fitted_model.device,
            dtype=fitted_model.dtype,
        )
        self.training_values = training_values.to(
            device=fitted_model.device,
            dtype=fitted_model.dtype,
        )

    @torch.no_grad()
    def predict(self, prediction_coordinates, return_variance=True):
        check_coordinates(prediction_coordinates, "prediction_coordinates")

        model = self.model
        model.eval()

        prediction_coordinates = prediction_coordinates.to(device=model.device, dtype=model.dtype)

        # 학습과 예측 좌표 모두에 동일한 scaler와 저장된 warping 경계를 적용한다.
        training_base = model.base_coordinates(self.training_coordinates, check_domain=False)
        prediction_base = model.base_coordinates(prediction_coordinates, check_domain=True)

        training_transformed = model.transform_base_coordinates(
            training_base,
            use_saved_bounds=True,
        )
        prediction_transformed = model.transform_base_coordinates(
            prediction_base,
            use_saved_bounds=True,
        )

        if model.vecchia is None:
            result = self._predict_exact(
                prediction_coordinates,
                training_transformed,
                prediction_transformed,
                return_variance,
            )
        else:
            result = self._predict_vecchia(
                prediction_coordinates,
                training_base,
                prediction_base,
                training_transformed,
                prediction_transformed,
                return_variance,
            )

        if model.uses_warping:
            result["warped_coords"] = prediction_transformed.detach().cpu()

        return result

    def _predict_exact(
        self,
        prediction_coordinates,
        training_transformed,
        prediction_transformed,
        return_variance,
    ):
        model = self.model
        covariance_model = model.covariance_model

        training_covariance = covariance_model.covariance_matrix(
            training_transformed,
            training_transformed,
            self.training_coordinates,
            self.training_coordinates,
        )

        training_covariance = training_covariance + (
            model.tau2 + 1e-6
        ) * torch.eye(self.training_values.shape[0], device=model.device, dtype=model.dtype)

        factor, _ = safe_cholesky(training_covariance)

        centered_training_values = (self.training_values - model.mean).unsqueeze(1)

        solved_values = torch.cholesky_solve(centered_training_values, factor)

        prediction_training_covariance = covariance_model.covariance_matrix(
            prediction_transformed,
            training_transformed,
            prediction_coordinates,
            self.training_coordinates,
        )

        prediction_mean = model.mean + (prediction_training_covariance @ solved_values).squeeze(1)

        result = {"y_pred": prediction_mean.detach().cpu()}

        if not return_variance:
            return result

        marginal_variance = covariance_model.marginal_variance(prediction_coordinates)

        solved_cross_covariance = torch.cholesky_solve(prediction_training_covariance.T, factor)

        variance_reduction = torch.sum(
            prediction_training_covariance * solved_cross_covariance.T,
            dim=1,
        )

        latent_variance = torch.clamp(marginal_variance - variance_reduction, min=1e-12)
        # var_f: 잠재장 예측분산, var_y: 관측 오차를 포함한 예측분산.
        observation_variance = latent_variance + model.tau2

        result.update(self._variance_output(latent_variance, observation_variance))

        return result

    def _predict_vecchia(
        self,
        prediction_coordinates,
        training_base,
        prediction_base,
        training_transformed,
        prediction_transformed,
        return_variance,
    ):
        model = self.model
        vecchia = model.vecchia
        covariance_model = model.covariance_model

        # 예측 block마다 선택한 학습 이웃만으로 조건부 예측을 계산한다.
        block_indices, neighbor_indices = vecchia.build_prediction_structure(
            training_base,
            prediction_base,
            training_transformed,
            prediction_transformed,
        )

        number_of_predictions = prediction_coordinates.shape[0]
        prediction_mean = torch.empty(number_of_predictions, device=model.device, dtype=model.dtype)

        latent_variance = torch.empty_like(prediction_mean) if return_variance else None
        observation_variance = torch.empty_like(prediction_mean) if return_variance else None

        if covariance_model.covariance_model == "matern_general":
            for start in range(0, len(block_indices), vecchia.bessel_batch_blocks):
                end = min(start + vecchia.bessel_batch_blocks, len(block_indices))

                covariance_items = vecchia.prediction_covariance_items(
                    covariance_model,
                    training_transformed,
                    prediction_transformed,
                    self.training_coordinates,
                    prediction_coordinates,
                    block_indices[start:end],
                    neighbor_indices[start:end],
                )

                for item in covariance_items:
                    self._fill_prediction_block(
                        item["block_index"],
                        item["neighbor_index"],
                        item["neighbor_covariance"],
                        item["block_neighbor_covariance"],
                        prediction_coordinates,
                        prediction_mean,
                        latent_variance,
                        observation_variance,
                    )
        else:
            for (block_index, neighbor_index) in zip(block_indices, neighbor_indices):
                if neighbor_index.numel() == 0:
                    neighbor_covariance = None
                    block_neighbor_covariance = None
                else:
                    neighbor_coordinates = training_transformed[neighbor_index]
                    neighbor_original = self.training_coordinates[neighbor_index]
                    block_coordinates = prediction_transformed[block_index]
                    block_original = prediction_coordinates[block_index]

                    neighbor_covariance = covariance_model.covariance_matrix(
                        neighbor_coordinates,
                        neighbor_coordinates,
                        neighbor_original,
                        neighbor_original,
                    )

                    block_neighbor_covariance = covariance_model.covariance_matrix(
                        block_coordinates,
                        neighbor_coordinates,
                        block_original,
                        neighbor_original,
                    )

                self._fill_prediction_block(
                    block_index,
                    neighbor_index,
                    neighbor_covariance,
                    block_neighbor_covariance,
                    prediction_coordinates,
                    prediction_mean,
                    latent_variance,
                    observation_variance,
                )

        result = {"y_pred": prediction_mean.detach().cpu()}

        if return_variance:
            result.update(self._variance_output(latent_variance, observation_variance))

        return result

    def _fill_prediction_block(
        self,
        block_index,
        neighbor_index,
        neighbor_covariance,
        block_neighbor_covariance,
        prediction_coordinates,
        prediction_mean,
        latent_variance,
        observation_variance,
    ):
        model = self.model

        if neighbor_index.numel() == 0:
            prediction_mean[block_index] = model.mean

            if latent_variance is not None:
                block_variance = model.covariance_model.marginal_variance(
                    prediction_coordinates[block_index]
                )
                latent_variance[block_index] = block_variance
                observation_variance[block_index] = block_variance + model.tau2
            return

        neighbor_values = self.training_values[neighbor_index]
        neighbor_covariance = neighbor_covariance + (
            model.tau2 + 1e-6
        ) * torch.eye(neighbor_values.shape[0], device=model.device, dtype=model.dtype)

        factor, _ = safe_cholesky(neighbor_covariance)

        centered_neighbors = (neighbor_values - model.mean).unsqueeze(1)

        solved_neighbors = torch.cholesky_solve(centered_neighbors, factor)

        prediction_mean[block_index] = model.mean + (
            block_neighbor_covariance @ solved_neighbors
        ).squeeze(1)

        if latent_variance is None:
            return

        marginal_variance = model.covariance_model.marginal_variance(
            prediction_coordinates[block_index]
        )

        solved_cross_covariance = torch.cholesky_solve(block_neighbor_covariance.T, factor)

        variance_reduction = torch.sum(block_neighbor_covariance * solved_cross_covariance.T, dim=1)

        block_latent_variance = torch.clamp(marginal_variance - variance_reduction, min=1e-12)

        latent_variance[block_index] = block_latent_variance
        observation_variance[block_index] = block_latent_variance + model.tau2

    def _variance_output(self, latent_variance, observation_variance):
        latent_variance = latent_variance.detach().cpu()
        observation_variance = observation_variance.detach().cpu()

        return {
            "var_f": latent_variance,
            "var_y": observation_variance,
            "sd_f": torch.sqrt(torch.clamp(latent_variance, min=0.0)),
            "sd_y": torch.sqrt(torch.clamp(observation_variance, min=0.0)),
        }
