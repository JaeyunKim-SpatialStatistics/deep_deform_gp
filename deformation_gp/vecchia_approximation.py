# K-means block과 이전 관측점의 이웃 집합을 이용한 Vecchia 근사.

import numpy as np
import torch
from sklearn.cluster import KMeans

from .numerical_utils import (
    multivariate_normal_log_likelihood,
    safe_cholesky,
    to_device_index_list,
)


class VecchiaApproximation:
    def __init__(
        self,
        number_of_blocks=100,
        number_of_neighbors=30,
        ordering="random",
        neighbor_space="base",
        update_neighbors_every=0,
        bessel_batch_blocks=25,
        seed=1,
    ):
        self.number_of_blocks = int(number_of_blocks)
        self.number_of_neighbors = int(number_of_neighbors)
        self.ordering = str(ordering).lower().strip()
        self.neighbor_space = str(neighbor_space).lower().strip()
        self.update_neighbors_every = int(update_neighbors_every)
        self.bessel_batch_blocks = max(1, int(bessel_batch_blocks))
        self.seed = int(seed)

        self.training_block_indices = None
        self.training_neighbor_indices = None

    def _build_kmeans_blocks(self, coordinates):
        number_of_observations = coordinates.shape[0]
        number_of_blocks = max(1, min(self.number_of_blocks, number_of_observations))

        if number_of_blocks == 1:
            labels = np.zeros(number_of_observations, dtype=int)
        else:
            kmeans = KMeans(n_clusters=number_of_blocks, random_state=self.seed, n_init=10)
            labels = kmeans.fit_predict(coordinates)

        blocks = []

        for block_number in range(number_of_blocks):
            block_index = np.where(labels == block_number)[0]
            if block_index.size > 0:
                blocks.append(block_index.astype(int))

        centers = np.stack([coordinates[block_index].mean(axis=0) for block_index in blocks])

        return blocks, centers

    def _order_block_indices(self, centers):
        if self.ordering == "x":
            return np.argsort(centers[:, 0])
        if self.ordering == "y":
            return np.argsort(centers[:, 1])

        random_generator = np.random.default_rng(self.seed)
        return random_generator.permutation(centers.shape[0])

    def _nearest_previous_points(self, current_block, previous_points, coordinates):
        if previous_points.size == 0 or self.number_of_neighbors <= 0:
            return np.array([], dtype=int)

        block_center = coordinates[current_block].mean(axis=0, keepdims=True)
        previous_coordinates = coordinates[previous_points]

        squared_distance = np.sum((previous_coordinates - block_center) ** 2, axis=1)

        number_to_keep = min(self.number_of_neighbors, previous_points.size)

        nearest = np.argsort(squared_distance)[:number_to_keep]
        return previous_points[nearest].astype(int)

    def _training_neighbor_coordinates(self, base_coordinates, transformed_coordinates):
        if self.neighbor_space == "warped":
            return transformed_coordinates
        return base_coordinates

    def build_training_structure(self, base_coordinates, transformed_coordinates):
        neighbor_coordinates = self._training_neighbor_coordinates(
            base_coordinates,
            transformed_coordinates,
        )

        coordinates_numpy = neighbor_coordinates.detach().cpu().numpy()
        blocks, centers = self._build_kmeans_blocks(coordinates_numpy)
        order = self._order_block_indices(centers)
        # 각 block은 현재 순서에서 앞서 나온 관측점에만 조건부로 의존한다.
        ordered_blocks = [blocks[index] for index in order]

        neighbor_indices = []
        previous_points = np.array([], dtype=int)

        for block_number, block_index in enumerate(ordered_blocks):
            if block_number == 0:
                neighbors = np.array([], dtype=int)
            else:
                neighbors = self._nearest_previous_points(
                    current_block=block_index,
                    previous_points=previous_points,
                    coordinates=coordinates_numpy,
                )

            neighbor_indices.append(neighbors)
            previous_points = np.concatenate([previous_points, block_index]).astype(int)

        device = base_coordinates.device
        self.training_block_indices = to_device_index_list(ordered_blocks, device)
        self.training_neighbor_indices = to_device_index_list(neighbor_indices, device)

    def update_training_neighbors(self, base_coordinates, transformed_coordinates):
        if self.training_block_indices is None:
            return

        neighbor_coordinates = self._training_neighbor_coordinates(
            base_coordinates,
            transformed_coordinates,
        )

        # Block 구성과 순서는 유지하고 이웃만 갱신한다.
        new_neighbors = []
        previous_points = torch.empty(0, dtype=torch.long, device=neighbor_coordinates.device)

        for block_number, block_index in enumerate(self.training_block_indices):
            if block_number == 0 or previous_points.numel() == 0:
                neighbors = torch.empty(0, dtype=torch.long, device=neighbor_coordinates.device)
            else:
                block_center = neighbor_coordinates[block_index].mean(dim=0, keepdim=True)
                candidate_coordinates = neighbor_coordinates[previous_points]
                squared_distance = torch.sum((candidate_coordinates - block_center) ** 2, dim=1)

                number_to_keep = min(self.number_of_neighbors, int(previous_points.numel()))

                nearest = torch.topk(squared_distance, k=number_to_keep, largest=False).indices
                neighbors = previous_points[nearest]

            new_neighbors.append(neighbors)
            previous_points = torch.cat([previous_points, block_index])

        self.training_neighbor_indices = new_neighbors

    def build_prediction_structure(
        self,
        training_base_coordinates,
        prediction_base_coordinates,
        training_transformed_coordinates,
        prediction_transformed_coordinates,
    ):
        if self.neighbor_space == "warped":
            training_coordinates = training_transformed_coordinates
            prediction_coordinates = prediction_transformed_coordinates
        else:
            training_coordinates = training_base_coordinates
            prediction_coordinates = prediction_base_coordinates

        training_numpy = training_coordinates.detach().cpu().numpy()
        prediction_numpy = prediction_coordinates.detach().cpu().numpy()

        blocks, _ = self._build_kmeans_blocks(prediction_numpy)
        all_training_indices = np.arange(training_numpy.shape[0], dtype=int)
        neighbor_indices = []

        for block_index in blocks:
            block_center = prediction_numpy[block_index].mean(axis=0, keepdims=True)

            squared_distance = np.sum((training_numpy - block_center) ** 2, axis=1)

            number_to_keep = min(self.number_of_neighbors, training_numpy.shape[0])

            nearest = np.argsort(squared_distance)[:number_to_keep]
            neighbor_indices.append(all_training_indices[nearest].astype(int))

        device = training_base_coordinates.device
        return (
            to_device_index_list(blocks, device),
            to_device_index_list(neighbor_indices, device),
        )

    def log_likelihood(self, model, values, transformed_coordinates, original_coordinates):
        covariance_model = model.covariance_model

        if covariance_model.covariance_model == "matern_general":
            return self._batched_log_likelihood(
                model,
                values,
                transformed_coordinates,
                original_coordinates,
            )

        total_log_likelihood = values.new_tensor(0.0)

        for block_index, neighbor_index in zip(
            self.training_block_indices,
            self.training_neighbor_indices,
        ):
            block_coordinates = transformed_coordinates[block_index]
            block_original_coordinates = original_coordinates[block_index]
            block_values = values[block_index]

            block_covariance = covariance_model.covariance_matrix(
                block_coordinates,
                block_coordinates,
                block_original_coordinates,
                block_original_coordinates,
            )

            block_covariance = block_covariance + (
                model.tau2 + 1e-6
            ) * torch.eye(block_values.shape[0], device=values.device, dtype=values.dtype)

            if neighbor_index.numel() == 0:
                total_log_likelihood = (
                    total_log_likelihood
                    + multivariate_normal_log_likelihood(
                        block_values,
                        model.mean.expand_as(block_values),
                        block_covariance,
                    )
                )
                continue

            neighbor_coordinates = transformed_coordinates[neighbor_index]
            neighbor_original_coordinates = original_coordinates[neighbor_index]
            neighbor_values = values[neighbor_index]

            neighbor_covariance = covariance_model.covariance_matrix(
                neighbor_coordinates,
                neighbor_coordinates,
                neighbor_original_coordinates,
                neighbor_original_coordinates,
            )

            neighbor_covariance = neighbor_covariance + (
                model.tau2 + 1e-6
            ) * torch.eye(neighbor_values.shape[0], device=values.device, dtype=values.dtype)

            block_neighbor_covariance = covariance_model.covariance_matrix(
                block_coordinates,
                neighbor_coordinates,
                block_original_coordinates,
                neighbor_original_coordinates,
            )

            conditional_mean, conditional_covariance = self._conditional_distribution(
                model,
                block_covariance,
                neighbor_covariance,
                block_neighbor_covariance,
                neighbor_values,
            )

            total_log_likelihood = (
                total_log_likelihood
                + multivariate_normal_log_likelihood(
                    block_values,
                    conditional_mean,
                    conditional_covariance,
                )
            )

        return total_log_likelihood

    def _batched_log_likelihood(self, model, values, transformed_coordinates, original_coordinates):
        total_log_likelihood = values.new_tensor(0.0)

        for start in range(0, len(self.training_block_indices), self.bessel_batch_blocks):
            end = min(start + self.bessel_batch_blocks, len(self.training_block_indices))

            covariance_items = self.training_covariance_items(
                model.covariance_model,
                transformed_coordinates,
                original_coordinates,
                self.training_block_indices[start:end],
                self.training_neighbor_indices[start:end],
            )

            for item in covariance_items:
                block_index = item["block_index"]
                neighbor_index = item["neighbor_index"]
                block_values = values[block_index]

                block_covariance = item["block_covariance"] + (
                    model.tau2 + 1e-6
                ) * torch.eye(block_values.shape[0], device=values.device, dtype=values.dtype)

                if neighbor_index.numel() == 0:
                    total_log_likelihood = (
                        total_log_likelihood
                        + multivariate_normal_log_likelihood(
                            block_values,
                            model.mean.expand_as(block_values),
                            block_covariance,
                        )
                    )
                    continue

                neighbor_values = values[neighbor_index]
                neighbor_covariance = item["neighbor_covariance"] + (
                    model.tau2 + 1e-6
                ) * torch.eye(neighbor_values.shape[0], device=values.device, dtype=values.dtype)

                conditional_mean, conditional_covariance = self._conditional_distribution(
                    model,
                    block_covariance,
                    neighbor_covariance,
                    item["block_neighbor_covariance"],
                    neighbor_values,
                )

                total_log_likelihood = (
                    total_log_likelihood
                    + multivariate_normal_log_likelihood(
                        block_values,
                        conditional_mean,
                        conditional_covariance,
                    )
                )

        return total_log_likelihood

    def _conditional_distribution(
        self,
        model,
        block_covariance,
        neighbor_covariance,
        block_neighbor_covariance,
        neighbor_values,
    ):
        factor, _ = safe_cholesky(neighbor_covariance)

        centered_neighbors = (neighbor_values - model.mean).unsqueeze(1)

        solved_neighbors = torch.cholesky_solve(centered_neighbors, factor)

        conditional_mean = model.mean + (block_neighbor_covariance @ solved_neighbors).squeeze(1)

        solved_cross_covariance = torch.cholesky_solve(block_neighbor_covariance.T, factor)

        conditional_covariance = (
            block_covariance - block_neighbor_covariance @ solved_cross_covariance
        )

        return conditional_mean, conditional_covariance

    def training_covariance_items(
        self,
        covariance_model,
        transformed_coordinates,
        original_coordinates,
        block_indices,
        neighbor_indices,
    ):
        # 1. 거리 행렬을 모으고 복원에 필요한 shape와 원소 수를 기록한다.
        specifications = []
        all_distances = []
        all_log_variance = []

        for block_index, neighbor_index in zip(block_indices, neighbor_indices):
            block_coordinates = transformed_coordinates[block_index]
            block_distance = torch.cdist(block_coordinates, block_coordinates)

            specification = {
                "block_index": block_index,
                "neighbor_index": neighbor_index,
                "block_shape": block_distance.shape,
                "block_size": block_distance.numel(),
                "has_neighbors": bool(neighbor_index.numel() > 0),
            }

            all_distances.append(block_distance.reshape(-1))

            if covariance_model.use_spatial_variance:
                block_log_variance = covariance_model.spatial_log_variance(
                    original_coordinates[block_index]
                )
                all_log_variance.append(
                    (0.5 * (block_log_variance[:, None] + block_log_variance[None, :])).reshape(-1)
                )

            if neighbor_index.numel() > 0:
                neighbor_coordinates = transformed_coordinates[neighbor_index]
                neighbor_distance = torch.cdist(neighbor_coordinates, neighbor_coordinates)
                block_neighbor_distance = torch.cdist(block_coordinates, neighbor_coordinates)

                specification.update({
                    "neighbor_shape": neighbor_distance.shape,
                    "neighbor_size": neighbor_distance.numel(),
                    "cross_shape": block_neighbor_distance.shape,
                    "cross_size": block_neighbor_distance.numel(),
                })

                all_distances.append(neighbor_distance.reshape(-1))
                all_distances.append(block_neighbor_distance.reshape(-1))

                if covariance_model.use_spatial_variance:
                    neighbor_log_variance = covariance_model.spatial_log_variance(
                        original_coordinates[neighbor_index]
                    )

                    all_log_variance.append(
                        (
                            0.5 * (neighbor_log_variance[:, None] + neighbor_log_variance[None, :])
                        ).reshape(-1)
                    )

                    all_log_variance.append(
                        (
                            0.5 * (block_log_variance[:, None] + neighbor_log_variance[None, :])
                        ).reshape(-1)
                    )

            specifications.append(specification)

        # 2. 연결된 거리 벡터에 공분산 함수를 적용한다.
        all_distances = torch.cat(all_distances)
        covariance_values = (
            covariance_model.sigma2 * covariance_model.correlation_from_distance(all_distances)
        )

        if covariance_model.use_spatial_variance:
            covariance_values = covariance_values * torch.exp(torch.cat(all_log_variance))

        # 3. 기록한 크기에 맞춰 공분산 행렬들을 원래 순서로 복원한다.
        items = []
        position = 0

        for specification in specifications:
            block_end = position + specification["block_size"]
            block_covariance = covariance_values[position:block_end].reshape(
                specification["block_shape"]
            )
            position = block_end

            item = {
                "block_index": specification["block_index"],
                "neighbor_index": specification["neighbor_index"],
                "block_covariance": block_covariance,
                "neighbor_covariance": None,
                "block_neighbor_covariance": None,
            }

            if specification["has_neighbors"]:
                neighbor_end = position + specification["neighbor_size"]
                item["neighbor_covariance"] = covariance_values[position:neighbor_end].reshape(
                    specification["neighbor_shape"]
                )
                position = neighbor_end

                cross_end = position + specification["cross_size"]
                item["block_neighbor_covariance"] = covariance_values[position:cross_end].reshape(
                    specification["cross_shape"]
                )
                position = cross_end

            items.append(item)

        return items

    def prediction_covariance_items(
        self,
        covariance_model,
        training_transformed_coordinates,
        prediction_transformed_coordinates,
        training_original_coordinates,
        prediction_original_coordinates,
        block_indices,
        neighbor_indices,
    ):
        # 1. 거리 행렬을 모으고 복원에 필요한 shape와 원소 수를 기록한다.
        specifications = []
        all_distances = []
        all_log_variance = []

        for block_index, neighbor_index in zip(block_indices, neighbor_indices):
            specification = {
                "block_index": block_index,
                "neighbor_index": neighbor_index,
                "has_neighbors": bool(neighbor_index.numel() > 0),
            }

            if neighbor_index.numel() > 0:
                block_coordinates = prediction_transformed_coordinates[block_index]
                neighbor_coordinates = training_transformed_coordinates[neighbor_index]

                neighbor_distance = torch.cdist(neighbor_coordinates, neighbor_coordinates)
                block_neighbor_distance = torch.cdist(block_coordinates, neighbor_coordinates)

                specification.update({
                    "neighbor_shape": neighbor_distance.shape,
                    "neighbor_size": neighbor_distance.numel(),
                    "cross_shape": block_neighbor_distance.shape,
                    "cross_size": block_neighbor_distance.numel(),
                })

                all_distances.append(neighbor_distance.reshape(-1))
                all_distances.append(block_neighbor_distance.reshape(-1))

                if covariance_model.use_spatial_variance:
                    neighbor_log_variance = covariance_model.spatial_log_variance(
                        training_original_coordinates[neighbor_index]
                    )
                    block_log_variance = covariance_model.spatial_log_variance(
                        prediction_original_coordinates[block_index]
                    )

                    all_log_variance.append(
                        (
                            0.5 * (neighbor_log_variance[:, None] + neighbor_log_variance[None, :])
                        ).reshape(-1)
                    )
                    all_log_variance.append(
                        (
                            0.5 * (block_log_variance[:, None] + neighbor_log_variance[None, :])
                        ).reshape(-1)
                    )

            specifications.append(specification)

        if all_distances:
            # 2. 연결된 거리 벡터에 공분산 함수를 적용한다.
            all_distances = torch.cat(all_distances)
            covariance_values = (
                covariance_model.sigma2 * covariance_model.correlation_from_distance(all_distances)
            )

            if covariance_model.use_spatial_variance:
                covariance_values = covariance_values * torch.exp(torch.cat(all_log_variance))
        else:
            covariance_values = None

        # 3. 기록한 크기에 맞춰 공분산 행렬들을 원래 순서로 복원한다.
        items = []
        position = 0

        for specification in specifications:
            item = {
                "block_index": specification["block_index"],
                "neighbor_index": specification["neighbor_index"],
                "neighbor_covariance": None,
                "block_neighbor_covariance": None,
            }

            if specification["has_neighbors"]:
                neighbor_end = position + specification["neighbor_size"]
                item["neighbor_covariance"] = covariance_values[position:neighbor_end].reshape(
                    specification["neighbor_shape"]
                )
                position = neighbor_end

                cross_end = position + specification["cross_size"]
                item["block_neighbor_covariance"] = covariance_values[position:cross_end].reshape(
                    specification["cross_shape"]
                )
                position = cross_end

            items.append(item)

        return items

    def to(self, device):
        if self.training_block_indices is not None:
            self.training_block_indices = [
                index.to(device)
                for index in self.training_block_indices
            ]

        if self.training_neighbor_indices is not None:
            self.training_neighbor_indices = [
                index.to(device)
                for index in self.training_neighbor_indices
            ]

        return self

    def get_summary(self):
        return {
            "approximation": True,
            "number_of_vecchia_blocks": self.number_of_blocks,
            "number_of_neighbors": self.number_of_neighbors,
            "ordering": self.ordering,
            "vecchia_neighbor_space": self.neighbor_space,
            "update_neighbors_every": self.update_neighbors_every,
            "bessel_batch_blocks": self.bessel_batch_blocks,
        }
