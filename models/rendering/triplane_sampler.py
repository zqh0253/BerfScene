"""Contain the functions to sample point features from the triplane
   representation."""

import torch

__all__ = ['TriplaneSampler']


class TriplaneSampler(torch.nn.Module):
    """Defines the class to help sample point features from the triplane
       representation.

    Basically, this class implements the following functions for sampling point
    features (rgb && sigma) from the triplane representation:

    1. `generate_planes()`.
    2. `project_onto_planes()`.
    3. `sample_from_planes()`.
    4. `sample_from_3dgrid()`.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def generate_planes():
        """
        Defines planes by the three vectors that form the "axes" of the
        plane. Should work with arbitrary number of planes and planes of
        arbitrary orientation.
        """
        return torch.tensor([[[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]],
                                [[1, 0, 0],
                                [0, 0, 1],
                                [0, 1, 0]],
                                [[0, 0, 1],
                                [1, 0, 0],
                                [0, 1, 0]]], dtype=torch.float32)

    @staticmethod
    def project_onto_planes(planes, coordinates):
        """
        Does a projection of a 3D point onto a batch of 2D planes,
        returning 2D plane coordinates.

        Args:
            planes: Plane axes of shape (n_planes, 3, 3)
            coordinates: Coordinates of shape (N, M, 3)

        Returns:
            projections: Projections of shape (N*n_planes, M, 2)
        """
        N, M, C = coordinates.shape
        n_planes, _, _ = planes.shape
        coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1,
                                                      -1).reshape(
                                                          N * n_planes, M, 3)
        inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(
            N, -1, -1, -1).reshape(N * n_planes, 3, 3)
        projections = torch.bmm(coordinates, inv_planes)
        return projections[..., :2]

    @staticmethod
    def sample_from_planes(plane_axes,
                           plane_features,
                           coordinates,
                           mode='bilinear',
                           padding_mode='zeros',
                           box_warp=None):
        assert padding_mode == 'zeros'
        N, n_planes, C, H, W = plane_features.shape
        _, M, _ = coordinates.shape
        plane_features = plane_features.view(N * n_planes, C, H, W)

        coordinates = (2 / box_warp) * coordinates

        projected_coordinates = TriplaneSampler.project_onto_planes(
            plane_axes, coordinates).unsqueeze(1)
        output_features = torch.nn.functional.grid_sample(
            plane_features,
            projected_coordinates.float(),
            mode=mode,
            padding_mode=padding_mode,
            align_corners=False).permute(0, 3, 2,
                                         1).reshape(N, n_planes, M, C)
        return output_features

    @staticmethod
    def sample_from_3dgrid(grid, coordinates):
        """
        Expects coordinates in shape (batch_size, num_points_per_batch, 3)
        Expects grid in shape (1, channels, H, W, D)
        (Also works if grid has batch size)
        Returns:
            Sampled features
            with shape: (batch_size, num_points_per_batch, feature_channels).
        """
        batch_size, n_coords, n_dims = coordinates.shape
        sampled_features = torch.nn.functional.grid_sample(
            grid.expand(batch_size, -1, -1, -1, -1),
            coordinates.reshape(batch_size, 1, 1, -1, n_dims),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        N, C, H, W, D = sampled_features.shape
        sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(
            N, H * W * D, C)
        return sampled_features