# python 3.7
"""Contains the function to march rays (integration)."""

import torch
import torch.nn.functional as F

__all__ = ['Integrator']


class Integrator(torch.nn.Module):
    """Defines the class to help march rays, i.e. do integral along each ray.

    The ray marcher takes the raw output of the implicit representation
    (including colors(i.e. rgbs) and densities(i.e. sigmas)) and uses the
    volume rendering equation to produce composited colors and depths.
    """

    def __init__(self):
        super().__init__()

    def integration(self, rgbs, sigmas, depths, rendering_options):
        """Integrate the values along the ray.

        `N` denotes batch size.
        `R` denotes the number of rays, equals `H * W`.
        `K` denotes the number of points on each ray.

        Args:
            rgbs (torch.tensor): colors' value of each point in the fields, with
                shape [N, R, K, 3].
            sigmas (torch.tensor): densities' value of each point in the fields,
                with shape [N, R, K, 1].
            depths (torch.tensor): depths' value of each point in the fields,
                with shape [N, R, K, 1].
            rendering_options (dict): Additional keyword arguments of rendering
                option.

        Returns:
            A dictionary, containing
                - `composite_rgb`: camera radius w.r.t. the world coordinate
                    system, with shape [N, R, 3].
                - `composite_depth`: camera polar w.r.t. the world coordinate
                    system, with shape [N, R, 1].
                - `weights`: importance weights of each point in the field,
                    with shape [N, R, K, 1].
        """
        num_dims = rgbs.ndim
        assert num_dims == 4
        assert sigmas.ndim == num_dims and depths.ndim == num_dims

        N, R, K = rgbs.shape[:3]

        # Get deltas for rendering.
        deltas = depths[:, :, 1:] - depths[:, :, :-1]
        if rendering_options.get('use_max_depth', False):
            max_depth = rendering_options.get('max_depth', None)
            if max_depth is not None:
                delta_inf = max_depth - deltas[:, :, -1:]
            else:
                delta_inf = 1e10 * torch.ones_like(deltas[:, :, :1])
            deltas = torch.cat([deltas, delta_inf], -2)
        if rendering_options.get('no_dist', False):
            deltas[:] = 1

        use_mid_point = rendering_options.get('use_mid_point', True)
        if use_mid_point:
            rgbs = (rgbs[:, :, :-1] + rgbs[:, :, 1:]) / 2
            sigmas = (sigmas[:, :, :-1] + sigmas[:, :, 1:]) / 2
            depths = (depths[:, :, :-1] + depths[:, :, 1:]) / 2

        clamp_mode = rendering_options.get('clamp_mode', 'mipnerf')
        if clamp_mode == 'softplus':
            sigmas = F.softplus(sigmas)
        elif clamp_mode == 'relu':
            sigmas = F.relu(sigmas)
        elif clamp_mode == 'mipnerf':
            sigmas = F.softplus(sigmas - 1)
        else:
            raise ValueError(f'Invalid clamping mode: `{clamp_mode}`!\n')

        alphas = 1 - torch.exp(- deltas * sigmas)
        alphas_shifted = torch.cat(
            [torch.ones_like(alphas[:, :, :1]), 1 - alphas + 1e-10], -2)
        weights = alphas * torch.cumprod(alphas_shifted, -2)[:, :, :-1]
        weights_sum = weights.sum(2)
        if rendering_options.get('last_back', False):
            weights[:, :, -1] =  weights[:, :, -1] + (1 - weights_sum)

        composite_rgb = torch.sum(weights * rgbs, -2)
        composite_depth = torch.sum(weights * depths, -2)

        if rendering_options.get('normalize_rgb', False):
            composite_rgb = composite_rgb / weights_sum
        if rendering_options.get('normalize_depth', True):
            composite_depth = composite_depth / weights_sum
        if rendering_options.get('clip_depth', True):
            composite_depth = torch.nan_to_num(composite_depth, float('inf'))
            composite_depth = torch.clip(composite_depth, torch.min(depths),
                                        torch.max(depths))

        if rendering_options.get('white_back', False):
            composite_rgb = composite_rgb + 1 - weights_sum

        composite_rgb = composite_rgb * 2 - 1   # Scale to (-1, 1)

        results = {
            'composite_rgb': composite_rgb,
            'composite_depth': composite_depth,
            'weights': weights
        }

        return results

    def forward(self, rgbs, sigmas, depths, rendering_options):
        results = self.integration(rgbs, sigmas, depths, rendering_options)
        return results
