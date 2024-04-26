# python3.8
"""Contains image renderer class."""

import torch
import torch.nn as nn
from .point_sampler import PointSampler
from .integrator import Integrator

__all__ = ['Renderer']


class Renderer(nn.Module):
    """Defines the class to render images.

    The renderer is a module that takes in latent codes and points, decides
    where to sample along each ray, and computes pixel colors/features using the
    volume rendering equation.

    Basically, the volume rendering pipiline consists of the following steps:

    1. Sample points in 3D Space.
    2. (Optional) Get the reference representation by injecting latent codes
       into the reference representation generator. Generally, the reference
       representation can be a feature volume (VolumenGAN), a triplane (EG3D) or
       others.
    3. Get the corresponding feature of each sampled point by the given feature
       extractor. Typically, the overall formulation is:
            feat = F(wp, points, options, ref_representation, post_module)
       where
        `feat`: The output points' features.
        `F`: The feature extractor.
        `wp`: The latent codes in W-sapce.
        `points`: Sampled points.
        `options`: Some options for rendering.
        `ref_representation`: The reference representation obtained in step 2.
        `post_module`: The post module, is usually a MLP.
    4. Get the sigma's and rgb's value (or feature) by feeding `feat` in
       step 3 into one or two fully-connected layer head.
    5. Coarse pass to do the integration.
    6. Hierarchically sample points on top of step 5.
    6. Fine pass to do the integration.

    Note: In the following scripts, meanings of variables `N, H, W, R, K, C` are:

    - `N`: Batch size.
    - `H`: Height of image.
    - `W`: Width of image.
    - `R`: Number of rays, usually equals `H * W`.
    - `K`: Number of points on each ray.
    - `C`: Number of channels w.r.t. features or images, e.t.c.
    """

    def __init__(self):
        super().__init__()
        self.point_sampler = PointSampler()
        self.integrator = Integrator()

    def forward(
        self,
        wp,
        feature_extractor,
        rendering_options,
        cam2world_matrix=None,
        position_encoder=None,
        ref_representation=None,
        post_module=None,
        post_module_kwargs={},
        fc_head=None,
        fc_head_kwargs={},
    ):
        #TODO: Organize `rendering_options` like the following format:
        '''
            rendering_options = dict(
                point_sampler_options=dict(
                    focal=None,
                    ...
                )
                integrator_options=dict(...),
                ....,
                xxx=xxx,  # some public parameters.
                ...
            )
        '''
        batch_size= wp.shape[0]

        # Sample points.
        sampling_point_res = self.point_sampler(
            batch_size=batch_size,
            focal=rendering_options.get('focal', None),
            image_boundary_value=rendering_options.get('image_boundary_value',
                                                       0.5),
            cam_look_at_dir=rendering_options.get('cam_look_at_dir', +1),
            pixel_center=rendering_options.get('pixel_center', True),
            y_descending=rendering_options.get('y_descending', False),
            image_size=rendering_options.get('resolution', 64),
            dis_min=rendering_options.get('ray_start', None),
            dis_max=rendering_options.get('ray_end', None),
            cam2world_matrix=cam2world_matrix,
            num_points=rendering_options.get('depth_resolution', 48),
            perturbation_strategy=rendering_options.get(
                'perturbation_strategy', 'uniform'),
            radius_strategy=rendering_options.get('radius_strategy', None),
            radius_fix=rendering_options.get('radius_fix', None),
            polar_strategy=rendering_options.get('polar_strategy', None),
            polar_fix=rendering_options.get('polar_fix', None),
            polar_mean=rendering_options.get('polar_mean', None),
            polar_stddev=rendering_options.get('polar_stddev', None),
            azimuthal_strategy=rendering_options.get('azimuthal_strategy',
                                                     None),
            azimuthal_fix=rendering_options.get('azimuthal_fix', None),
            azimuthal_mean=rendering_options.get('azimuthal_mean', None),
            azimuthal_stddev=rendering_options.get('azimuthal_stddev', None),
            fov=rendering_options.get('fov', 30),
        )
        points = sampling_point_res['points_world']   # [N, H, W, K, 3]
        ray_dirs = sampling_point_res['rays_world']   # [N, H, W, 3]
        ray_origins = sampling_point_res['ray_origins_world'] # [N, H, W, 3]
        z_coarse = sampling_point_res['radii']  # [N, H, W, K]

        # NOTE: `pitch` is used to stand for `polar` in other code.
        camera_polar = sampling_point_res['camera_polar'] # [N]
        # NOTE: `yaw` is used to stand for `azimuthal` in other code.
        camera_azimuthal = sampling_point_res['camera_azimuthal'] # [N]
        if camera_polar is not None:
            camera_polar = camera_polar.unsqueeze(-1)
        if camera_azimuthal is not None:
            camera_azimuthal = camera_azimuthal.unsqueeze(-1)

        # Reshape.
        N, H, W, K, _ = points.shape
        assert N == batch_size
        R = H * W   # number of rays
        points = points.reshape(N, R, K, -1)
        ray_dirs = ray_dirs.reshape(N, R, -1)
        ray_origins = ray_origins.reshape(N, R, -1)
        z_coarse = z_coarse.reshape(N, R, K, -1)

        out = self.get_sigma_rgb(wp,
                                 points,
                                 feature_extractor,
                                 rendering_options=rendering_options,
                                 position_encoder=position_encoder,
                                 ref_representation=ref_representation,
                                 post_module=post_module,
                                 post_module_kwargs=post_module_kwargs,
                                 fc_head=fc_head,
                                 fc_head_kwargs=dict(**fc_head_kwargs,
                                                     wp=wp),
                                 ray_dirs=ray_dirs,
                                 cam_matrix=cam2world_matrix)

        sigmas_coarse = out['sigma']  # [N, H * W * K, 1]
        rgbs_coarse = out['rgb']      # [N, H * W * K, C]
        sigmas_coarse = sigmas_coarse.reshape(N, R, K,
                                              sigmas_coarse.shape[-1])
        rgbs_coarse = rgbs_coarse.reshape(N, R, K, rgbs_coarse.shape[-1])

        # Do the integration.
        N_importance = rendering_options.get('depth_resolution_importance', 0)
        if N_importance > 0:
            # Do the integration in coarse pass.
            rendering_result = self.integrator(rgbs_coarse, sigmas_coarse,
                                                   z_coarse, rendering_options)
            weights = rendering_result['weights']

            # Importrance sampling.
            z_fine = self.sample_importance(
                z_coarse,
                weights,
                N_importance,
                smooth_weights=rendering_options.get('smooth_weights', True))
            points = ray_origins.unsqueeze(-2) + z_fine * ray_dirs.unsqueeze(-2)

            # Get sigma's and rgb's value (or feature).
            out = self.get_sigma_rgb(wp,
                                     points,
                                     feature_extractor,
                                     rendering_options=rendering_options,
                                     position_encoder=position_encoder,
                                     ref_representation=ref_representation,
                                     post_module=post_module,
                                     post_module_kwargs=post_module_kwargs,
                                     fc_head=fc_head,
                                     fc_head_kwargs=dict(**fc_head_kwargs,
                                                         wp=wp),
                                     ray_dirs=ray_dirs,
                                     cam_matrix=cam2world_matrix)

            sigmas_fine = out['sigma']
            rgbs_fine = out['rgb']
            sigmas_fine = sigmas_fine.reshape(N, R, N_importance,
                                              sigmas_fine.shape[-1])
            rgbs_fine = rgbs_fine.reshape(N, R, N_importance,
                                          rgbs_fine.shape[-1])

            # Gather coarse and fine results.
            all_zs, all_rgbs, all_sigmas = self.unify_samples(
                z_coarse, rgbs_coarse, sigmas_coarse,
                z_fine, rgbs_fine, sigmas_fine)

            # Do the integration in fine pass.
            final_rendering_result = self.integrator(
                all_rgbs, all_sigmas, all_zs, rendering_options)

        else:
            final_rendering_result = self.integrator(
                rgbs_coarse, sigmas_coarse, z_coarse, rendering_options)

        return {
            **final_rendering_result,
            **{
                'camera_azimuthal': camera_azimuthal,
                'camera_polar': camera_polar
            }, 
            **{
                'points': points,
                'sigmas': sigmas_fine,
                }
        }

    def get_sigma_rgb(self,
                      wp,
                      points,
                      feature_extractor,
                      rendering_options,
                      position_encoder=None,
                      ref_representation=None,
                      post_module=None,
                      post_module_kwargs={},
                      fc_head=None,
                      fc_head_kwargs={},
                      ray_dirs=None,
                      cam_matrix=None):
        # Get point feature in coarse pass.
        point_features = feature_extractor(wp, points, rendering_options,
                                           position_encoder,
                                           ref_representation, post_module,
                                           post_module_kwargs, ray_dirs, cam_matrix)

        # Get sigma's and rgb's value (or feature).
        if ray_dirs.ndim != points.ndim:
            ray_dirs = ray_dirs.unsqueeze(-2).expand_as(points)
        ray_dirs = ray_dirs.reshape(ray_dirs.shape[0], -1, ray_dirs.shape[-1])
                                                    # with shape [N, R * K, 3]
        out = fc_head(point_features, dirs=ray_dirs, **fc_head_kwargs)

        if rendering_options.get('noise_std', 0) > 0:
            out['sigma'] = out['sigma'] + torch.randn_like(
                out['sigma']) * rendering_options['noise_std']

        return out

    def unify_samples(self, depths1, rgbs1, sigmas1, depths2, rgbs2, sigmas2):
        all_depths = torch.cat([depths1, depths2], dim=-2)
        all_colors = torch.cat([rgbs1, rgbs2], dim=-2)
        all_densities = torch.cat([sigmas1, sigmas2], dim=-2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(
            all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2,
                                     indices.expand(-1, -1, -1, 1))

        return all_depths, all_colors, all_densities

    def sample_importance(self,
                          z_vals,
                          weights,
                          N_importance,
                          smooth_weights=False):
        """ Implements NeRF importance sampling.

        Returns:
            importance_z_vals: Depths of importance sampled points along rays.
        """
        with torch.no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape
            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1) + 1e-5

            # smooth weights
            if smooth_weights:
                weights = torch.nn.functional.max_pool1d(
                    weights.unsqueeze(1).float(), 2, 1, padding=1)
                weights = torch.nn.functional.avg_pool1d(weights, 2,
                                                         1).squeeze()
                weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1],
                                                N_importance).detach().reshape(
                                                    batch_size, num_rays,
                                                    N_importance, 1)
        return importance_z_vals

    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """Sample `N_importance` samples from `bins` with distribution defined
           by `weights`.

        Args:
            bins: (N_rays, N_samples_+1) where N_samples_ is the number of
                coarse samples per ray - 2
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero

        Returns:
            samples: the sampled samples

        Source:
            https://github.com/kwea123/nerf_pl/blob/master/models/rendering.py

        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps
        # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1,
                                keepdim=True)  # (N_rays, N_samples_)
        cdf = torch.cumsum(pdf, -1)  # (N_rays, N_samples),
        # cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf],
                        -1)          # (N_rays, N_samples_+1)
        # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u)
        below = torch.clamp_min(inds - 1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above],
                                -1).view(N_rays, 2 * N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled)
        cdf_g = cdf_g.view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1,
                              inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom[denom < eps] = 1  # denom equals 0 means a bin has weight 0,
        # in which case it will not be sampled
        # anyway, therefore any value for it is fine
        # (set to 1 here)

        samples = (bins_g[..., 0] + (u - cdf_g[..., 0]) /
                                denom * (bins_g[..., 1] - bins_g[..., 0]))

        return samples
