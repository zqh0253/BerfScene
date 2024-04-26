# python3.8
"""Contains the implementation of generator described in EG3D."""

import torch
import torch.nn as nn
import numpy as np
from models.utils.official_stylegan2_model_helper import MappingNetwork
from models.utils.official_stylegan2_model_helper import FullyConnectedLayer
from models.utils.eg3d_superres import SuperresolutionHybrid2X
from models.utils.eg3d_superres import SuperresolutionHybrid4X
from models.utils.eg3d_superres import SuperresolutionHybrid8XDC
from models.rendering.renderer import Renderer
from models.rendering.feature_extractor import FeatureExtractor
from models.volumegan_generator import FeatureVolume
from models.volumegan_generator import PositionEncoder


class EG3DGeneratorFV(nn.Module):

    def __init__(
            self,
            # Input latent (Z) dimensionality.
            z_dim,
            # Conditioning label (C) dimensionality.
            c_dim,
            # Intermediate latent (W) dimensionality.
            w_dim,
            # Final output image resolution.
            img_resolution,
            # Number of output color channels.
            img_channels,
            # Number of fp16 layers of SR Network.
            sr_num_fp16_res=0,
            # Arguments for MappingNetwork.
            mapping_kwargs={},
            # Arguments for rendering.
            rendering_kwargs={},
            # Arguments for SuperResolution Network.
            sr_kwargs={},
            # Configs for FeatureVolume.
            fv_cfg=dict(feat_res=32,
                        init_res=4,
                        base_channels=256,
                        output_channels=32,
                        w_dim=512),
            # Configs for position encoder.
            embed_cfg=dict(input_dim=3, max_freq_log2=10 - 1, N_freqs=10),
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        # Set up mapping network.
        # Here `num_ws = 2`: one for FeatureVolume Network injection and one for
        # post_neural_renderer injection.
        num_ws = 2
        self.mapping_network = MappingNetwork(z_dim=z_dim,
                                              c_dim=c_dim,
                                              w_dim=w_dim,
                                              num_ws=num_ws,
                                              **mapping_kwargs)

        # Set up the overall renderer.
        self.renderer = Renderer()

        # Set up the feature extractor.
        self.feature_extractor = FeatureExtractor(ref_mode='feature_volume')

        # Set up the reference representation generator.
        self.ref_representation_generator = FeatureVolume(**fv_cfg)

        # Set up the position encoder.
        self.position_encoder = PositionEncoder(**embed_cfg)

        # Set up the post module in the feature extractor.
        self.post_module = None

        # Set up the post neural renderer.
        self.post_neural_renderer = None
        sr_kwargs_total = dict(
            channels=32,
            img_resolution=img_resolution,
            sr_num_fp16_res=sr_num_fp16_res,
            sr_antialias=rendering_kwargs['sr_antialias'],)
        sr_kwargs_total.update(**sr_kwargs)
        if img_resolution == 128:
            self.post_neural_renderer = SuperresolutionHybrid2X(
                **sr_kwargs_total)
        elif img_resolution == 256:
            self.post_neural_renderer = SuperresolutionHybrid4X(
                **sr_kwargs_total)
        elif img_resolution == 512:
            self.post_neural_renderer = SuperresolutionHybrid8XDC(
                **sr_kwargs_total)
        else:
            raise TypeError(f'Unsupported image resolution: {img_resolution}!')

        # Set up the fully-connected layer head.
        self.fc_head = OSGDecoder(
            32, {
                'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1),
                'decoder_output_dim': 32
            })

        # Set up some rendering related arguments.
        self.neural_rendering_resolution = rendering_kwargs.get(
            'resolution', 64)
        self.rendering_kwargs = rendering_kwargs

    def mapping(self,
                z,
                c,
                truncation_psi=1,
                truncation_cutoff=None,
                update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        return self.mapping_network(z,
                                    c *
                                    self.rendering_kwargs.get('c_scale', 0),
                                    truncation_psi=truncation_psi,
                                    truncation_cutoff=truncation_cutoff,
                                    update_emas=update_emas)

    def synthesis(self,
                  wp,
                  c,
                  neural_rendering_resolution=None,
                  update_emas=False,
                  **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        if self.rendering_kwargs.get('random_pose', False):
            cam2world_matrix = None

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        feature_volume = self.ref_representation_generator(wp)

        rendering_result = self.renderer(
            wp=wp,
            feature_extractor=self.feature_extractor,
            rendering_options=self.rendering_kwargs,
            cam2world_matrix=cam2world_matrix,
            position_encoder=self.position_encoder,
            ref_representation=feature_volume,
            post_module=self.post_module,
            fc_head=self.fc_head)

        feature_samples = rendering_result['composite_rgb']
        depth_samples = rendering_result['composite_depth']

        # Reshape to keep consistent with 'raw' neural-rendered image.
        N = wp.shape[0]
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(
            N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run the post neural renderer to get final image.
        # Here, the post neural renderer is a super-resolution network.
        rgb_image = feature_image[:, :3]
        sr_image = self.post_neural_renderer(
            rgb_image,
            feature_image,
            wp,
            noise_mode=self.rendering_kwargs['superresolution_noise_mode'],
            **{
                k: synthesis_kwargs[k]
                for k in synthesis_kwargs.keys() if k != 'noise_mode'
            })

        return {
            'image': sr_image,
            'image_raw': rgb_image,
            'image_depth': depth_image
        }

    def sample(self,
               coordinates,
               directions,
               z,
               c,
               truncation_psi=1,
               truncation_cutoff=None,
               update_emas=False):
        # Compute RGB features, density for arbitrary 3D coordinates.
        # Mostly used for extracting shapes.
        wp = self.mapping_network(z,
                          c,
                          truncation_psi=truncation_psi,
                          truncation_cutoff=truncation_cutoff,
                          update_emas=update_emas)
        feature_volume = self.ref_representation_generator(wp)
        result = self.renderer.get_sigma_rgb(
            wp=wp,
            points=coordinates,
            feature_extractor=self.feature_extractor,
            fc_head=self.fc_head,
            rendering_options=self.rendering_kwargs,
            ref_representation=feature_volume,
            position_encoder=self.position_encoder,
            post_module=self.post_module,
            ray_dirs=directions)

        return result

    def sample_mixed(self,
                     coordinates,
                     directions,
                     wp):
        # Same as function `self.sample()`, but expects latent vectors 'wp'
        # instead of Gaussian noise 'z'.
        feature_volume = self.ref_representation_generator(wp)
        result = self.renderer.get_sigma_rgb(
            wp=wp,
            points=coordinates,
            feature_extractor=self.feature_extractor,
            fc_head=self.fc_head,
            rendering_options=self.rendering_kwargs,
            ref_representation=feature_volume,
            position_encoder=self.position_encoder,
            post_module=self.post_module,
            ray_dirs=directions)

        return result

    def forward(self,
                z,
                c,
                c_swapped=None,      # `c_swapped` is swapped pose conditioning.
                style_mixing_prob=0,
                truncation_psi=1,
                truncation_cutoff=None,
                neural_rendering_resolution=None,
                update_emas=False,
                sample_mixed=False,
                coordinates=None,
                **synthesis_kwargs):

        # Render a batch of generated images.
        c_wp = c.clone()
        if c_swapped is not None:
            c_wp = c_swapped.clone()
        wp = self.mapping_network(z,
                                  c_wp,
                                  truncation_psi=truncation_psi,
                                  truncation_cutoff=truncation_cutoff,
                                  update_emas=update_emas)
        if style_mixing_prob > 0:
            cutoff = torch.empty([], dtype=torch.int64,
                                 device=wp.device).random_(1, wp.shape[1])
            cutoff = torch.where(
                torch.rand([], device=wp.device) < style_mixing_prob, cutoff,
                torch.full_like(cutoff, wp.shape[1]))
            wp[:, cutoff:] = self.mapping_network(
                torch.randn_like(z), c, update_emas=update_emas)[:, cutoff:]
        if not sample_mixed:
            gen_output = self.synthesis(
                wp,
                c,
                update_emas=update_emas,
                neural_rendering_resolution=neural_rendering_resolution,
                **synthesis_kwargs)

            return {
                'wp': wp,
                'gen_output': gen_output,
            }

        else:
            # Only for density regularization in training process.
            assert coordinates is not None
            sample_sigma = self.sample_mixed(coordinates,
                                             torch.randn_like(coordinates),
                                             wp)['sigma']

            return {
                'wp': wp,
                'sample_sigma': sample_sigma
            }


class OSGDecoder(nn.Module):
    """Defines fully-connected layer head in EG3D."""
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = nn.Sequential(
            FullyConnectedLayer(n_features,
                                self.hidden_dim,
                                lr_multiplier=options['decoder_lr_mul']),
            nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim,
                                1 + options['decoder_output_dim'],
                                lr_multiplier=options['decoder_lr_mul']))

    def forward(self, point_features, wp=None, dirs=None):
        # point_features.shape: [N, C, M, 1].
        point_features = point_features.squeeze(-1)
        point_features = point_features.permute(0, 2, 1)
        x = point_features

        N, M, C = x.shape
        x = x.reshape(N * M, C)

        x = self.net(x)
        x = x.reshape(N, M, -1)

        # Uses sigmoid clamping from MipNeRF
        rgb = torch.sigmoid(x[..., 1:]) * (1 + 2 * 0.001) - 0.001
        sigma = x[..., 0:1]

        return {'rgb': rgb, 'sigma': sigma}
