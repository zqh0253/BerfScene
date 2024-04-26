# python3.8
"""Contains the implementation of generator described in BEV3D."""

import torch
import torch.nn as nn
from models.utils.official_stylegan2_model_helper import Generator as StyleGAN2Backbone
from models.utils.official_stylegan2_model_helper import FullyConnectedLayer
from models.utils.eg3d_superres import SuperresolutionHybrid2X
from models.utils.eg3d_superres import SuperresolutionHybrid4X
from models.utils.eg3d_superres import SuperresolutionHybrid4X_conststyle
from models.utils.eg3d_superres import SuperresolutionHybrid8XDC
from models.rendering.renderer import Renderer
from models.rendering.feature_extractor import FeatureExtractor

from models.utils.spade import SPADEGenerator

class BEV3DGenerator(nn.Module):

    def __init__(
            self,
            z_dim,
            semantic_nc,
            ngf,
            bev_grid_size,
            aspect_ratio,
            num_upsampling_layers,
            not_use_vae,
            norm_G,
            img_resolution,
            interpolate_sr,
            segmask=False,
            dim_seq='16,8,4,2,1',
            xyz_pe=False,
            hidden_dim=64,
            additional_layer_num=0,
            sr_num_fp16_res=0,      # Number of fp16 layers of SR Network.
            rendering_kwargs={},    # Arguments for rendering.
            sr_kwargs={},           # Arguments for SuperResolution Network.
    ):
        super().__init__()

        self.z_dim = z_dim
        self.interpolate_sr = interpolate_sr
        self.segmask = segmask

        # Set up the overall renderer.
        self.renderer = Renderer()

        # Set up the feature extractor.
        self.feature_extractor = FeatureExtractor(ref_mode='bev_plane_clevr', xyz_pe=xyz_pe)

        # Set up the reference representation generator.
        self.backbone = SPADEGenerator(z_dim=z_dim, semantic_nc=semantic_nc, ngf=ngf, dim_seq=dim_seq, bev_grid_size=bev_grid_size,
                                        aspect_ratio=aspect_ratio, num_upsampling_layers=num_upsampling_layers, 
                                        not_use_vae=not_use_vae, norm_G=norm_G)
        print('backbone SPADEGenerator set up!')

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
            self.post_neural_renderer = SuperresolutionHybrid4X_conststyle(
                **sr_kwargs_total)
        elif img_resolution == 512:
            self.post_neural_renderer = SuperresolutionHybrid8XDC(
                **sr_kwargs_total)
        else:
            raise TypeError(f'Unsupported image resolution: {img_resolution}!')

        # Set up the fully-connected layer head.
        self.fc_head = OSGDecoder(
            128 if xyz_pe else 64 , {
                'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1),
                'decoder_output_dim': 32
            },
            hidden_dim=hidden_dim,
            additional_layer_num=additional_layer_num
            )

        # Set up some rendering related arguments.
        self.neural_rendering_resolution = rendering_kwargs.get(
            'resolution', 64)
        self.rendering_kwargs = rendering_kwargs

    def synthesis(self,
                  z,
                  c,
                  seg,
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

        xy_planes = self.backbone(z=z, input=seg)
        if self.segmask:
            xy_planes = xy_planes * seg[:, 0, ...][:, None, ...]

        # import pdb;pdb.set_trace()

        wp = z   # in our case, we do not use wp.

        rendering_result = self.renderer(
            wp=wp,
            feature_extractor=self.feature_extractor,
            rendering_options=self.rendering_kwargs,
            cam2world_matrix=cam2world_matrix,
            position_encoder=None,
            ref_representation=xy_planes,
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
        if self.interpolate_sr:
            sr_image = torch.nn.functional.interpolate(rgb_image, size=(256, 256), mode='bilinear', align_corners=False)
        else:
            sr_image = self.post_neural_renderer(
            rgb_image,
            feature_image,
            # wp,
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
               seg,
               truncation_psi=1,
               truncation_cutoff=None,
               update_emas=False,
               **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates.
        # Mostly used for extracting shapes.
        cam2world_matrix = c[:, :16].view(-1, 4, 4) 
        xy_planes = self.backbone(z=z, input=seg)
        wp = z
        result = self.renderer.get_sigma_rgb(
            wp=wp,
            points=coordinates,
            feature_extractor=self.feature_extractor,
            fc_head=self.fc_head,
            rendering_options=self.rendering_kwargs,
            ref_representation=xy_planes,
            post_module=self.post_module,
            ray_dirs=directions,
            cam_matrix=cam2world_matrix)

        return result

    def sample_mixed(self,
                     coordinates,
                     directions,
                     z, c, seg,
                     truncation_psi=1,
                     truncation_cutoff=None,
                     update_emas=False,
                     **synthesis_kwargs):
        # Same as function `self.sample()`, but expects latent vectors 'wp'
        # instead of Gaussian noise 'z'.
        cam2world_matrix = c[:, :16].view(-1, 4, 4) 
        xy_planes = self.backbone(z=z, input=seg)
        wp = z
        result = self.renderer.get_sigma_rgb(
            wp=wp,
            points=coordinates,
            feature_extractor=self.feature_extractor,
            fc_head=self.fc_head,
            rendering_options=self.rendering_kwargs,
            ref_representation=xy_planes,
            post_module=self.post_module,
            ray_dirs=directions,
            cam_matrix=cam2world_matrix)

        return result

    def forward(self,
                z,
                c,
                seg,
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

        if not sample_mixed:
            gen_output = self.synthesis(
                z,
                c,
                seg,
                update_emas=update_emas,
                neural_rendering_resolution=neural_rendering_resolution,
                **synthesis_kwargs)

            return {
                'wp': z,
                'gen_output': gen_output,
            }

        else:
            # Only for density regularization in training process.
            assert coordinates is not None
            sample_sigma = self.sample_mixed(coordinates,
                                             torch.randn_like(coordinates),
                                             z, c, seg,
                                             update_emas=False)['sigma']

            return {
                'wp': z,
                'sample_sigma': sample_sigma
            }


class OSGDecoder(nn.Module):
    """Defines fully-connected layer head in EG3D."""
    def __init__(self, n_features, options, hidden_dim=64, additional_layer_num=0):
        super().__init__()
        self.hidden_dim = hidden_dim

        lst = []
        lst.append(FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']))
        lst.append(nn.Softplus())
        for i in range(additional_layer_num):
            lst.append(FullyConnectedLayer(self.hidden_dim, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']))
            lst.append(nn.Softplus())
        lst.append(FullyConnectedLayer(self.hidden_dim, 1+options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul']))
        self.net = nn.Sequential(*lst)

        # self.net = nn.Sequential(
        #     FullyConnectedLayer(n_features,
        #                         self.hidden_dim,
        #                         lr_multiplier=options['decoder_lr_mul']),
        #     nn.Softplus(),
        #     FullyConnectedLayer(self.hidden_dim,
        #                         1 + options['decoder_output_dim'],
        #                         lr_multiplier=options['decoder_lr_mul']))

    def forward(self, point_features, wp=None, dirs=None):
        # Aggregate features
        # point_features.shape: [N, R, K, C].
        # Average across 'X, Y, Z' planes.

        N, R, K, C = point_features.shape
        x = point_features.reshape(-1, point_features.shape[-1])
        x = self.net(x)
        x = x.view(N, -1, x.shape[-1])

        # Uses sigmoid clamping from MipNeRF
        rgb = torch.sigmoid(x[..., 1:]) * (1 + 2 * 0.001) - 0.001
        sigma = x[..., 0:1]

        return {'rgb': rgb, 'sigma': sigma}
