# python3.8
"""Defines feature extractor in 3D generation pipeline."""

import torch
from .triplane_sampler import TriplaneSampler
from .utils import interpolate_feature
from einops import rearrange
import math

__all__ = ['FeatureExtractor']


_REF_MODE = ['none', 'tri_plane', 'feature_volume', 'bev_plane_clevr_256', 'bev_plane_clevr_512', 'bev_plane_carla']


class FeatureExtractor(torch.nn.Module):
    """Defines the feature extractor in 3D Generation Pipeline.

    Basically, the feature extractor takes in the latent code and sampled points
    in addition to the reference representation as input, and outputs the
    feature representation which contains information of each point's color and
    density.

    """

    def __init__(self, ref_mode='none', xyz_pe=False, reverse_xy=True):
        super().__init__()
        self.ref_mode = ref_mode
        self.xyz_pe = xyz_pe
        self.reverse_xy = reverse_xy
        assert ref_mode in _REF_MODE
        if ref_mode == 'tri_plane':
            self.plane_axes = TriplaneSampler.generate_planes()

    def forward(self,
                wp,
                points,
                rendering_options,
                position_encoder=None,
                ref_representation=None,
                post_module=None,
                post_module_kwargs={},
                ray_dirs=None,
                cam_matrix=None,):
        assert points.ndim in [3, 4]
        if points.ndim == 3:
            points = points.unsqueeze(2) # shape: [N, R, C] -> [N, R, 1, C]
        N, R, K, _ = points.shape[:4]
        # (Optional) Positional encoding.
        if position_encoder is not None:
            points_encoding = position_encoder(points) # shape: [N, R, K, C].
            points_encoding = rearrange(points_encoding,
                                        'N R K C -> N C (R K) 1').contiguous()

        # Reshape `points` with shape [N, R*K, 3].
        points = points.reshape(points.shape[0], -1, points.shape[-1])

        # Get pre-point-features by sampling from
        # the reference representation (if exists).
        pre_point_features = points
        if ref_representation is not None:
            assert self.ref_mode is not None
            if self.ref_mode == 'tri_plane':
                pre_point_features = TriplaneSampler.sample_from_planes(
                    self.plane_axes.to(points.device),
                    ref_representation,
                    points,
                    padding_mode='zeros',
                    box_warp=rendering_options.get('box_warp', 1.0))
                # shape: [N, 3, num_points, C], where num_points = H*W*K.
            elif self.ref_mode == 'feature_volume':
                bounds = rendering_options.get(
                    'bounds',
                    [[-0.1886, -0.1671, -0.1956], [0.1887, 0.1692, 0.1872]])
                bounds = torch.Tensor(bounds).to(points.device)
                pre_point_features = interpolate_feature(
                    points, ref_representation, bounds) # shape: [N, C, R*K].
                pre_point_features = pre_point_features.unsqueeze(-1)
                                                        # shape: [N, C, R*K, 1].
                post_module_kwargs.update(points_encoding=points_encoding)
            elif 'bev_plane_clevr' in self.ref_mode:
                h = w = int(self.ref_mode[-3:])
                # first, transform points from world coordinates to bev coordinates
                # cam_matrix: N, 4, 4
                # points: N, 3, R*K
                points_reshape = points # N, R*K, 3
                if self.reverse_xy:
                    y = (0.5 * w - 128 + 256 - (points_reshape[..., 0] /9 + .5) * 256 ) / w * 2 - 1
                    x = (0.5 * h - 128 + (points_reshape[..., 1] /9 + .5) * 256 ) / h * 2 - 1
                else:
                    x = (0.5 * w - 128 + 256 - (points_reshape[..., 0] /9 + .5) * 256 ) / w * 2 - 1
                    y = (0.5 * h - 128 + (points_reshape[..., 1] /9 + .5) * 256 ) / h * 2 - 1
                z = points_reshape[..., -1] / 9 
                points_bev = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], -1)

                # second, sample feature from BEV map
                # ref_representation: N, C, A, A
                # points_bev: N, R*K, 3
                xy = points_bev[..., :2]     # N, R*K, 2
                xy = xy.unsqueeze(2)         # N, R*K, 1, 2
                feat_xy = torch.nn.functional.grid_sample(ref_representation, xy, mode='bilinear', 
                                padding_mode='zeros', align_corners=False)   # N, C, R*K, 1
                feat_xy = feat_xy.squeeze(3) # N, C，R*K
                x = points_bev[..., 0]       # N, R*K
                y = points_bev[..., 1]       # N, R*K
                z = points_bev[..., -1]      # N, R*K

                # third, do positional encoding on z 
                d_model = 32
                div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *-(math.log(10000.0) / d_model))).to(z.device)
                
                pe_x = torch.zeros([x.shape[0], x.shape[1], d_model]).to(x.device)
                pe_x[..., 0::2] = torch.sin(x.unsqueeze(-1).float() * div_term)
                pe_x[..., 1::2] = torch.cos(x.unsqueeze(-1).float() * div_term)
                pe_y = torch.zeros([y.shape[0], y.shape[1], d_model]).to(y.device)
                pe_y[..., 0::2] = torch.sin(y.unsqueeze(-1).float() * div_term)
                pe_y[..., 1::2] = torch.cos(y.unsqueeze(-1).float() * div_term)
                pe_z = torch.zeros([z.shape[0], z.shape[1], d_model]).to(z.device)
                pe_z[..., 0::2] = torch.sin(z.unsqueeze(-1).float() * div_term)
                pe_z[..., 1::2] = torch.cos(z.unsqueeze(-1).float() * div_term)
                if self.xyz_pe:
                    feat_xyz = torch.cat([feat_xy, pe_x.permute(0, 2, 1), pe_y.permute(0,2,1),pe_z.permute(0, 2, 1)], 1)    # N, C+d_model, R*K 
                else:
                    feat_xyz = torch.cat([feat_xy ,pe_z.permute(0, 2, 1)], 1)    # N, C+d_model, R*K 
                pre_point_features = feat_xyz.permute(0, 2, 1)             # N, RK, C+d_model
                pre_point_features = pre_point_features.view(N, R, K, -1)
            elif self.ref_mode == 'bev_plane_carla':
                x = (217.5 - 8 * points[..., 0]) / 128 - 1
                y = (128.0 + 8 * points[..., 1]) / 128 - 1
                z = points[..., 2]
                points_bev = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], -1)

                xy = points_bev[..., :2]
                xy = xy.unsqueeze(2)
                feat_xy = torch.nn.functional.grid_sample(ref_representation, xy, mode='bilinear',padding_mode='zeros', align_corners=False)
                feat_xy = feat_xy.squeeze(3)
                z = points_bev[..., -1] 
                d_model = 32
                div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *-(math.log(10000.0) / d_model))).to(z.device)
                pe_x = torch.zeros([x.shape[0], x.shape[1], d_model]).to(x.device)
                pe_x[..., 0::2] = torch.sin(x.unsqueeze(-1).float() * div_term)
                pe_x[..., 1::2] = torch.cos(x.unsqueeze(-1).float() * div_term)
                pe_y = torch.zeros([y.shape[0], y.shape[1], d_model]).to(y.device)
                pe_y[..., 0::2] = torch.sin(y.unsqueeze(-1).float() * div_term)
                pe_y[..., 1::2] = torch.cos(y.unsqueeze(-1).float() * div_term)
                pe_z = torch.zeros([z.shape[0], z.shape[1], d_model]).to(z.device)
                pe_z[..., 0::2] = torch.sin(z.unsqueeze(-1).float() * div_term)
                pe_z[..., 1::2] = torch.cos(z.unsqueeze(-1).float() * div_term)
                if self.xyz_pe:
                    feat_xyz = torch.cat([feat_xy, pe_x.permute(0, 2, 1), pe_y.permute(0,2,1),pe_z.permute(0, 2, 1)], 1)    # N, C+d_model, R*K 
                else:
                    feat_xyz = torch.cat([feat_xy ,pe_z.permute(0, 2, 1)], 1)    # N, C+d_model, R*K 
                pre_point_features = feat_xyz.permute(0, 2, 1)             # N, RK, C+d_model
                pre_point_features = pre_point_features.view(N, R, K, -1)
            else:
                raise NotImplementedError
                
        # Get post-point-features by feeding pre-point-features into the
        # post-module (if exists).
        if post_module is not None:
            post_point_features = post_module(pre_point_features, wp,
                                              **post_module_kwargs)
        else:
            post_point_features = pre_point_features

        if post_point_features.ndim == 2:
            post_point_features = rearrange('(N R K) C -> N R K C',
                                            N=N, R=R, K=K).contiguous()

        return post_point_features
