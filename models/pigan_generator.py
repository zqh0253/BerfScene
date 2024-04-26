# python3.7
"""Contains the implementation of generator described in PiGAN."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from .utils.ops import all_gather
from .rendering.renderer import Renderer
from .rendering.feature_extractor import FeatureExtractor

__all__ = ['PiGANGenerator']


class PiGANGenerator(nn.Module):
    """Defines the generator network in PiGAN."""
    def __init__(self,
                 # Settings for mapping network.
                 z_dim=256,
                 w_dim=256,
                 repeat_w=False,
                 normalize_z=False,
                 mapping_layers=3,
                 mapping_hidden_dim=256,
                 # Settings for conditional generation.
                 label_dim=0,
                 embedding_dim=512,
                 normalize_embedding=True,
                 normalize_embedding_latent=False,
                 label_concat=True,
                 # Settings for synthesis network.
                 resolution=-1,
                 synthesis_input_dim=3,
                 synthesis_output_dim=256,
                 synthesis_layers=8,
                 grid_scale=0.24,
                 eps=1e-8,
                 # Settings for rendering module.
                 rendering_kwargs={}):
        """Initializes with basic settings."""
        super().__init__()

        self.z_dim = z_dim
        self.w_dim = w_dim
        self.repeat_w = repeat_w
        self.normalize_z = normalize_z
        self.mapping_layers = mapping_layers

        self.latent_dim = (z_dim,)
        self.label_dim = label_dim
        self.embedding_dim = embedding_dim
        self.normalize_embedding = normalize_embedding
        self.normalize_embedding_latent = normalize_embedding_latent

        self.resolution = resolution
        self.num_layers = synthesis_layers
        self.eps = eps

        if self.repeat_w:
            self.mapping_space_dim = self.w_dim
        else:
            self.mapping_space_dim = self.w_dim * (self.num_layers + 1)

        # Mapping Network to tranform latent codes from Z-Space into W-Space.
        self.mapping = MappingNetwork(
            input_dim=z_dim,
            output_dim=w_dim,
            num_outputs=synthesis_layers + 1,
            repeat_output=repeat_w,
            normalize_input=normalize_z,
            num_layers=mapping_layers,
            hidden_dim=mapping_hidden_dim,
            label_dim=label_dim,
            embedding_dim=embedding_dim,
            normalize_embedding=normalize_embedding,
            normalize_embedding_latent=normalize_embedding_latent,
            eps=eps,
            label_concat=label_concat,
            lr=None)

        # Set up the overall renderer.
        self.renderer = Renderer()

        # Set up the reference representation generator.
        self.ref_representation_generator = None

        # Set up the feature extractor.
        self.feature_extractor = FeatureExtractor(ref_mode='none')

        # Set up the  post module in the feature extractor.
        self.post_module = MLPNetwork(w_dim=w_dim,
                                      in_channels=synthesis_input_dim,
                                      num_layers=synthesis_layers,
                                      out_channels=synthesis_output_dim,
                                      grid_scale=grid_scale)

        # Set up the fully-connected layer head.
        self.fc_head = FCHead(w_dim=w_dim,
                              channels=synthesis_output_dim,
                              mlp_length=self.post_module.mlp_length)

        # Set up the post neural renderer.
        self.post_neural_renderer = None

        # This is used for truncation trick.
        if self.repeat_w:
            self.register_buffer('w_avg', torch.zeros(w_dim))
        else:
            self.register_buffer('w_avg', torch.zeros(self.num_layers * w_dim))

        # Set up some rendering related arguments.
        self.rendering_kwargs = rendering_kwargs

        # Initialize weights.
        self.init_weights()

    def init_weights(self):
        self.mapping.init_weights()
        self.post_module.init_weights()
        self.fc_head.init_weights()

    def forward(self,
                z,
                label=None,
                lod=None,
                w_moving_decay=None,
                sync_w_avg=False,
                style_mixing_prob=None,
                noise_std=None,
                trunc_psi=None,
                trunc_layers=None,
                enable_amp=False):
        if noise_std is not None:
            self.rendering_kwargs.update(noise_std=noise_std)

        lod = self.post_module.lod.cpu().tolist() if lod is None else lod

        mapping_results = self.mapping(z, label)
        w = mapping_results['w']
        wp = mapping_results.pop('wp')

        if self.training and w_moving_decay is not None:
            if sync_w_avg:
                batch_w_avg = all_gather(w.detach()).mean(dim=0)
            else:
                batch_w_avg = w.detach().mean(dim=0)
            self.w_avg.copy_(batch_w_avg.lerp(self.w_avg, w_moving_decay))

        # Truncation.
        if not self.training:
            trunc_psi = 1.0 if trunc_psi is None else trunc_psi
            trunc_layers = 0 if trunc_layers is None else trunc_layers
            if trunc_psi < 1.0 and trunc_layers > 0:
                w_avg = self.w_avg.reshape(1, -1, self.w_dim)[:, :trunc_layers]
                wp[:, :trunc_layers] = w_avg.lerp(
                    wp[:, :trunc_layers], trunc_psi)

        with autocast(enabled=enable_amp):
            rendering_result = self.renderer(
                wp=wp,
                feature_extractor=self.feature_extractor,
                rendering_options=self.rendering_kwargs,
                position_encoder=None,
                ref_representation=None,
                post_module=self.post_module,
                post_module_kwargs=dict(lod=lod),
                fc_head=self.fc_head)

        image = rendering_result['composite_rgb'].reshape(
            z.shape[0], self.resolution, self.resolution,
            -1).permute(0, 3, 1, 2)

        camera = torch.cat([
            rendering_result['camera_polar'],
            rendering_result['camera_azimuthal']
        ], -1)

        return {
            **mapping_results,
            'image': image,
            'camera': camera,
            'latent': z
        }


class MappingNetwork(nn.Module):
    """Implements the latent space mapping module.

    Basically, this module executes several dense layers in sequence, and the
    label embedding if needed.
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 num_outputs,
                 repeat_output,
                 normalize_input,
                 num_layers,
                 hidden_dim,
                 label_dim,
                 embedding_dim,
                 normalize_embedding,
                 normalize_embedding_latent,
                 eps,
                 label_concat,
                 lr=None):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_outputs = num_outputs
        self.repeat_output = repeat_output
        self.normalize_input = normalize_input
        self.num_layers = num_layers
        # self.out_channels = out_channels
        # TODO
        # self.lr_mul = lr_mul

        self.label_dim = label_dim
        self.embedding_dim = embedding_dim
        self.normalize_embedding = normalize_embedding
        self.normalize_embedding_latent = normalize_embedding_latent
        self.eps = eps
        self.label_concat = label_concat

        self.norm = PixelNormLayer(dim=1, eps=eps)

        if num_outputs is not None and not repeat_output:
            output_dim = output_dim * num_outputs

        if self.label_dim > 0:
            if self.label_concat:
                input_dim = input_dim + embedding_dim
                self.embedding = EqualLinear(label_dim,
                                            embedding_dim,
                                            bias=True,
                                            bias_init=0,
                                            lr_mul=1)
            else:
                self.embedding = EqualLinear(label_dim,
                                            output_dim,
                                            bias=True,
                                            bias_init=0,
                                            lr_mul=1)

        network = []
        for i in range(num_layers):
            in_channels = (input_dim if i == 0 else hidden_dim)
            out_channels = (output_dim if i == (num_layers - 1) else hidden_dim)
            network.append(nn.Linear(in_channels, out_channels))
            network.append(nn.LeakyReLU(0.2, inplace=True))
        self.network = nn.Sequential(*network)

    def init_weights(self):
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight,
                                        a=0.2,
                                        mode='fan_in',
                                        nonlinearity='leaky_relu')

    def forward(self, z, label=None):
        if z.ndim != 2 or z.shape[1] != self.input_dim:
            raise ValueError(f'Input latent code should be with shape '
                             f'[batch_size, input_dim], where '
                             f'`input_dim` equals to {self.input_dim}!\n'
                             f'But `{z.shape}` is received!')
        if self.normalize_input:
            z = self.norm(z)
        if self.label_dim > 0:
            if label is None:
                raise ValueError(f'Model requires an additional label '
                                 f'(with dimension {self.label_dim}) as input, '
                                 f'but no label is received!')
            if label.ndim != 2 or label.shape != (z.shape[0], self.label_dim):
                raise ValueError(f'Input label should be with shape '
                                 f'[batch_size, label_dim], where '
                                 f'`batch_size` equals to that of '
                                 f'latent codes ({z.shape[0]}) and '
                                 f'`label_dim` equals to {self.label_dim}!\n'
                                 f'But `{label.shape}` is received!')
            label = label.to(dtype=torch.float32)

            embedding = self.embedding(label)
            if self.normalize_embedding and self.label_concat:
                embedding = self.norm(embedding)
            if self.label_concat:
                w = torch.cat((z, embedding), dim=1)
            else:
                w = z
        else:
            w = z

        if (self.label_dim > 0 and self.normalize_embedding_latent
            and self.label_concat):
            w = self.norm(w)

        for layer in self.network:
            w = layer(w)

        if self.label_dim > 0 and (not self.label_concat):
            w = w * embedding

        wp = None
        if self.num_outputs is not None:
            if self.repeat_output:
                wp = w.unsqueeze(1).repeat((1, self.num_outputs, 1))
            else:
                wp = w.reshape(-1, self.num_outputs, self.output_dim)

        results = {
            'z': z,
            'label': label,
            'w': w,
            'wp': wp,
        }
        if self.label_dim > 0:
            results['embedding'] = embedding
        return results


class MLPNetwork(nn.Module):
    """Defines MLP Network in Pi-GAN."""
    def __init__(self,
                 w_dim,
                 in_channels,
                 num_layers,
                 out_channels,
                 grid_scale=0.24):
        super().__init__()

        self.in_channels = in_channels
        self.w_dim = w_dim
        self.out_channels = out_channels

        self.register_buffer('lod', torch.zeros(()))

        self.grid_warper = UniformBoxWarp(grid_scale)

        network = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            out_channels = out_channels
            film = FiLMLayer(in_channels, out_channels, w_dim)
            network.append(film)
        self.mlp_network = nn.Sequential(*network)

        self.mlp_length = len(self.mlp_network)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, FiLMLayer):
                module.init_weights()

        self.mlp_network[0].init_weights(first=True)

    def forward(self, pts, wp, lod=None):
        num_dims = pts.ndim
        assert num_dims in [3, 4, 5]
        if num_dims == 5:
            N, H, W, K, C = pts.shape
            pts = pts.reshape(N, H * W * K, C)
        elif num_dims == 4:
            N, R, K, C = pts.shape
            pts = pts.reshape(N, R * K, C)

        x = self.grid_warper(pts)

        for idx, layer in enumerate(self.mlp_network):
            x = layer(x, wp[:, idx])

        return x


class FCHead(nn.Module):
    """Defines fully-connected layer head in Pi-GAN to decode `feature` into
    `sigma` and `rgb`."""

    def __init__(self, w_dim, channels, mlp_length):
        super().__init__()

        self.w_dim = w_dim
        self.channels = channels
        self.mlp_length = mlp_length

        self.sigma_head = nn.Linear(channels, 1)
        self.rgb_film = FiLMLayer(channels + 3, channels, w_dim)
        self.rgb_head = nn.Linear(channels, 3)

    def init_weights(self,):
        self.sigma_head.apply(freq_init(25))
        self.rgb_head.apply(freq_init(25))

        self.rgb_film.init_weights()

    def forward(self, point_features, wp, dirs):
        sigma = self.sigma_head(point_features)

        dirs = torch.cat([point_features, dirs], dim=-1)
        rgb = self.rgb_film(dirs, wp[:, self.mlp_length])
        rgb = self.rgb_head(rgb).sigmoid()

        results = {'sigma': sigma, 'rgb': rgb}

        return results


class FiLMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, w_dim, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w_dim = w_dim

        self.layer = nn.Linear(input_dim, output_dim)
        self.style = nn.Linear(w_dim, output_dim*2)

    def init_weights(self, first=False):
        # initial with 25 frequency
        if not first:
            self.layer.apply(freq_init(25))
        else:
            self.layer.apply(first_film_init)
        # kaiming initial && scale 1/4
        nn.init.kaiming_normal_(self.style.weight,
                                a=0.2,
                                mode='fan_in',
                                nonlinearity='leaky_relu')
        with torch.no_grad(): self.style.weight *= 0.25

    def extra_repr(self):
        return (f'in_ch={self.input_dim}, '
                f'out_ch={self.output_dim}, '
                f'w_ch={self.w_dim}')

    def forward(self, x, wp):
        x = self.layer(x)
        style = self.style(wp)
        style_split = style.unsqueeze(1).chunk(2, dim=2)
        freq = style_split[0]
        # Scale for sin activation
        freq = freq*15 + 30
        phase_shift = style_split[1]
        return torch.sin(freq * x + phase_shift)

class PixelNormLayer(nn.Module):
    """Implements pixel-wise feature vector normalization layer."""

    def __init__(self, dim, eps):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def extra_repr(self):
        return f'dim={self.dim}, epsilon={self.eps}'

    def forward(self, x):
        scale = (x.square().mean(dim=self.dim, keepdim=True) + self.eps).rsqrt()
        return x * scale

class UniformBoxWarp(nn.Module):
    def __init__(self, sidelength):
        super().__init__()
        self.scale_factor = 2 / sidelength

    def forward(self, coordinates):
        return coordinates * self.scale_factor

def first_film_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1/num_input, 1/num_input)

def freq_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6/num_input)/freq,
                                  np.sqrt(6/num_input)/freq)
    return init

class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1,
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )