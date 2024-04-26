# python3.8
"""Contains the implementation of generator described in VolumeGAN.

Paper: https://arxiv.org/pdf/2112.10759.pdf
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .stylegan2_generator import MappingNetwork
from .stylegan2_generator import ModulateConvLayer
from .stylegan2_generator import ConvLayer
from .stylegan2_generator import DenseLayer
from third_party.stylegan2_official_ops import upfirdn2d
from .rendering import Renderer
from .rendering import FeatureExtractor
from .utils.ops import all_gather


class VolumeGANGenerator(nn.Module):
    """Defines the generator network in VoumeGAN."""

    def __init__(
        self,
        # Settings for mapping network.
        z_dim=512,
        w_dim=512,
        repeat_w=True,
        normalize_z=True,
        mapping_layers=8,
        mapping_fmaps=512,
        mapping_use_wscale=True,
        mapping_wscale_gain=1.0,
        mapping_lr_mul=0.01,
        # Settings for conditional generation.
        label_dim=0,
        embedding_dim=512,
        embedding_bias=True,
        embedding_use_wscale=True,
        embedding_wscale_gian=1.0,
        embedding_lr_mul=1.0,
        normalize_embedding=True,
        normalize_embedding_latent=False,
        # Settings for post neural renderer network.
        resolution=-1,
        nerf_res=32,
        image_channels=3,
        final_tanh=False,
        demodulate=True,
        use_wscale=True,
        wscale_gain=1.0,
        lr_mul=1.0,
        noise_type='spatial',
        fmaps_base=32 << 10,
        fmaps_max=512,
        filter_kernel=(1, 3, 3, 1),
        conv_clamp=None,
        eps=1e-8,
        rgb_init_res_out=True,
        # Settings for feature volume.
        fv_cfg=dict(feat_res=32,
                    init_res=4,
                    base_channels=256,
                    output_channels=32,
                    w_dim=512),
        # Settings for position encoder.
        embed_cfg=dict(input_dim=3, max_freq_log2=10 - 1, N_freqs=10),
        # Settings for MLP network.
        fg_cfg=dict(num_layers=4, hidden_dim=256, activation_type='lrelu'),
        bg_cfg=None,
        out_dim=512,
        # Settings for rendering.
        rendering_kwargs={}):

        super().__init__()

        self.z_dim = z_dim
        self.w_dim = w_dim
        self.repeat_w = repeat_w
        self.normalize_z = normalize_z
        self.mapping_layers = mapping_layers
        self.mapping_fmaps = mapping_fmaps
        self.mapping_use_wscale = mapping_use_wscale
        self.mapping_wscale_gain = mapping_wscale_gain
        self.mapping_lr_mul = mapping_lr_mul

        self.latent_dim = (z_dim,)
        self.label_size = label_dim
        self.label_dim = label_dim
        self.embedding_dim = embedding_dim
        self.embedding_bias = embedding_bias
        self.embedding_use_wscale = embedding_use_wscale
        self.embedding_wscale_gain = embedding_wscale_gian
        self.embedding_lr_mul = embedding_lr_mul
        self.normalize_embedding = normalize_embedding
        self.normalize_embedding_latent = normalize_embedding_latent

        self.resolution = resolution
        self.nerf_res = nerf_res
        self.image_channels = image_channels
        self.final_tanh = final_tanh
        self.demodulate = demodulate
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.noise_type = noise_type.lower()
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max
        self.filter_kernel = filter_kernel
        self.conv_clamp = conv_clamp
        self.eps = eps

        self.num_nerf_layers = fg_cfg['num_layers']
        self.num_cnn_layers = int(np.log2(resolution // nerf_res * 2)) * 2
        self.num_layers = self.num_nerf_layers + self.num_cnn_layers

        # Set up `w_avg` for truncation trick.
        if self.repeat_w:
            self.register_buffer('w_avg', torch.zeros(w_dim))
        else:
            self.register_buffer('w_avg', torch.zeros(self.num_layers * w_dim))

        # Set up the mapping network.
        self.mapping = MappingNetwork(
            input_dim=z_dim,
            output_dim=w_dim,
            num_outputs=self.num_layers,
            repeat_output=repeat_w,
            normalize_input=normalize_z,
            num_layers=mapping_layers,
            hidden_dim=mapping_fmaps,
            use_wscale=mapping_use_wscale,
            wscale_gain=mapping_wscale_gain,
            lr_mul=mapping_lr_mul,
            label_dim=label_dim,
            embedding_dim=embedding_dim,
            embedding_bias=embedding_bias,
            embedding_use_wscale=embedding_use_wscale,
            embedding_wscale_gian=embedding_wscale_gian,
            embedding_lr_mul=embedding_lr_mul,
            normalize_embedding=normalize_embedding,
            normalize_embedding_latent=normalize_embedding_latent,
            eps=eps)

        # Set up the overall renderer.
        self.renderer = Renderer()

        # Set up the reference representation generator.
        self.ref_representation_generator = FeatureVolume(**fv_cfg)

        # Set up the position encoder.
        self.position_encoder = PositionEncoder(**embed_cfg)

        # Set up the feature extractor.
        self.feature_extractor = FeatureExtractor(ref_mode='feature_volume')

        # Set up the  post module in the feature extractor.
        self.post_module = NeRFMLPNetwork(input_dim=self.position_encoder.out_dim +
                                      fv_cfg['output_channels'],
                                      fg_cfg=fg_cfg,
                                      bg_cfg=bg_cfg)

        # Set up the fully-connected layer head.
        self.fc_head = FCHead(fg_cfg=fg_cfg, bg_cfg=bg_cfg, out_dim=out_dim)

        # Set up the post neural renderer.
        self.post_neural_renderer = PostNeuralRendererNetwork(
            resolution=resolution,
            init_res=nerf_res,
            w_dim=w_dim,
            image_channels=image_channels,
            final_tanh=final_tanh,
            demodulate=demodulate,
            use_wscale=use_wscale,
            wscale_gain=wscale_gain,
            lr_mul=lr_mul,
            noise_type=noise_type,
            fmaps_base=fmaps_base,
            filter_kernel=filter_kernel,
            fmaps_max=fmaps_max,
            conv_clamp=conv_clamp,
            eps=eps,
            rgb_init_res_out=rgb_init_res_out)

        # Set up some rendering related arguments.
        self.rendering_kwargs = rendering_kwargs

        # Set up vars' mapping from current implementation to the official
        # implementation. Note that this is only for debug.
        self.cur_to_official_part_mapping = {
            'w_avg': 'w_avg',
            'mapping': 'mapping',
            'ref_representation_generator': 'nerfmlp.fv',
            'post_module.fg_mlp': 'nerfmlp.fg_mlps',
            'fc_head.fg_sigma_head': 'nerfmlp.fg_density',
            'fc_head.fg_rgb_head': 'nerfmlp.fg_color',
            'post_neural_renderer': 'synthesis'
        }

        # Set debug mode only when debugging.
        if self.rendering_kwargs.get('debug_mode', False):
            self.set_weights_from_official(
                rendering_kwargs.get('cur_state', None),
                rendering_kwargs.get('official_state', None))

    def get_cur_to_official_full_mapping(self, keys_cur):
        cur_to_official_full_mapping = {}
        for key, val in self.cur_to_official_part_mapping.items():
            for key_cur_full in keys_cur:
                if key in key_cur_full:
                    sub_key = key_cur_full.replace(key, '')
                    cur_to_official_full_mapping[key + sub_key] = val + sub_key
        return cur_to_official_full_mapping

    def set_weights_from_official(self, cur_state, official_state):
        keys_cur = cur_state['models']['generator_smooth'].keys()
        self.cur_to_official_full_mapping = (
            self.get_cur_to_official_full_mapping(keys_cur))
        for name, param in self.named_parameters():
            param.data = (official_state['models']['generator_smooth'][
                self.cur_to_official_full_mapping[name]])

    def forward(
            self,
            z,
            label=None,
            lod=None,
            w_moving_decay=None,
            sync_w_avg=False,
            style_mixing_prob=None,
            trunc_psi=None,
            trunc_layers=None,
            noise_mode='const',
            fused_modulate=False,
            impl='cuda',
            fp16_res=None,
    ):
        mapping_results = self.mapping(z, label, impl=impl)
        w = mapping_results['w']
        lod = self.post_neural_renderer.lod.item() if lod is None else lod

        if self.training and w_moving_decay is not None:
            if sync_w_avg:
                batch_w_avg = all_gather(w.detach()).mean(dim=0)
            else:
                batch_w_avg = w.detach().mean(dim=0)
            self.w_avg.copy_(batch_w_avg.lerp(self.w_avg, w_moving_decay))

        wp = mapping_results['wp']

        if self.training and style_mixing_prob is not None:
            if np.random.uniform() < style_mixing_prob:
                new_z = torch.randn_like(z)
                new_wp = self.mapping(new_z, label, impl=impl)['wp']
                current_layers = self.num_layers
                if current_layers > self.num_nerf_layers:
                    mixing_cutoff = np.random.randint(self.num_nerf_layers,
                                                      current_layers)
                    wp[:, mixing_cutoff:] = new_wp[:, mixing_cutoff:]

        if not self.training:
            trunc_psi = 1.0 if trunc_psi is None else trunc_psi
            trunc_layers = 0 if trunc_layers is None else trunc_layers
            if trunc_psi < 1.0 and trunc_layers > 0:
                w_avg = self.w_avg.reshape(1, -1, self.w_dim)[:, :trunc_layers]
                wp[:, :trunc_layers] = w_avg.lerp(
                    wp[:, :trunc_layers], trunc_psi)

        nerf_w = wp[:,:self.num_nerf_layers]
        cnn_w = wp[:,self.num_nerf_layers:]

        feature_volume = self.ref_representation_generator(nerf_w)

        rendering_results = self.renderer(
            wp=nerf_w,
            feature_extractor=self.feature_extractor,
            rendering_options=self.rendering_kwargs,
            position_encoder=self.position_encoder,
            ref_representation=feature_volume,
            post_module=self.post_module,
            fc_head=self.fc_head)

        feature2d = rendering_results['composite_rgb']
        feature2d = feature2d.reshape(feature2d.shape[0], self.nerf_res,
                                      self.nerf_res, -1).permute(0, 3, 1, 2)

        final_results = self.post_neural_renderer(
            feature2d,
            cnn_w,
            lod=None,
            noise_mode=noise_mode,
            fused_modulate=fused_modulate,
            impl=impl,
            fp16_res=fp16_res)

        return {**mapping_results, **final_results}


class PositionEncoder(nn.Module):
    """Implements the class for positional encoding."""

    def __init__(self,
                 input_dim,
                 max_freq_log2,
                 N_freqs,
                 log_sampling=True,
                 include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
        """Initializes with basic settings.

        Args:
            input_dim: Dimension of input to be embedded.
            max_freq_log2: `log2` of max freq; min freq is 1 by default.
            N_freqs: Number of frequency bands.
            log_sampling: If True, frequency bands are linerly sampled in
                log-space.
            include_input: If True, raw input is included in the embedding.
                Defaults to True.
            periodic_fns: Periodic functions used to embed input.
                Defaults to (torch.sin, torch.cos).
        """
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2.**torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2.**0., 2.**max_freq_log2,
                                             N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input):
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)

        return out


class FeatureVolume(nn.Module):
    """Defines feature volume in VolumeGAN."""

    def __init__(self,
                 feat_res=32,
                 init_res=4,
                 base_channels=256,
                 output_channels=32,
                 w_dim=512,
                 **kwargs):
        super().__init__()
        self.num_stages = int(np.log2(feat_res // init_res)) + 1

        self.const = nn.Parameter(
            torch.ones(1, base_channels, init_res, init_res, init_res))
        inplanes = base_channels
        outplanes = base_channels

        self.stage_channels = []
        for i in range(self.num_stages):
            conv = nn.Conv3d(inplanes,
                             outplanes,
                             kernel_size=(3, 3, 3),
                             padding=(1, 1, 1))
            self.stage_channels.append(outplanes)
            self.add_module(f'layer{i}', conv)
            instance_norm = InstanceNormLayer(num_features=outplanes,
                                              affine=False)

            self.add_module(f'instance_norm{i}', instance_norm)
            inplanes = outplanes
            outplanes = max(outplanes // 2, output_channels)
            if i == self.num_stages - 1:
                outplanes = output_channels

        self.mapping_network = nn.Linear(w_dim, sum(self.stage_channels) * 2)
        self.mapping_network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.mapping_network.weight *= 0.25
        self.upsample = UpsamplingLayer()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, w, **kwargs):
        if w.ndim == 3:
            _w = w[:, 0]
        else:
            _w = w
        scale_shifts = self.mapping_network(_w)
        scales = scale_shifts[..., :scale_shifts.shape[-1] // 2]
        shifts = scale_shifts[..., scale_shifts.shape[-1] // 2:]

        x = self.const.repeat(w.shape[0], 1, 1, 1, 1)
        for idx in range(self.num_stages):
            if idx != 0:
                x = self.upsample(x)
            conv_layer = self.__getattr__(f'layer{idx}')
            x = conv_layer(x)
            instance_norm = self.__getattr__(f'instance_norm{idx}')
            scale = scales[:,
                           sum(self.stage_channels[:idx]
                               ):sum(self.stage_channels[:idx + 1])]
            shift = shifts[:,
                           sum(self.stage_channels[:idx]
                               ):sum(self.stage_channels[:idx + 1])]
            scale = scale.view(scale.shape + (1, 1, 1))
            shift = shift.view(shift.shape + (1, 1, 1))
            x = instance_norm(x, weight=scale, bias=shift)
            x = self.lrelu(x)

        return x


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight,
                                      a=0.2,
                                      mode='fan_in',
                                      nonlinearity='leaky_relu')


class InstanceNormLayer(nn.Module):
    """Implements instance normalization layer."""

    def __init__(self, num_features, epsilon=1e-8, affine=False):
        super().__init__()
        self.eps = epsilon
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1, 1))
            self.bias = nn.Parameter(torch.Tensor(1, num_features, 1, 1, 1))
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def forward(self, x, weight=None, bias=None):
        x = x - torch.mean(x, dim=[2, 3, 4], keepdim=True)
        norm = torch.sqrt(
            torch.mean(x**2, dim=[2, 3, 4], keepdim=True) + self.eps)
        x = x / norm
        isnot_input_none = weight is not None and bias is not None
        assert (isnot_input_none and not self.affine) or (not isnot_input_none
                                                          and self.affine)
        if self.affine:
            x = x * self.weight + self.bias
        else:
            x = x * weight + bias
        return x


class UpsamplingLayer(nn.Module):

    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        if self.scale_factor <= 1:
            return x
        return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')


class NeRFMLPNetwork(nn.Module):
    """Defines class of MLP Network described in VolumeGAN.

    Basically, this class takes in latent codes and point coodinates as input,
    and outputs features of each point, which is followed by two fully-connected
    layer heads.
    """

    def __init__(self, input_dim, fg_cfg, bg_cfg=None):
        super().__init__()
        self.fg_mlp = self.build_mlp(input_dim=input_dim, **fg_cfg)

    def build_mlp(self, input_dim, num_layers, hidden_dim, activation_type,
                  **kwargs):
        """Implements function to build the `MLP`.

        Note that here the `MLP` network is consists of a series of
        `ModulateConvLayer` with `kernel_size=1` to simulate fully-connected
        layer. Typically, the input's shape of convolutional layers is
        `[N, C, H, W]`. And the input's shape is `[N, C, R*K, 1]` here, which
        aims to keep consistent with `MLP`.
        """
        default_conv_cfg = dict(resolution=32,
                                w_dim=512,
                                kernel_size=1,
                                add_bias=True,
                                scale_factor=1,
                                filter_kernel=None,
                                demodulate=True,
                                use_wscale=True,
                                wscale_gain=1,
                                lr_mul=1,
                                noise_type='none',
                                conv_clamp=None,
                                eps=1e-8)
        mlp_list = nn.ModuleList()
        in_ch = input_dim
        out_ch = hidden_dim
        for _ in range(num_layers):
            mlp = ModulateConvLayer(in_channels=in_ch,
                                    out_channels=out_ch,
                                    activation_type=activation_type,
                                    **default_conv_cfg)
            mlp_list.append(mlp)
            in_ch = out_ch
            out_ch = hidden_dim

        return mlp_list

    def forward(self,
                pre_point_features,
                wp,
                points_encoding=None,
                fused_modulate=False,
                impl='cuda'):
        N, C, R_K, _ = points_encoding.shape
        x = torch.cat([pre_point_features, points_encoding], dim=1)

        for idx, mlp in enumerate(self.fg_mlp):
            if wp.ndim == 3:
                _w = wp[:, idx]
            else:
                _w = wp
            x, _ = mlp(x, _w, fused_modulate=fused_modulate, impl=impl)

        return x  # x's shape: [N, C, R*K, 1]


class FCHead(nn.Module):
    """Defines fully-connected layer head in VolumeGAN to decode `feature` into
    `sigma` and `rgb`."""

    def __init__(self, fg_cfg, bg_cfg=None, out_dim=512):
        super().__init__()
        self.fg_sigma_head = DenseLayer(in_channels=fg_cfg['hidden_dim'],
                                           out_channels=1,
                                           add_bias=True,
                                           init_bias=0.0,
                                           use_wscale=True,
                                           wscale_gain=1,
                                           lr_mul=1,
                                           activation_type='linear')
        self.fg_rgb_head = DenseLayer(in_channels=fg_cfg['hidden_dim'],
                                             out_channels=out_dim,
                                             add_bias=True,
                                             init_bias=0.0,
                                             use_wscale=True,
                                             wscale_gain=1,
                                             lr_mul=1,
                                             activation_type='linear')

    def forward(self, post_point_features, wp=None, dirs=None):
        post_point_features = rearrange(
            post_point_features, 'N C (R_K) 1 -> (N R_K) C').contiguous()
        fg_sigma = self.fg_sigma_head(post_point_features)
        fg_rgb = self.fg_rgb_head(post_point_features)

        results = {'sigma': fg_sigma, 'rgb': fg_rgb}

        return results


class PostNeuralRendererNetwork(nn.Module):
    """Implements the neural renderer in VolumeGAN to render high-resolution
    images.

    Basically, this network executes several convolutional layers in sequence.
    """

    def __init__(
        self,
        resolution,
        init_res,
        w_dim,
        image_channels,
        final_tanh,
        demodulate,
        use_wscale,
        wscale_gain,
        lr_mul,
        noise_type,
        fmaps_base,
        fmaps_max,
        filter_kernel,
        conv_clamp,
        eps,
        rgb_init_res_out=False,
    ):
        super().__init__()

        self.init_res = init_res
        self.init_res_log2 = int(np.log2(init_res))
        self.resolution = resolution
        self.final_res_log2 = int(np.log2(resolution))
        self.w_dim = w_dim
        self.image_channels = image_channels
        self.final_tanh = final_tanh
        self.demodulate = demodulate
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.noise_type = noise_type.lower()
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max
        self.filter_kernel = filter_kernel
        self.conv_clamp = conv_clamp
        self.eps = eps
        self.rgb_init_res_out = rgb_init_res_out

        self.num_layers = (self.final_res_log2 - self.init_res_log2 + 1) * 2

        self.register_buffer('lod', torch.zeros(()))

        for res_log2 in range(self.init_res_log2, self.final_res_log2 + 1):
            res = 2**res_log2
            in_channels = self.get_nf(res // 2)
            out_channels = self.get_nf(res)
            block_idx = res_log2 - self.init_res_log2

            # Early layer.
            if res > init_res:
                layer_name = f'layer{2 * block_idx - 1}'
                self.add_module(
                    layer_name,
                    ModulateConvLayer(in_channels=in_channels,
                                      out_channels=out_channels,
                                      resolution=res,
                                      w_dim=w_dim,
                                      kernel_size=1,
                                      add_bias=True,
                                      scale_factor=2,
                                      filter_kernel=filter_kernel,
                                      demodulate=demodulate,
                                      use_wscale=use_wscale,
                                      wscale_gain=wscale_gain,
                                      lr_mul=lr_mul,
                                      noise_type=noise_type,
                                      activation_type='lrelu',
                                      conv_clamp=conv_clamp,
                                      eps=eps))
            if block_idx == 0:
                if self.rgb_init_res_out:
                    self.rgb_init_res = ConvLayer(
                        in_channels=out_channels,
                        out_channels=image_channels,
                        kernel_size=1,
                        add_bias=True,
                        scale_factor=1,
                        filter_kernel=None,
                        use_wscale=use_wscale,
                        wscale_gain=wscale_gain,
                        lr_mul=lr_mul,
                        activation_type='linear',
                        conv_clamp=conv_clamp,
                    )
                continue
            # Second layer (kernel 1x1) without upsampling.
            layer_name = f'layer{2 * block_idx}'
            self.add_module(
                layer_name,
                ModulateConvLayer(in_channels=out_channels,
                                  out_channels=out_channels,
                                  resolution=res,
                                  w_dim=w_dim,
                                  kernel_size=1,
                                  add_bias=True,
                                  scale_factor=1,
                                  filter_kernel=None,
                                  demodulate=demodulate,
                                  use_wscale=use_wscale,
                                  wscale_gain=wscale_gain,
                                  lr_mul=lr_mul,
                                  noise_type=noise_type,
                                  activation_type='lrelu',
                                  conv_clamp=conv_clamp,
                                  eps=eps))

            # Output convolution layer for each resolution (if needed).
            layer_name = f'output{block_idx}'
            self.add_module(
                layer_name,
                ModulateConvLayer(in_channels=out_channels,
                                  out_channels=image_channels,
                                  resolution=res,
                                  w_dim=w_dim,
                                  kernel_size=1,
                                  add_bias=True,
                                  scale_factor=1,
                                  filter_kernel=None,
                                  demodulate=False,
                                  use_wscale=use_wscale,
                                  wscale_gain=wscale_gain,
                                  lr_mul=lr_mul,
                                  noise_type='none',
                                  activation_type='linear',
                                  conv_clamp=conv_clamp,
                                  eps=eps))

        # Used for upsampling output images for each resolution block for sum.
        self.register_buffer('filter', upfirdn2d.setup_filter(filter_kernel))

    def get_nf(self, res):
        """Gets number of feature maps according to current resolution."""
        return min(self.fmaps_base // res, self.fmaps_max)

    def set_space_of_latent(self, space_of_latent):
        """Sets the space to which the latent code belong.

        Args:
            space_of_latent: The space to which the latent code belong. Case
                insensitive. Support `W` and `Y`.
        """
        space_of_latent = space_of_latent.upper()
        for module in self.modules():
            if isinstance(module, ModulateConvLayer):
                setattr(module, 'space_of_latent', space_of_latent)

    def forward(self,
                x,
                wp,
                lod=None,
                noise_mode='const',
                fused_modulate=False,
                impl='cuda',
                fp16_res=None,
                nerf_out=False):
        lod = self.lod.item() if lod is None else lod

        results = {}

        # Cast to `torch.float16` if needed.
        if fp16_res is not None and self.init_res >= fp16_res:
            x = x.to(torch.float16)

        for res_log2 in range(self.init_res_log2, self.final_res_log2 + 1):
            cur_lod = self.final_res_log2 - res_log2
            block_idx = res_log2 - self.init_res_log2

            layer_idxs = [2 * block_idx - 1, 2 *
                          block_idx] if block_idx > 0 else [
                              2 * block_idx,
                          ]
            # determine forward until cur resolution
            if lod < cur_lod + 1:
                for layer_idx in layer_idxs:
                    if layer_idx == 0:
                        # image = x[:,:3]
                        if self.rgb_init_res_out:
                            cur_image = self.rgb_init_res(x,
                                                          runtime_gain=1,
                                                          impl=impl)
                        else:
                            cur_image = x[:, :3]
                        continue
                    layer = getattr(self, f'layer{layer_idx}')
                    x, style = layer(
                        x,
                        wp[:, layer_idx],
                        noise_mode=noise_mode,
                        fused_modulate=fused_modulate,
                        impl=impl,
                    )
                    results[f'style{layer_idx}'] = style
                    if layer_idx % 2 == 0:
                        output_layer = getattr(self, f'output{layer_idx // 2}')
                        y, style = output_layer(
                            x,
                            wp[:, layer_idx + 1],
                            fused_modulate=fused_modulate,
                            impl=impl,
                        )
                        results[f'output_style{layer_idx // 2}'] = style
                        if layer_idx == 0:
                            cur_image = y.to(torch.float32)
                        else:
                            if not nerf_out:
                                cur_image = y.to(
                                    torch.float32) + upfirdn2d.upsample2d(
                                        cur_image, self.filter, impl=impl)
                            else:
                                cur_image = y.to(torch.float32) + cur_image

                        # Cast to `torch.float16` if needed.
                        if layer_idx != self.num_layers - 2:
                            res = self.init_res * (2**(layer_idx // 2))
                            if fp16_res is not None and res * 2 >= fp16_res:
                                x = x.to(torch.float16)
                            else:
                                x = x.to(torch.float32)

            # rgb interpolation
            if cur_lod - 1 < lod <= cur_lod:
                image = cur_image
            elif cur_lod < lod < cur_lod + 1:
                alpha = np.ceil(lod) - lod
                image = F.interpolate(image, scale_factor=2, mode='nearest')
                image = cur_image * alpha + image * (1 - alpha)
            elif lod >= cur_lod + 1:
                image = F.interpolate(image, scale_factor=2, mode='nearest')

        if self.final_tanh:
            image = torch.tanh(image)
        results['image'] = image

        return results
