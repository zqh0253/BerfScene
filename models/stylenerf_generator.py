# python3.8
"""Contains the implementation of generator described in StyleNeRF."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from einops import rearrange

from utils import eg3d_misc as misc
from models.utils.official_stylegan2_model_helper import modulated_conv2d
from third_party.stylegan2_official_ops import upfirdn2d
from third_party.stylegan2_official_ops import bias_act
from models.utils.official_stylegan2_model_helper import FullyConnectedLayer
from models.utils.official_stylegan2_model_helper import MappingNetwork
from models.utils.official_stylegan2_model_helper import SynthesisBlock
from models.utils.official_stylegan2_model_helper import ToRGBLayer
from models.utils.official_stylegan2_model_helper import Conv2dLayer
from models.rendering import Renderer
from models.rendering import FeatureExtractor
from models.volumegan_generator import PositionEncoder


class StyleNeRFGenerator(nn.Module):
    """Defines the generator network in StyleNeRF."""

    def __init__(self, ):
        super().__init__()

        # Set up mapping network.
        self.mapping = MappingNetwork()  ### TODO: Accomplish filling kwargs.

        # Set up overall Renderer.
        self.renderer = Renderer()

        # Set up the position encoder.
        self.position_encoder = PositionEncoder() ### TODO: Accomplish filling kwargs.

        # Set up the feature extractor.
        self.feature_extractor = FeatureExtractor(ref_mode='none')

        # Set up the  post module in the feature extractor.
        self.post_module = NeRFMLPNetwork()  ### TODO: Accomplish filling kwargs.

        # Set up the fully-connected layer head.
        self.fc_head = FCHead()  ### TODO: Accomplish filling kwargs.

        # Set up the post neural renderer.
        self.post_neural_renderer = PostNeuralRendererNetwork() ### TODO: Accomplish filling kwargs.

    def forward(self,):
        pass


class NeRFMLPNetwork(nn.Module):
    """Defines class of FOREGROUND/BACKGROUND NeRF MLP Network in StyleNeRF.

    Basically, this module consists of several `Style2Layer`s where convolutions
    with 1x1 kernel are involved. Note that this module is not strictly
    equivalent to MLP. Since 1x1 convolution is equal to fully-connected layer,
    we name this module `NeRFMLPNetwork`. Besides, our `NeRFMLPNetwork` takes in
    sampled points, view directions, latent codes as input, and outputs features
    for the following computation of `sigma` and `rgb`.
    """

    def __init__(
        self,
        # dimensions
        input_dim=60,
        w_dim=512,  # style latent
        hidden_size=128,
        n_blocks=8,
        # architecture settings
        activation='lrelu',
        use_skip=False,
        nerf_kwargs={}
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.w_dim = w_dim
        self.activation = activation
        self.n_blocks = n_blocks
        self.use_skip = use_skip

        for key in nerf_kwargs:
            setattr(self, key, nerf_kwargs[key])

        self.fc_in = Style2Layer(self.input_dim,
                                 self.hidden_size,
                                 self.w_dim,
                                 activation=self.activation)
        self.num_wp = 1
        self.skip_layer = self.n_blocks // 2 - 1 if self.use_skip else None
        if self.n_blocks > 1:
            self.blocks = nn.ModuleList([
                Style2Layer(self.hidden_size if i != self.skip_layer else
                            self.hidden_size + self.input_dim,
                            self.hidden_size,
                            w_dim,
                            activation=self.activation,
                            magnitude_ema_beta=self.magnitude_ema_beta)
                for i in range(self.n_blocks - 1)
            ])
            self.num_wp += (self.n_blocks - 1)

    def forward(self,
                pre_point_features,
                points_encoding,
                wp=None,
                use_both=False):
        input_p = points_encoding
        if use_both:
            input_p = torch.cat([pre_point_features, input_p], 1)
        out = self.fc_in(points_encoding, wp[:, 0] if wp is not None else None)
        if self.n_blocks > 1:
            for idx, layer in enumerate(self.blocks):
                wp_i = wp[:, idx + 1] if wp is not None else None
                if (self.skip_layer is not None) and (idx == self.skip_layer):
                    out = torch.cat([out, input_p], 1)
                out = layer(out, wp_i, up=1)
        return out


class FCHead(nn.Module):
    """Defines the fully connnected layer head in StyleNeRF.

    Basically, this module is composed of several `ToRGBLayer`s and
    `Conv2dLayer`s where all convolutions are with kernel size 1x1, in order to
    decode the common feature of each point to the sigma (feature) and
    rgb (feature). Note that this module is not strictly equivalent to the fully
    connnected layer. Since 1x1 convolution is equal to fully-connected layer,
    we name this module `FCHead`.
    """

    def __init__(self,
                 in_dim=128,
                 w_dim=512,
                 w_idx=8,
                 sigma_out_dim=1,
                 rgb_out_dim=256,
                 img_channels=3,
                 predict_rgb=True):
        super().__init__()
        self.predict_rgb = predict_rgb
        self.w_idx = w_idx
        self.sigma_head = ToRGBLayer(in_dim,
                                     sigma_out_dim,
                                     w_dim,
                                     kernel_size=1)
        self.rgb_head = ToRGBLayer(in_dim, rgb_out_dim, w_dim, kernel_size=1)
        # Predict RGB over features.
        if self.predict_rgb:
            self.to_rgb = Conv2dLayer(rgb_out_dim,
                                      img_channels,
                                      kernel_size=1,
                                      activation='linear')

    def forward(self,
                post_point_features,
                wp=None,
                dirs=None,
                height=None,
                width=None):
        assert (height is not None) and (width is not None)
        # TODO: Check shape.
        post_point_features = rearrange(post_point_features,
                                        'N C R_K 1 -> N C R K',
                                        R=height * width)
        post_point_features = rearrange(post_point_features,
                                        'N C R K -> (N K) C H W',
                                        H=height,
                                        W=width)

        sigma = self.sigma_head(post_point_features, wp[:, self.w_idx])
        rgb_feat = self.rgb_head(post_point_features, wp[:, -1])
        rgb = self.to_rgb(post_point_features)
        rgb_feat = torch.cat([rgb_feat, rgb], dim=1)

        results = {'sigma': sigma, 'rgb': rgb_feat}

        return results


class PostNeuralRendererNetwork(nn.Module):
    """Implements the post neural renderer network in StyleNeRF to renderer
    high-resolution images.

    Basically, this module comprises several `SynthesisBlock` with respect to
    different resolutions, which is analogous to StyleGAN2 architecure, and it
    is trained progressively during training. Besides, it is called `Upsampler`
    in the official implemetation.
    """

    no_2d_renderer   = False
    block_reses      = None
    upsample_type    = 'default'
    img_channels     = 3
    in_res           = 32
    out_res          = 512
    channel_base     = 1
    channel_base_sz  = None     # usually 32768, which equals 2 ** 15.
    channel_max      = 512
    channel_dict     = None
    out_channel_dict = None

    def __init__(self, upsampler_kwargs, **other_kwargs):
        super().__init__()
        for key in other_kwargs:
            if hasattr(self, key) and (key not in upsampler_kwargs):
                setattr(upsampler_kwargs, key, other_kwargs[key])
        for key in upsampler_kwargs:
            if hasattr(self, key):
                setattr(self, key, upsampler_kwargs[key])

        self.out_res_log2 = int(np.log2(self.out_res))

        # Set up resolution of blocks.
        if self.block_reses is None:
            self.block_resolutions = [
                2**i for i in range(2, self.out_res_log2 + 1)
            ]
            self.block_resolutions = [
                res for res in self.block_resolutions if res > self.in_res
            ]
        else:
            self.block_resolutions = self.block_reses

        if self.no_2d_renderer:
            self.block_resolutions = []

    def build_network(self, w_dim, in_dim, **block_kwargs):
        networks = []
        if len(self.block_resolutions) == 0:
            return networks

        channel_base = int(
            self.channel_base * 32768
        ) if self.channel_base_sz is None else self.channel_base_sz

        # Don't use fp16 for the first block.
        fp16_resolution = self.block_resolutions[0] * 2

        if self.channel_dict is None:
            channel_dict = {
                res: min(channel_base // res, self.channel_max)
                for res in self.block_resolutions
            }
        else:
            channel_dict = self.channel_dict

        if self.out_channel_dict is None:
            img_channels = self.out_channel_dict
        else:
            img_channels = {
                res: self.img_channels
                for res in self.block_resolutions
            }

        for idx, res in enumerate(self.block_resolutions):
            res_before = self.block_resolutions[idx - 1] if idx > 0 else self.in_res
            in_channels = channel_dict[res_before] if idx > 0 else in_dim
            out_channels = channel_dict[res]
            use_fp16 = (res > fp16_resolution)
            is_last = (idx == (len(self.block_resolutions) - 1))
            block = SynthesisBlock(in_channels=in_channels,
                    out_channels=out_channels,
                    w_dim=w_dim,
                    resolution=res,
                    img_channels=img_channels[res],
                    is_last=is_last,
                    use_fp16=use_fp16,
                    **block_kwargs)  # TODO: Check the kwargs of `SynthesisBlock`, and add `upsample_mode` in our `SynthesisBlock`
            networks += [
                {'block': block,
                 'num_wp': block.num_conv if not is_last else block.num_conv + block.num_torgb,
                 'name': f'b{res}' if res_before != res else f'b{res}_l{idx}'}
            ]
        self.num_wp = sum(net['num_wp'] for net in networks)

        return networks

    def split_wp(self, wp, blocks):
        block_wp = []
        w_idx = 0
        for idx, _ in enumerate(self.block_resolutions):
            block = blocks[idx]
            block_wp.append(
                wp.narrow(1, w_idx, block.num_conv + block.num_torgb))
            w_idx = w_idx + block.num_conv
        return block_wp

    def forward(self, blocks, block_wp, x, image, target_res):
        images = []
        for idx, (res,
                  cur_wp) in enumerate(zip(self.block_resolutions, block_wp)):
            if res > target_res:
                break

            block = blocks[idx]
            x, image = block(x, image, cur_wp)   # TODO: Check whether use noise here.

            images.append(image)

        return images


class Style2Layer(nn.Module):
    """Defines the class of simplified `SynthesisLayer` used in NeRF block with
    the following modifications:

    - No noise injection;
    - Kernel size set to be 1x1.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            w_dim,
            activation='lrelu',
            resample_filter=[1, 3, 3, 1],
            magnitude_ema_beta=-1,  # -1 means not using magnitude ema
            **unused_kwargs):

        super().__init__()
        self.activation = activation
        self.conv_clamp = None
        self.register_buffer('resample_filter',
                             upfirdn2d.setup_filter(resample_filter))
        self.padding = 0
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.w_dim = w_dim
        self.in_features = in_channels
        self.out_features = out_channels
        memory_format = torch.contiguous_format

        if w_dim > 0:
            self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
            self.weight = torch.nn.Parameter(
                torch.randn([out_channels, in_channels, 1,
                             1]).to(memory_format=memory_format))
            self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

        else:
            self.weight = torch.nn.Parameter(
                torch.Tensor(out_channels, in_channels))
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
            self.weight_gain = 1.

            # Initialization.
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        self.magnitude_ema_beta = magnitude_ema_beta
        if magnitude_ema_beta > 0:
            self.register_buffer('w_avg', torch.ones([]))

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, style={}'.format(
            self.in_features, self.out_features, self.w_dim)

    def forward(self,
                x,
                w=None,
                fused_modconv=None,
                gain=1,
                up=1,
                **unused_kwargs):
        flip_weight = True
        act = self.activation

        if (self.magnitude_ema_beta > 0):
            if self.training:  # updating EMA.
                with torch.autograd.profiler.record_function(
                        'update_magnitude_ema'):
                    magnitude_cur = x.detach().to(
                        torch.float32).square().mean()
                    self.w_avg.copy_(
                        magnitude_cur.lerp(self.w_avg,
                                           self.magnitude_ema_beta))
            input_gain = self.w_avg.rsqrt()
            x = x * input_gain

        if fused_modconv is None:
            with misc.suppress_tracer_warnings():
                # this value will be treated as a constant
                fused_modconv = not self.training

        if self.w_dim > 0:  # modulated convolution
            assert x.ndim == 4, "currently not support modulated MLP"
            styles = self.affine(w)  # Batch x style_dim
            if x.size(0) > styles.size(0):
                styles = repeat(styles,
                                'b c -> (b s) c',
                                s=x.size(0) // styles.size(0))

            x = modulated_conv2d(x=x,
                                 weight=self.weight,
                                 styles=styles,
                                 noise=None,
                                 up=up,
                                 padding=self.padding,
                                 resample_filter=self.resample_filter,
                                 flip_weight=flip_weight,
                                 fused_modconv=fused_modconv)
            act_gain = self.act_gain * gain
            act_clamp = (self.conv_clamp *
                         gain if self.conv_clamp is not None else None)
            x = bias_act.bias_act(x,
                                  self.bias.to(x.dtype),
                                  act=act,
                                  gain=act_gain,
                                  clamp=act_clamp)

        else:
            if x.ndim == 2:  # MLP mode
                x = F.relu(F.linear(x, self.weight, self.bias.to(x.dtype)))
            else:
                x = F.relu(
                    F.conv2d(x, self.weight[:, :, None, None], self.bias))
        return x