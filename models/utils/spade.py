import re
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

from models.utils.batchnorm import SynchronizedBatchNorm2d

class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, norm_G, semantic_nc):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, semantic_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

class SPADEGenerator(BaseNetwork):
    def __init__(self, z_dim, semantic_nc, ngf, dim_seq, bev_grid_size, aspect_ratio, 
                    num_upsampling_layers, not_use_vae, norm_G):
        super().__init__()
        nf = ngf
        self.not_use_vae = not_use_vae
        self.z_dim = z_dim
        self.ngf = ngf
        self.dim_seq = list(map(int, dim_seq.split(',')))
        self.num_upsampling_layers = num_upsampling_layers

        self.sw, self.sh = self.compute_latent_vector_size(num_upsampling_layers, bev_grid_size, aspect_ratio)

        if not not_use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(z_dim, self.dim_seq[0] * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(semantic_nc, self.dim_seq[0] * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(self.dim_seq[0] * nf, self.dim_seq[0] * nf, norm_G, semantic_nc)

        self.G_middle_0 = SPADEResnetBlock(self.dim_seq[0] * nf, self.dim_seq[0] * nf, norm_G, semantic_nc)
        self.G_middle_1 = SPADEResnetBlock(self.dim_seq[0] * nf, self.dim_seq[0] * nf, norm_G, semantic_nc)

        self.up_0 = SPADEResnetBlock(self.dim_seq[0] * nf, self.dim_seq[1] * nf, norm_G, semantic_nc)
        self.up_1 = SPADEResnetBlock(self.dim_seq[1] * nf, self.dim_seq[2] * nf, norm_G, semantic_nc)
        self.up_2 = SPADEResnetBlock(self.dim_seq[2] * nf, self.dim_seq[3] * nf, norm_G, semantic_nc)
        self.up_3 = SPADEResnetBlock(self.dim_seq[3] * nf, self.dim_seq[4] * nf, norm_G, semantic_nc)

        final_nc = nf * self.dim_seq[4]

        if num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(self.dim_seq[4] * nf, nf // 2, norm_G, semantic_nc)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 32, 3, padding=1)
        # self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, num_upsampling_layers, bev_grid_size, aspect_ratio):
        if num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif num_upsampling_layers == 'more':
            num_up_layers = 6
        elif num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('num_upsampling_layers [%s] not recognized' %
                             num_upsampling_layers)

        sw = bev_grid_size // (2**num_up_layers)
        sh = round(sw / aspect_ratio)

        return sw, sh

    def forward(self, input, z=None):
        seg = input

        if not self.not_use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, self.dim_seq[0] * self.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.num_upsampling_layers == 'more' or \
           self.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)

        # TODO: Wtf is this leaky relu
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        # x = torch.tanh(x)

        return x

if __name__ == '__main__':
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument('--z_dim', type=int, default=10)
   parser.add_argument('--semantic_nc', type=int, default=10)
   parser.add_argument('--ngf', type=int, default=64)
   parser.add_argument('--bev_grid_size', type=int, default=512)
   parser.add_argument('--aspect_ratio', type=float, default=1.0)
   parser.add_argument('--num_upsampling_layers', type=str, default='more')
   parser.add_argument('--not_use_vae', action="store_true")
   parser.add_argument('--norm_G', type=str, default='spectralspadesyncbatch3x3', help='instance normalization or batch normalization')

   args = parser.parse_args()
   sg = SPADEGenerator(args).cuda()
   seg = torch.zeros([2, 10, 5, 5]).cuda()
   while 1:
       import pdb;pdb.set_trace()
       out = sg(seg)
