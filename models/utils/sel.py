import torch
import torch.nn as nn
import numpy as np

from third_party.stylegan3_official_ops import bias_act
from third_party.stylegan3_official_ops import upfirdn2d

#----------------------------------------------------------------------------
class SEL(torch.nn.Module):
    def __init__(self, norm_nc, label_nc, hidden_nc=128):
        super().__init__()
        self.norm = nn.InstanceNorm2d(norm_nc, affine=False)
        self.mlp_shared = nn.Conv2d(label_nc, hidden_nc, kernel_size=1, padding=0)
        self.actv = nn.ReLU()
        self.mlp_gamma = nn.Conv2d(hidden_nc, norm_nc, kernel_size=1, padding=0)
        self.mlp_beta = nn.Conv2d(hidden_nc, norm_nc, kernel_size=1, padding=0)

    def forward(self, x, hm):
        x_s = x
        x = self.norm(x)
        hm = F.interpolate(hm, size=x.size()[2:], mode='bilinear', align_corners=True)
        actv = self.actv(self.mlp_shared(hm))
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = x * (1+gamma) + beta

        return out + 0.1 * x_s

class SEL_filt(SEL):
    def __init__(self, norm_nc,  label_nc, hidden_nc=128, down_filter=None, slope=0.2, gain=np.sqrt(2), clamp=None):
        super().__init__(norm_nc, label_nc, hidden_nc)
        self.register_buffer('down_filter', down_filter)
        self.slope = slope
        self.gain = gain
        self.clamp = clamp

    def forward(self, x, hm):
        x_size = x.shape[-1]
        hm_size = hm.shape[-1]
        if x_size != hm_size:
            hm = upfirdn2d.upfirdn2d(x=hm, f=self.down_filter, down=hm_size//x_size, flip_filter=False, padding=int(2.5 * hm_size//x_size))
        hm = self.mlp_shared(hm)
        hm = bias_act.bias_act(x=hm, act='lrelu', alpha=self.slope, gain=self.gain, clamp=self.clamp)
        gamma = self.mlp_gamma(hm)
        beta = self.mlp_beta(hm)

        out = self.norm(x) * (1+gamma) + beta
        return out + 0.1 * x

