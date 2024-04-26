import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.signal
from .blurpool import BlurPool
from .sel import SEL, SEL_filt
from .official_stylegan3_model_helper import MappingNetwork, FullyConnectedLayer, modulated_conv2d, SynthesisInput
from third_party.stylegan3_official_ops import filtered_lrelu
from third_party.stylegan3_official_ops import upfirdn2d
from third_party.stylegan3_official_ops import bias_act

class UNetBlock(nn.Module):
    def __init__(self, w_dim, in_channel, latent_channel, out_channel, ks=3, layer_num=2):
        super().__init__() 
        self.ks = ks
        
        self.layer_num = layer_num
        self.weight1 = nn.Parameter(torch.randn([latent_channel, in_channel, ks, ks]))
        self.weight2 = nn.Parameter(torch.randn([out_channel, latent_channel, ks, ks]))
        self.bias1 = nn.Parameter(torch.zeros([latent_channel]))
        self.bias2 = nn.Parameter(torch.zeros([out_channel]))
        self.affine1 = FullyConnectedLayer(w_dim, in_channel, bias_init=1)
        self.affine2 = FullyConnectedLayer(w_dim, latent_channel, bias_init=1)

        if self.layer_num == 3:
            self.weight_mid = nn.Parameter(torch.randn([latent_channel, latent_channel, ks, ks]))
            self.bias_mid = nn.Parameter(torch.zeros([latent_channel]))
            self.affine_mid = FullyConnectedLayer(w_dim, latent_channel, bias_init=1)

    def forward(self, x, *w):
        s1 = self.affine1(w[0])
        x = modulated_conv2d(x, w=self.weight1, s=s1, padding=self.ks//2)
        x = bias_act.bias_act(x, self.bias1.to(x.dtype), act='lrelu')

        if self.layer_num == 3:
            s_mid = self.affine_mid(w[1])
            x = modulated_conv2d(x, w=self.weight_mid, s=s_mid, padding=self.ks//2)
            x = bias_act.bias_act(x, self.bias_mid.to(x.dtype), act='lrelu')
            w_index = 2
        else:
            w_index = 1

        s2 = self.affine2(w[w_index])
        x = modulated_conv2d(x, w=self.weight2, s=s2, padding=self.ks//2)
        x = bias_act.bias_act(x, self.bias2.to(x.dtype), act='lrelu')

        return x

class UNet(nn.Module):
    def __init__(self, w_dim, in_dim=3, base_dim=64, ks=3, block_num=3, layer_num=2, filt_size=3, output_dim=3, label_nc=14, sel_type='normal', img_resolution=256, wo_transform = False,):
        super().__init__()

        self.block_num = block_num
        self.layer_num = layer_num

        self.sel_type = sel_type
        if self.sel_type == 'normal':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        else:
            for i in range(block_num):
                self.register_buffer(f'down_filter_{i}', self.design_lowpass_filter(numtaps=12, cutoff=2**((block_num-i+1)/2), width=None, fs=img_resolution//(2**i)))
                self.register_buffer(f'sel_down_filter_{i}',  self.design_lowpass_filter(numtaps=6*2**i, cutoff=2**((block_num-i+2)/2), width=None, fs=img_resolution//(2**(i-1))))

        self.input = SynthesisInput(w_dim=w_dim, channels=in_dim, size=img_resolution, sampling_rate=img_resolution, bound_len=0, bandwidth=4, wo_transform=wo_transform) 

        encoder_list, sel_enc_list, sel_dec_list, decoder_list, bp_list = [], [], [], [], []

        for i in range(block_num):
            if i == 0:
                encoder_list.append(UNetBlock(w_dim, in_dim, base_dim, base_dim, layer_num=layer_num))
            else:
                encoder_list.append(UNetBlock(w_dim, base_dim * 2 ** (i-1), base_dim * 2 ** i, base_dim * 2 ** i, layer_num=layer_num))

            decoder_list.append(UNetBlock(w_dim, 
                                          base_dim * 2 ** (block_num-i), 
                                          base_dim * 2 ** (block_num-i-1), 
                                          base_dim * 2 ** (block_num-i-2) if i < block_num-1 else base_dim * 2 ** (block_num-i-1),
                                          layer_num=layer_num
                                         ))

            if self.sel_type == 'normal':
                sel_enc_list.append(SEL(in_dim if i==0 else base_dim * 2 ** (i-1), label_nc))
                sel_dec_list.append(SEL(base_dim * 2 ** (block_num-i-1), label_nc))
            else:
                sel_enc_list.append(SEL_filt(in_dim if i==0 else base_dim * 2 ** (i-1), label_nc, down_filter=getattr(self, f'sel_down_filter_{i}')))
                sel_dec_list.append(SEL_filt(base_dim * 2 ** (block_num-i-1), label_nc, down_filter=getattr(self, f'sel_down_filter_{block_num-i-1}')))

        self.encoders = nn.ModuleList(encoder_list)
        self.decoders = nn.ModuleList(decoder_list)
        self.enc_sels = nn.ModuleList(sel_enc_list)
        self.dec_sels = nn.ModuleList(sel_dec_list)

        self.torgb = UNetBlock(w_dim, base_dim, base_dim, output_dim)

    @staticmethod
    def design_lowpass_filter(numtaps, cutoff, fs, width=None):
        if numtaps == 1:
            return None
        f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
        return torch.as_tensor(f, dtype=torch.float32)

    def forward(self, ws, heatmap, **kwargs):
        ws = ws.unbind(1)
        x = self.input(ws[0])
        ws = ws[1:]

        enc_x = []
        for i in range(self.block_num):
            # modulate with SEL
            x = self.enc_sels[i] (x, heatmap)

            if self.layer_num==2:
                x = self.encoders[i] (x, ws[2*i], ws[2*i+1])
            else:
                x = self.encoders[i] (x, ws[3*i], ws[3*i+1], ws[3*i+2])

            enc_x.append(x)
            if self.sel_type == 'normal':
                x = self.pool(x)
            else:
                x = upfirdn2d.upfirdn2d(x=x, f=getattr(self, f'down_filter_{i}'), down=2, flip_filter=False, padding=5)

        ws = ws[self.layer_num*self.block_num:]
        for i in range(self.block_num):
            x = F.interpolate(x, size=x.shape[-1] * 2, mode='bilinear', align_corners=False)
            x = self.dec_sels[i](x, heatmap)
            x = self.decoders[i](torch.cat([x, enc_x[-1-i]], 1), *ws[self.layer_num*i:self.layer_num*(i+1)])

        ws = ws[self.layer_num*self.block_num:]
        x = self.torgb(x, ws[0], ws[1])
        return x

class Generator(nn.Module):
    def __init__(self, z_dim, c_dim, w_dim, img_resolution=256, img_channels=3,
                       in_dim=3,  base_dim=64, ks=3, block_num=3, layer_num=2, filt_size=3, output_dim=3, label_nc=14, sel_type='normal', wo_transform=False, **kwargs):
        super().__init__()
        print(kwargs)
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=2*layer_num*block_num+3)
        self.synthesis = UNet(w_dim=w_dim, in_dim=in_dim, base_dim=base_dim, ks=ks, block_num=block_num, layer_num=layer_num, filt_size=filt_size, output_dim=img_channels, label_nc=label_nc, sel_type=sel_type, img_resolution=img_resolution, wo_transform=wo_transform)

    def forward(self, z, c, heatmap, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        ret = self.synthesis(ws, heatmap=heatmap)

        return ret


if __name__ == '__main__':
    g = Generator(z_dim=64, c_dim=0, w_dim=512, block_num=4,layer_num=3, img_resolution=512, img_channels=32, sel_type='abn')
    hm = torch.ones([10, 14, 512, 512])
    z = torch.zeros([10, 64])
    c = None
    opt = g(z, c, hm)
    g = Generator(z_dim=64, c_dim=0, w_dim=512, block_num=4,layer_num=3, img_resolution=256, img_channels=32, sel_type='abn')
    hm = torch.ones([10, 14,256,256])
    z = torch.zeros([10, 64])
    c = None
    opt = g(z, c, hm)
