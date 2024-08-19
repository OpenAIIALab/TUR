# from torch import nn as nn
# from torch.nn import functional as F
# import torch

# from basicsr.utils.registry import ARCH_REGISTRY
# from .arch_util import ResidualBlockNoBN, default_init_weights, make_layer

# class F_ext(nn.Module):
#     def __init__(self, in_nc=3, nf=64):
#         super(F_ext, self).__init__()
#         stride = 2
#         pad = 0
#         self.pad = nn.ZeroPad2d(1)
#         self.conv1 = nn.Conv2d(in_nc, nf, 7, stride, pad, bias=True)
#         self.conv2 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
#         self.conv3 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
#         self.act = nn.ReLU(inplace=True)

#     def forward(self, x):
#         conv1_out = self.act(self.conv1(self.pad(x)))
#         conv2_out = self.act(self.conv2(self.pad(conv1_out)))
#         conv3_out = self.act(self.conv3(self.pad(conv2_out)))
#         out = torch.mean(conv3_out, dim=[2, 3], keepdim=False)

#         return out

# @ARCH_REGISTRY.register()
# class SRResNet_uncertanty(nn.Module):
#     """Modified SRResNet with uncertainty head.

#     A compacted version modified from SRResNet in
#     "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
#     It uses residual blocks without BN, similar to EDSR.
#     Currently, it supports x2, x3 and x4 upsampling scale factor.

#     Args:
#         num_in_ch (int): Channel number of inputs. Default: 3.
#         num_out_ch (int): Channel number of outputs. Default: 3.
#         num_feat (int): Channel number of intermediate features. Default: 64.
#         num_block (int): Block number in the body network. Default: 16.
#         upscale (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
#     """

#     def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
#         super(SRResNet_uncertanty, self).__init__()
#         self.upscale = upscale

#         self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
#         self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat)

#         self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

#         # Uncertainty head
#         self.uncertainty_head = nn.Conv2d(num_feat, 1, 3, 1, 1)

#         # activation function
#         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

#         # initialization
#         default_init_weights([self.conv_first, self.conv_hr, self.conv_last, self.uncertainty_head], 0.1)
#         if self.upscale == 4:
#             default_init_weights(self.upconv2, 0.1)

#     def forward(self, x):
#         feat = self.lrelu(self.conv_first(x))
#         out = self.body(feat)

#         out_hr = self.conv_last(self.lrelu(self.conv_hr(out)))

#         out_hr += x

#         uncertainty = self.uncertainty_head(out)

#         return out_hr, uncertainty


from torch import nn as nn
from torch.nn import functional as F
import torch

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, default_init_weights, make_layer

class F_ext(nn.Module):
    def __init__(self, in_nc=3, nf=64):
        super(F_ext, self).__init__()
        stride = 2
        pad = 0
        self.pad = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(in_nc, nf, 7, stride, pad, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
        self.conv3 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1_out = self.act(self.conv1(self.pad(x)))
        conv2_out = self.act(self.conv2(self.pad(conv1_out)))
        conv3_out = self.act(self.conv3(self.pad(conv2_out)))
        out = torch.mean(conv3_out, dim=[2, 3], keepdim=False)

        return out

@ARCH_REGISTRY.register()
class SRResNet_uncertanty_ule_1(nn.Module):
    """Modified SRResNet with uncertainty head.

    A compacted version modified from SRResNet in
    "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
    It uses residual blocks without BN, similar to EDSR.
    Currently, it supports x2, x3 and x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(SRResNet_uncertanty_ule_1, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # Uncertainty head
        self.var_conv = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.ELU(),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.ELU(),
            nn.Conv2d(num_feat, num_out_ch, 3, 1, 1),nn.ELU()
        )

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last, self.var_conv], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)

        out_hr = self.conv_last(self.lrelu(self.conv_hr(out)))
        out_hr += x

        var = self.var_conv(F.interpolate(out, scale_factor=self.upscale, mode='nearest'))

        return out_hr, var

@ARCH_REGISTRY.register()
class SRResNet_uncertanty_ule_8(nn.Module):
    """Modified SRResNet with uncertainty head.

    A compacted version modified from SRResNet in
    "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
    It uses residual blocks without BN, similar to EDSR.
    Currently, it supports x2, x3 and x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(SRResNet_uncertanty_ule_8, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # Uncertainty head
        self.var_conv = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.ELU(),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.ELU(),
            nn.Conv2d(num_feat, num_out_ch, 3, 1, 1),nn.ELU(alpha=8.0)
        )

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last, self.var_conv], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)

        out_hr = self.conv_last(self.lrelu(self.conv_hr(out)))
        out_hr += x

        var = self.var_conv(F.interpolate(out, scale_factor=self.upscale, mode='nearest'))

        return out_hr, var

@ARCH_REGISTRY.register()
class SRResNet_uncertanty_softplus(nn.Module):
    """Modified SRResNet with uncertainty head.

    A compacted version modified from SRResNet in
    "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
    It uses residual blocks without BN, similar to EDSR.
    Currently, it supports x2, x3 and x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(SRResNet_uncertanty_softplus, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # Uncertainty head
        self.var_conv = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.ELU(),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.ELU(),
            nn.Conv2d(num_feat, num_out_ch, 3, 1, 1),nn.Softplus()
        )

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last, self.var_conv], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)

        out_hr = self.conv_last(self.lrelu(self.conv_hr(out)))
        out_hr += x

        var = self.var_conv(F.interpolate(out, scale_factor=self.upscale, mode='nearest'))

        return out_hr, var