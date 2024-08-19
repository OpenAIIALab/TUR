from torch import nn as nn
from torch.nn import functional as F
import torch

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import ResidualBlockNoBN, default_init_weights, make_layer




class ResidualBlockNoBN_DynaMic(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN_DynaMic, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        for i in range(7):
            setattr(self, f'single_conv1_{i+1}', nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True))
            setattr(self, f'single_conv2_{i+1}', nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True))

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

        self.gate_net = nn.Linear(num_feat, 7)
    
        for n, p in self.named_parameters():
            if 'single_conv' in n:
                p.requires_grad= False

    def forward(self, x):
        identity = x
        B, C, H, W = x.shape
        mapping = self.gate_net(x.mean(dim=[-2, -1])) # BxK
        gate_weight = F.softmax(mapping, dim=-1)

        x = x.reshape(1, B*C, H, W)
        kernel_weight1 = torch.stack([getattr(self, f'single_conv1_{i+1}').weight.data for i in range(7)]).reshape(7, -1) # Kxoxixk1xk2
        kernel_weight2 = torch.stack([getattr(self, f'single_conv2_{i+1}').weight.data for i in range(7)]).reshape(7, -1)# Kxoxixk1xk2

        kernel_bias1 = torch.stack([getattr(self, f'single_conv1_{i+1}').bias.data for i in range(7)]).reshape(7, -1) # Kxoxixk1xk2
        kernel_bias2 = torch.stack([getattr(self, f'single_conv2_{i+1}').bias.data for i in range(7)]).reshape(7, -1) # Kxoxixk1xk2

        kernel_weight1 = gate_weight @ kernel_weight1 + self.conv1.weight.reshape(1, -1)
        kernel_bias1 = (gate_weight @ kernel_bias1) + self.conv1.bias.reshape(1, -1)
        
        kernel_weight2 = gate_weight @ kernel_weight2 + self.conv2.weight.reshape(1, -1)
        kernel_bias2 = (gate_weight @ kernel_bias2) + self.conv2.bias.reshape(1, -1)

        out1 = F.conv2d(x, kernel_weight1.reshape(B*C, C, 3, 3), groups=B, bias=kernel_bias1.reshape(-1), padding=1)
        out1 = F.relu(out1)
        out2 = F.conv2d(out1, kernel_weight2.reshape(B*C, C, 3, 3), groups=B, bias=kernel_bias2.reshape(-1), padding=1)

        out = out2.reshape(B, C, H, W)

        return identity + out * self.res_scale


@ARCH_REGISTRY.register()
class SRResNet_MergeAll(nn.Module):
    """Modified SRResNet.

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
        super(SRResNet_MergeAll, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN_DynaMic, num_block, num_feat=num_feat)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)

        out = self.conv_last(self.lrelu(self.conv_hr(out)))

        out += x
        return out
    
if __name__ == '__main__':
    model = SRResNet_MergeAll(upscale=1, num_feat=64)
    model_dict = torch.load('SRResNet_merge.pth')
    missing = model.load_state_dict(model_dict, strict=False)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)