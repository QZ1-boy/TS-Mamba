'''
 Copyright 2023 xtudbxk
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''


import torch
from torch import nn as nn
from torch.nn import functional as F
import math
from functools import reduce
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, make_layer
import math
import logging
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from .tmpalign_util import TMPAlign
# from .spynet_arch import SpyNet
from spatial_correlation_sampler import SpatialCorrelationSampler
from basicsr.ops.msda import SingleScaleDeformAttnV1, SingleScaleDeformAttnV2, SingleScaleDeformAttnV3



# @ARCH_REGISTRY.register()
# class BasicUniVSRFeatPropWithFastFlowDeformAtt_Fast_V2(nn.Module):
#     """Online VSR with Flow Guided Deformable Alignment and ResidualNoBN Reconstuction Branch.

#     Args:
#         num_feat (int): Number of channels. Default: 64.
#         num_block (int): Number of residual blocks for each branch. Default: 15
#         spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
#     """

#     def __init__(self,
#                  num_feat=64,
#                  num_extract_block=0,
#                  num_block=15,
#                  num_levels=1,
#                  num_heads=8,
#                  num_points=4,
#                  max_residue_magnitude=10,
#                  flownet_path=None,
#                  return_flow=False,
#                  one_stage_up=False):
#         super().__init__()
#         self.num_feat = num_feat
#         self.return_flow = return_flow
#         self.one_stage_up = one_stage_up

#         # feature extraction
#         self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

#         # alignment
#         self.flownet = FastFlowNet(groups=3, load_path=flownet_path)
#         # AttnAlignment
#         self.flow_guided_dcn = FlowGuidedDeformAttnAlignV2(d_model=num_feat,
#                                                            n_levels=num_levels,
#                                                            n_heads=num_heads,
#                                                            n_points=num_points,
#                                                            max_residue_magnitude=max_residue_magnitude)

#         # propagation
#         self.forward_trunk = ConvResidualBlocks(2 * num_feat, num_feat, num_block)

#         # reconstruction
#         if self.one_stage_up:
#             self.upconv = nn.Conv2d(num_feat, 3 * 16, 3, 1, 1, bias=True)
#             self.pixel_shuffle = nn.PixelShuffle(4)
#         else:
#             self.upconv1 = nn.Conv2d(num_feat, 16 * 4, 3, 1, 1, bias=True)
#             self.upconv2 = nn.Conv2d(16, 16 * 4, 3, 1, 1, bias=True)
#             self.conv_last = nn.Conv2d(16, 3, 3, 1, 1)
#             self.pixel_shuffle = nn.PixelShuffle(2)

#         # activation functions
#         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

#     def get_flow(self, x):
#         b, n, c, h, w = x.size()

#         x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
#         x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

#         flows_forward = self.flownet(x_2, x_1).view(b, n - 1, 2, h, w)

#         return flows_forward

#     def forward(self, x):
#         """Forward function of BasicVSR.

#         Args:
#             x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
#         """
#         flows_forward = self.get_flow(x)
#         b, n, c, h, w = x.size()

#         feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

#         # backward branch
#         out_l = []

#         # forward branch
#         feat_prop = x.new_zeros(b, self.num_feat, h, w)
#         for i in range(0, n):
#             x_i = x[:, i, :, :, :]
#             feat_curr = feat[:, i, :, :, :]
#             if i > 0:
#                 flow = flows_forward[:, i - 1, :, :, :]
#                 extra_feat = torch.cat([feat_curr, flow_warp(feat_prop, flow.permute(0, 2, 3, 1))], dim=1)
#                 feat_prop = self.flow_guided_dcn(feat_prop, extra_feat, flow)

#             feat_prop = torch.cat([feat_curr, feat_prop], dim=1)
#             feat_prop = self.forward_trunk(feat_prop)

#             # upsample
#             out = feat_prop
#             if self.one_stage_up:
#                 out = self.pixel_shuffle(self.upconv(out))
#             else:
#                 out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
#                 out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
#                 out = self.conv_last(out)
#             base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
#             out += base
#             out_l.append(out)

#         if self.return_flow:
#             return torch.stack(out_l, dim=1), flows_forward
#         else:
#             return torch.stack(out_l, dim=1)

##### V4
@ARCH_REGISTRY.register()
class FDAN(nn.Module):
    """Online VSR with Flow Guided Deformable Alignment and ResidualNoBN Reconstuction Branch.
    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self,
                 num_feat=64,
                 num_extract_block=0,
                 num_block=15,
                 num_levels=1,
                 num_heads=8,
                 num_points=4,
                 max_residue_magnitude=10,
                 flownet_path=None,
                 return_flow=False,
                 one_stage_up=False):
        super().__init__()
        self.num_feat = num_feat
        self.return_flow = return_flow
        self.one_stage_up = one_stage_up

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

        # alignment
        self.flownet = FastFlowNet(groups=3, load_path='/share3/home/zqiang/TMP/experiments/pretrained_models/fastflownet_ft_mix.pth')
        self.flow_guided_dcn = FlowGuidedDeformAttnAlignV4(d_model=num_feat,
                                                           n_levels=num_levels,
                                                           n_heads=num_heads,
                                                           n_points=num_points,
                                                           max_residue_magnitude=max_residue_magnitude)

        # propagation
        self.forward_trunk = ConvResidualBlocks(2 * num_feat, num_feat, num_block)

        # reconstruction
        if self.one_stage_up:
            self.upconv = nn.Conv2d(num_feat, 3 * 16, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(4)
        else:
            self.upconv1 = nn.Conv2d(num_feat, 16 * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(16, 16 * 4, 3, 1, 1, bias=True)
            self.conv_last = nn.Conv2d(16, 3, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_forward = self.flownet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward = self.get_flow(x)
        b, n, c, h, w = x.size()

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                extra_feat = torch.cat([feat_curr, flow_warp(feat_prop, flow.permute(0, 2, 3, 1))], dim=1)
                feat_prop = self.flow_guided_dcn(feat_prop, feat_curr, extra_feat, flow)

            feat_prop = torch.cat([feat_curr, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = feat_prop
            if self.one_stage_up:
                out = self.pixel_shuffle(self.upconv(out))
            else:
                out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
                out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
                out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l.append(out)

        if self.return_flow:
            return torch.stack(out_l, dim=1), flows_forward
        else:
            return torch.stack(out_l, dim=1)



class FlowGuidedDeformAttnAlignV2(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, max_residue_magnitude=10):
        super().__init__()
        self.ms_deform_att = SingleScaleDeformAttnV1(d_model=d_model,
                                                     n_heads=n_heads,
                                                     n_points=n_points,
                                                     max_residue_magnitude=max_residue_magnitude)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_reference_points(self, spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, nbr_fea, ext_fea, flow):
        b, c, h, w = nbr_fea.shape
        device = nbr_fea.device

        mask = (torch.zeros(b, h, w) > 1).to(device)

        spatial_shapes = torch.as_tensor([(h, w)], dtype=torch.long).to(device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratio = torch.unsqueeze(self.get_valid_ratio(mask), dim=1)
        ref_point = self.get_reference_points(spatial_shapes, valid_ratio, device=device)

        output = self.ms_deform_att(ext_fea, ref_point, nbr_fea, spatial_shapes, level_start_index,
                                    input_padding_mask=mask.flatten(1), flow=flow)

        return output


class FlowGuidedDeformAttnAlignV4(nn.Module):

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, max_residue_magnitude=10):
        super().__init__()
        self.ms_deform_att = SingleScaleDeformAttnV3(d_model=d_model,
                                                     n_heads=n_heads,
                                                     n_points=n_points,
                                                     max_residue_magnitude=max_residue_magnitude)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_reference_points(self, spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, nbr_fea, cur_fea, ext_fea, flow):
        b, c, h, w = nbr_fea.shape
        device = nbr_fea.device

        mask = (torch.zeros(b, h, w) > 1).to(device)

        spatial_shapes = torch.as_tensor([(h, w)], dtype=torch.long).to(device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratio = torch.unsqueeze(self.get_valid_ratio(mask), dim=1)
        ref_point = self.get_reference_points(spatial_shapes, valid_ratio, device=device)

        output = self.ms_deform_att(ext_fea, ref_point, cur_fea, nbr_fea, spatial_shapes, level_start_index, input_padding_mask=mask.flatten(1), flow=flow)

        return output


class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15, act=nn.LeakyReLU(negative_slope=0.1, inplace=True)):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True),
            act,
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch)
        )

    def forward(self, fea):
        return self.main(fea)

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


# from mmcv.ops import Correlation as MMCVCorrelation


class Correlation(nn.Module):
    def __init__(self, max_displacement):
        super(Correlation, self).__init__()
        self.max_displacement = max_displacement
        self.kernel_size = 2*max_displacement+1
        # self.corr = MMCVCorrelation(1, self.max_displacement, 1, 0, 1)
        self.corr = SpatialCorrelationSampler(1, self.kernel_size, 1, 0, 1)

    def forward(self, x, y):
        b, c, h, w = x.shape
        return self.corr(x, y).view(b, -1, h, w) / c


def centralize(img1, img2):
    b, c, h, w = img1.shape
    rgb_mean = torch.cat([img1, img2], dim=2).view(b, c, -1).mean(2).view(b, c, 1, 1)
    return img1 - rgb_mean, img2 - rgb_mean, rgb_mean


def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias),
        nn.LeakyReLU(0.1, inplace=True)
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


class Decoder(nn.Module):
    def __init__(self, in_channels, groups):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.conv1 = convrelu(in_channels, 96, 3, 1)
        self.conv2 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv3 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv4 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv5 = convrelu(96, 64, 3, 1)
        self.conv6 = convrelu(64, 32, 3, 1)
        self.conv7 = nn.Conv2d(32, 2, 3, 1, 1)

    def channel_shuffle(self, x, groups):
        b, c, h, w = x.size()
        channels_per_group = c // groups
        x = x.view(b, groups, channels_per_group, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, -1, h, w)
        return x

    def forward(self, x):
        if self.groups == 1:
            out = self.conv7(self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))))
        else:
            out = self.conv1(x)
            out = self.channel_shuffle(self.conv2(out), self.groups)
            out = self.channel_shuffle(self.conv3(out), self.groups)
            out = self.channel_shuffle(self.conv4(out), self.groups)
            out = self.conv7(self.conv6(self.conv5(out)))
        return out


class FastFlowNet(nn.Module):

    def __init__(self, groups=3, load_path=None):
        super(FastFlowNet, self).__init__()
        self.groups = groups
        self.pconv1_1 = convrelu(3, 16, 3, 2)
        self.pconv1_2 = convrelu(16, 16, 3, 1)
        self.pconv2_1 = convrelu(16, 32, 3, 2)
        self.pconv2_2 = convrelu(32, 32, 3, 1)
        self.pconv2_3 = convrelu(32, 32, 3, 1)
        self.pconv3_1 = convrelu(32, 64, 3, 2)
        self.pconv3_2 = convrelu(64, 64, 3, 1)
        self.pconv3_3 = convrelu(64, 64, 3, 1)

        self.corr = Correlation(4)
        self.index = torch.tensor([0, 2, 4, 6, 8,
                                   10, 12, 14, 16,
                                   18, 20, 21, 22, 23, 24, 26,
                                   28, 29, 30, 31, 32, 33, 34,
                                   36, 38, 39, 40, 41, 42, 44,
                                   46, 47, 48, 49, 50, 51, 52,
                                   54, 56, 57, 58, 59, 60, 62,
                                   64, 66, 68, 70,
                                   72, 74, 76, 78, 80])

        self.rconv2 = convrelu(32, 32, 3, 1)
        self.rconv3 = convrelu(64, 32, 3, 1)
        self.rconv4 = convrelu(64, 32, 3, 1)
        self.rconv5 = convrelu(64, 32, 3, 1)
        self.rconv6 = convrelu(64, 32, 3, 1)

        self.up3 = deconv(2, 2)
        self.up4 = deconv(2, 2)
        self.up5 = deconv(2, 2)
        self.up6 = deconv(2, 2)

        self.decoder2 = Decoder(87, groups)
        self.decoder3 = Decoder(87, groups)
        self.decoder4 = Decoder(87, groups)
        self.decoder5 = Decoder(87, groups)
        self.decoder6 = Decoder(87, groups)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if load_path:
            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage))

    def warp(self, x, flo):
        B, C, H, W = x.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat([xx, yy], 1).to(x)
        vgrid = grid + flo
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(x, vgrid, mode='bilinear', align_corners=True)
        return output

    def process(self, img1, img2):

        f11 = self.pconv1_2(self.pconv1_1(img1))
        f21 = self.pconv1_2(self.pconv1_1(img2))
        f12 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f11)))
        f22 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f21)))
        f13 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f12)))
        f23 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f22)))
        f14 = F.avg_pool2d(f13, kernel_size=(2, 2), stride=(2, 2))
        f24 = F.avg_pool2d(f23, kernel_size=(2, 2), stride=(2, 2))
        f15 = F.avg_pool2d(f14, kernel_size=(2, 2), stride=(2, 2))
        f25 = F.avg_pool2d(f24, kernel_size=(2, 2), stride=(2, 2))
        f16 = F.avg_pool2d(f15, kernel_size=(2, 2), stride=(2, 2))
        f26 = F.avg_pool2d(f25, kernel_size=(2, 2), stride=(2, 2))

        flow7_up = torch.zeros(f16.size(0), 2, f16.size(2), f16.size(3)).to(f15)
        cv6 = torch.index_select(self.corr(f16, f26), dim=1, index=self.index.to(f16).long())
        r16 = self.rconv6(f16)
        cat6 = torch.cat([cv6, r16, flow7_up], 1)
        flow6 = self.decoder6(cat6)

        flow6_up = self.up6(flow6)
        f25_w = self.warp(f25, flow6_up * 0.625)
        cv5 = torch.index_select(self.corr(f15, f25_w), dim=1, index=self.index.to(f15).long())
        r15 = self.rconv5(f15)
        cat5 = torch.cat([cv5, r15, flow6_up], 1)
        flow5 = self.decoder5(cat5) + flow6_up

        flow5_up = self.up5(flow5)
        f24_w = self.warp(f24, flow5_up * 1.25)
        cv4 = torch.index_select(self.corr(f14, f24_w), dim=1, index=self.index.to(f14).long())
        r14 = self.rconv4(f14)
        cat4 = torch.cat([cv4, r14, flow5_up], 1)
        flow4 = self.decoder4(cat4) + flow5_up

        flow4_up = self.up4(flow4)
        f23_w = self.warp(f23, flow4_up * 2.5)
        cv3 = torch.index_select(self.corr(f13, f23_w), dim=1, index=self.index.to(f13).long())
        r13 = self.rconv3(f13)
        cat3 = torch.cat([cv3, r13, flow4_up], 1)
        flow3 = self.decoder3(cat3) + flow4_up

        flow3_up = self.up3(flow3)
        f22_w = self.warp(f22, flow3_up * 5.0)
        cv2 = torch.index_select(self.corr(f12, f22_w), dim=1, index=self.index.to(f12).long())
        r12 = self.rconv2(f12)
        cat2 = torch.cat([cv2, r12, flow3_up], 1)
        flow2 = self.decoder2(cat2) + flow3_up

        # if self.training:
        #     return flow2, flow3, flow4, flow5, flow6
        # else:
        #     return flow2
        return flow2

    def forward(self, ref, sup):
        assert ref.size() == sup.size()

        ref, sup, _ = centralize(ref, sup)

        h, w = ref.size(2), ref.size(3)
        w_floor = math.floor(math.ceil(w / 64.0) * 64.0)
        h_floor = math.floor(math.ceil(h / 64.0) * 64.0)

        ref = F.interpolate(input=ref, size=(h_floor, w_floor), mode='bilinear', align_corners=False)
        sup = F.interpolate(input=sup, size=(h_floor, w_floor), mode='bilinear', align_corners=False)

        flow = 20 * self.process(ref, sup)
        flow = F.interpolate(input=flow, size=(h, w), mode='bilinear', align_corners=False)

        flow[:, 0, :, :] *= float(w) / float(w_floor)
        flow[:, 1, :, :] *= float(h) / float(h_floor)

        return flow