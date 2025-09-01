import math
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import logging
from functools import reduce,partial
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import Optional, Callable
from collections import OrderedDict
from copy import Error, deepcopy
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, make_layer, ResidualBlockRCB
from spatial_correlation_sampler import SpatialCorrelationSampler
from basicsr.ops.msda import SingleScaleDeformAttnV3
import math
from typing import Iterable, List, Union
import multiprocessing
from multiprocessing import Pool
from timm.models.layers import  to_2tuple, trunc_normal_
import warnings
warnings.filterwarnings("ignore")


@ARCH_REGISTRY.register()
class TSAabl(nn.Module):
    def __init__(self,num_in_ch=3,num_out_ch=3,num_feat=64,num_frame=15):
        super().__init__()
        self.stride = 4
        self.k_num = 3
        self.num_feat = num_feat
        # self.fastflownet = FastFlow_process('/share3/home/zqiang/TMP/experiments/pretrained_models/fastflownet_ft_mix.pth')
        # extract features for each frame
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, padding=1)
        self.feature_extraction = make_layer(ResidualBlockNoBN, 2, num_feat=num_feat) #   ResidualBlockRCB
        self.fuse = nn.Conv2d(2*num_feat, num_feat, 3, padding=1)
        self.resblocks = make_layer(ResidualBlockNoBN, 13, num_feat=num_feat)
        # align  Trajectory-Aware Alignment
        self.LTAM = LTAM(stride = self.stride, k_num = self.k_num)
        # upsample
        self.upconv1 = nn.Conv2d(num_feat, 48, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(4)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    # def compute_flow(self, lrs):
    #     """Compute optical flow using SPyNet for feature warping.
    #     """
    #     n, t, c, h, w = lrs.size()
    #     lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
    #     lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)
    #     flows_forward = self.fastflownet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

    #     return flows_forward

    def forward(self, lrs, gts=None):
        #print('lrs',lrs.shape)
        n, t, c, h, w = lrs.size()
        assert h % 4 == 0 and w % 4 == 0, ('The height and width must be multiple of 4.')

        # extract features for each frame
        feat_origin = self.lrelu(self.conv_first(lrs.view(-1, c, h, w)))
        outputs = self.feature_extraction(feat_origin).view(n, t, -1,h,w)
        outputs = torch.unbind(outputs,dim=1)
        outputs = list(outputs)

        # compute optical flow
        # flows_forward = self.compute_flow(lrs)
        # if gts != None:
        #     gt_flows_forward = self.compute_flow(gts)
        #     n, t, c, H, W = gts.size()
        #     assert H % 16 == 0 and W % 16 == 0, ('The height and width must be multiple of 16.')
        #     gt_grid_y, gt_grid_x = torch.meshgrid(torch.arange(0, H//self.stride), torch.arange(0, W//self.stride))
        #     gt_location_update = torch.stack([gt_grid_x,gt_grid_y],dim=0).type_as(gts).expand(n,-1,-1,-1)
        fina_out = []
        index_feat_buffers_s1 = []

        # backward-time propgation
        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)
        out_feat = lrs.new_zeros(n, self.num_feat, h, w)
        # grid_y, grid_x = torch.meshgrid(torch.arange(0, h//self.stride), torch.arange(0, w//self.stride))
        # location_update = torch.stack([grid_x,grid_y],dim=0).type_as(lrs).expand(n,-1,-1,-1)
        for i in range(0, t):
            lr_curr = lrs[:, i, :, :, :]
            lr_curr_feat = outputs[i]
            if i > 0:  # no warping required for the first timestep
                # if flows_forward is not None:
                #     flow = flows_forward[:, i - 1, :, :, :]
                # extra_feat = torch.cat([lr_curr_feat, flow_warp(feat_prop, flow.permute(0, 2, 3, 1))], dim=1)
                # feat_prop = self.fuse(torch.cat([lr_curr_feat, feat_prop], dim=1))
                feat_prop = feat_prop + lr_curr_feat
                # feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                # feat_prop = self.flow_guided_dcn(feat_prop,lr_curr_feat, extra_feat, flow) # .permute(0, 2, 3, 1)

                # update the location map
                # flow = F.adaptive_avg_pool2d(flow,(h//self.stride,w//self.stride))/self.stride
                # location_update = flow_warp(location_update, flow.permute(0, 2, 3, 1),padding_mode='border',interpolation="nearest")  # n , 2t , h , w

                #  = torch.stack(index_feat_buffers_s1, dim=1)
                feat_prop = self.LTAM(lr_curr_feat, outputs, feat_prop) # memory_buffer,
                # location_update = torch.cat([location_update,torch.stack([grid_x,grid_y],dim=0).type_as(lrs).expand(n,-1,-1,-1)],dim=1)
                # if gts != None:
                #     if gt_flows_forward is not None:
                #         gt_flow = gt_flows_forward[:, i - 1, :, :, :]
                #     gt_flow = F.adaptive_avg_pool2d(gt_flow,(H//self.stride,W//self.stride))/self.stride
                #     gt_location_update = flow_warp(gt_location_update, gt_flow.permute(0, 2, 3, 1),padding_mode='border',interpolation="nearest")  # n , 2t , h , w
                #     gt_location_update = torch.cat([gt_location_update,torch.stack([gt_grid_x,gt_grid_y],dim=0).type_as(gts).expand(n,-1,-1,-1)],dim=1)
                #     gt_location_update_down = F.interpolate(gt_location_update, scale_factor=0.25, mode='bilinear', align_corners=False)/4

            # feature tokenization *4
            # bs * c * h * w --> # bs * (c*4*4) * (h//4*w//4)
            # index_feat_prop_s1 = F.unfold(lr_curr_feat, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)
            # bs * (c*4*4) * (h//4*w//4) -->  bs * (c*4*4) * h//4 * w//4
            # index_feat_prop_s1 = F.fold(index_feat_prop_s1, output_size=(h//self.stride,w//self.stride), kernel_size=(1,1), padding=0, stride=1)
            # index_feat_buffers_s1.append(index_feat_prop_s1)

            # first fusion
            feat_prop = torch.cat([lr_curr_feat, feat_prop], dim=1)
            feat_prop = self.resblocks(self.fuse(feat_prop))
            # feat_prop = self.resblocks(feat_prop)

            # extra_feat = torch.cat([lr_curr_feat, feat_prop], dim=1)
            # feat_prop = self.deformFusion(feat_prop,lr_curr_feat, extra_feat, flow=None)
            # feat_prop = self.resblocks(feat_prop)

            out_feat = feat_prop
            # upsample
            out = self.pixel_shuffle(self.upconv1(out_feat)).view(n, c, 4*h, 4*w)
            base = F.interpolate(lrs[:, i, :, :, :], scale_factor=4, mode='bilinear', align_corners=False).view(n, c, 4*h, 4*w)
            out += base
            fina_out.append(out)

        # del location_update
        del index_feat_buffers_s1

        # out_list = torch.stack(fina_out, dim=1)
        if gts != None:
            out_list = torch.stack(fina_out, dim=1)
            outcat_list = []
            outcat_list.append(out_list)
            outcat_list.append(location_update)
            outcat_list.append(gt_location_update_down)
            return outcat_list
        else:
            out_list = torch.stack(fina_out, dim=1)
            return out_list





class LTAM(nn.Module):
    def __init__(self, stride=4, k_num=3):
        super().__init__()
        self.stride = stride
        self.k_num = k_num

        self.fusion0 = nn.Conv2d(64, 64//2, 3, 1, 1, bias=True)
        self.fusion2 = nn.Conv2d((self.k_num+1) * 64, 64, 3, 1, 1, bias=True)

        self.S_Mamba = Shfited_Mamba_Block(
                            k_num = k_num,
                            hidden_dim=64//2,
                            norm_layer=nn.LayerNorm,
                            d_state=4,
                            expand=2,
                            window_size = [8,8],
                            shift_size_1 = [0,1],
                            shift_size_3 = [3,3],)
        self.deformFusion = FlowGDAAlign(d_model=64, n_levels=1,n_heads=8,n_points=4,max_residue_magnitude=10)


    def forward(self, curr_feat,  index_feat_set_s1, anchor_feat):
        """Compute the long-range trajectory-aware Mamba.
        Args:
            anchor_feat (tensor): Input feature with shape (n, c, h, w)
            location_feat (tensor): Input location map with shape (n, 2*t, h//4, w//4)
        Return:
            fusion_feature (tensor): Output fusion feature with shape (n, c, h, w).
        """
        n, c, h, w = anchor_feat.size()
        t = len(index_feat_set_s1) # .size(1)
        feat_len = int(c*self.stride*self.stride)
        feat_num = int((h//self.stride) * (w//self.stride))
        feat_curr0 = anchor_feat # curr_feat
        curr_feat0 = curr_feat

        # grid_flow [0,h-1][0,w-1] -> [-1,1][-1,1]
        #  grid_flow = location_feat.contiguous().view(n,t,2,h//self.stride,w//self.stride).permute(0, 1, 3, 4, 2)
        # grid_flow_x = 2.0 * grid_flow[:, :, :, :, 0] / max(w//self.stride - 1, 1) - 1.0
        # grid_flow_y = 2.0 * grid_flow[:, :, :, :, 1] / max(h//self.stride - 1, 1) - 1.0
        # grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=4)

        # index_output_s1 = F.grid_sample(index_feat_set_s1.contiguous().view(-1,(c*self.stride*self.stride),(h//self.stride),(w//self.stride)),grid_flow.contiguous().view(-1,(h//self.stride),(w//self.stride),2),mode='nearest',padding_mode='zeros',align_corners=True) # (nt) * (c*4*4) * (h//4) * (w//4)
        # n * c * h * w --> # n * (c*4*4) * (h//4*w//4)
        # curr_feat = F.unfold(curr_feat, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)
        # n * (c*4*4) * (h//4*w//4) --> n * (h//4*w//4) * (c*4*4)
        # curr_feat = curr_feat.permute(0, 2, 1)
        # curr_feat = F.normalize(curr_feat, dim=2).unsqueeze(3) # n * (h//4*w//4) * (c*4*4) * 1

        # cross-scale attention * 4
        # n * t * (c*4*4) * h//4 * w//4 --> nt * (c*4*4) * h//4 * w//4
        # index_output_s1 = index_output_s1.contiguous().view(n*t,(c*self.stride*self.stride),(h//self.stride),(w//self.stride))
        # nt * (c*4*4) * h//4 * w//4 --> n * t * (c*4*4) * (h//4*w//4)
        # index_output_s1 = F.unfold(index_output_s1, kernel_size=(1, 1), padding=0, stride=1).view(n,-1,feat_len,feat_num)
        # n * t * (c*4*4) * (h//4*w//4) --> n * (h//4*w//4) * t * (c*4*4)
        # index_output_s1 = index_output_s1.permute(0, 3, 1, 2)
        # index_output_s1 = F.normalize(index_output_s1, dim=3) # n * (h//4*w//4) * t * (c*4*4)
        # [ n * (h//4*w//4) * t * (c*4*4) ]  *  [ n * (h//4*w//4) * (c*4*4) * 1 ]  -->  n * (h//4*w//4) * t
        # matrix_index = torch.matmul(index_output_s1, curr_feat).squeeze(3) # n * (h//4*w//4) * t
        # matrix_index = matrix_index.view(n,feat_num,t)# n * (h//4*w//4) * t
        # corr_soft, corr_index = torch.max(matrix_index, dim=2)# n * (h//4*w//4)
        # n * (h//4*w//4) --> n * (c*4*4) * (h//4*w//4)

        # corr_soft_list = []
        # corr_index_list = []
        # if self.k_num <= t:
        #     corr_soft_k, corr_index_k = torch.topk(matrix_index, k=self.k_num, dim=2)  # n * (h//4*w//4)
        #     [corr_soft_list.append(corr_soft_k[:,:,i]) for i in range(self.k_num)][0]
        #     [corr_index_list.append(corr_index_k[:,:,i]) for i in range(self.k_num)][0]
        # else:
        #     corr_soft, corr_index = torch.max(matrix_index, dim=2)  # n * (h//4*w//4)
        #     [corr_soft_list.append(corr_soft) for i in range(self.k_num)][0]
        #     [corr_index_list.append(corr_index)  for i in range(self.k_num)][0]

        out = anchor_feat.new_zeros(n, c, h, w)
        out_list = []
        out_list.append(self.fusion0(feat_curr0))
        for ii in range(0,self.k_num):
            # corr_soft = corr_soft_list[ii]
            # corr_soft = corr_soft.unsqueeze(1).expand(-1,feat_len,-1)
            # n * (c*4*4) * (h//4*w//4) --> n * c * h * w
            # corr_soft = F.fold(corr_soft, output_size=(h,w), kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)
            # corr_index = corr_index_list[ii]
            # output_agg = output_agg * corr_soft feat_curr0
            out_list.append(self.fusion0(index_feat_set_s1[-ii]*anchor_feat))

        ##### Mamba Aggregation
        input_mamba = torch.cat(out_list,0)
        output = self.S_Mamba(curr_feat0, input_mamba)

        fused = self.fusion2(output.reshape(n, -1, h, w))

        extra_feat = torch.cat([curr_feat0, fused], dim=1)
        output = self.deformFusion(fused, curr_feat0, extra_feat, flow=None)

        return output





class FlowGDAAlign(nn.Module):

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, max_residue_magnitude=10):
        super().__init__()
        self.ms_deform_att = SingleScaleDeformAttnV3(d_model=d_model, n_heads=n_heads, n_points=n_points,
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









class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, num_feat=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)




def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or a feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (w, h, 2)
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output



##############   Residual Context Block   #################


class ContextBlock(nn.Module):

    def __init__(self, n_feat, bias=False):
        super(ContextBlock, self).__init__()

        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        )

    def modeling(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x, x_r):
        # [N, C, 1, 1]
        context = self.modeling(x) + self.modeling(x_r)

        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        x = x + channel_add_term

        return x


class RCB(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=2, bias=False, groups=1):
        super(RCB, self).__init__()

        # act = nn.LeakyReLU(0.2)

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat//reduction, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feat//reduction, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups)
        )

        self.act = nn.LeakyReLU(0.2)

        self.gcnet = ContextBlock(n_feat, bias=bias)
        self.gcnet1 = ContextBlock(n_feat, bias=bias)

    def forward(self, x, x_r):
        res = self.body(x)
        x_r = self.body(x_r)
        res = self.act(self.gcnet(res,x_r))
        res = self.act(self.gcnet1(res,x_r))
        res += x
        return res


########################   FastFlow   ########################

class Correlation(nn.Module):
    def __init__(self, max_displacement):
        super(Correlation, self).__init__()
        self.max_displacement = max_displacement
        self.kernel_size = 2*max_displacement+1
        self.corr = SpatialCorrelationSampler(1, self.kernel_size, 1, 0, 1)

    def forward(self, x, y):
        b, c, h, w = x.shape
        return self.corr(x, y).view(b, -1, h, w) / c



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
    def __init__(self, groups=3):
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


    def warp(self, x, flo):
        B, C, H, W = x.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat([xx, yy], 1).to(x)
        vgrid = grid + flo
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W-1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H-1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(x, vgrid, mode='bilinear', align_corners=True)
        return output


    def forward(self, x):
        img1 = x[:, :3, :, :]
        img2 = x[:, 3:6, :, :]
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
        f25_w = self.warp(f25, flow6_up*0.625)
        cv5 = torch.index_select(self.corr(f15, f25_w), dim=1, index=self.index.to(f15).long())
        r15 = self.rconv5(f15)
        cat5 = torch.cat([cv5, r15, flow6_up], 1)
        flow5 = self.decoder5(cat5) + flow6_up

        flow5_up = self.up5(flow5)
        f24_w = self.warp(f24, flow5_up*1.25)
        cv4 = torch.index_select(self.corr(f14, f24_w), dim=1, index=self.index.to(f14).long())
        r14 = self.rconv4(f14)
        cat4 = torch.cat([cv4, r14, flow5_up], 1)
        flow4 = self.decoder4(cat4) + flow5_up

        flow4_up = self.up4(flow4)
        f23_w = self.warp(f23, flow4_up*2.5)
        cv3 = torch.index_select(self.corr(f13, f23_w), dim=1, index=self.index.to(f13).long())
        r13 = self.rconv3(f13)
        cat3 = torch.cat([cv3, r13, flow4_up], 1)
        flow3 = self.decoder3(cat3) + flow4_up

        flow3_up = self.up3(flow3)
        f22_w = self.warp(f22, flow3_up*5.0)
        cv2 = torch.index_select(self.corr(f12, f22_w), dim=1, index=self.index.to(f12).long())
        r12 = self.rconv2(f12)
        cat2 = torch.cat([cv2, r12, flow3_up], 1)
        flow2 = self.decoder2(cat2) + flow3_up

        # if self.training:
        #     return flow2, flow3, flow4, flow5, flow6
        # else:
        return flow2



class FastFlow_process(nn.Module):
    """FastFlow_process architecture.
    Args:
        load_path (str): path for pretrained FastFlow_process. Default: None.
    """

    def __init__(self, load_path=None):
        super(FastFlow_process, self).__init__()
        # self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        self.model = FastFlowNet().cuda().eval()
        if load_path:
            self.model.load_state_dict(torch.load(load_path))

        self.div_flow = 20.0
        self.div_size = 64
        # self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


    def centralize(self, img1, img2):
        b, c, h, w = img1.shape
        rgb_mean = torch.cat([img1, img2], dim=2).view(b, c, -1).mean(2).view(b, c, 1, 1)
        return img1 - rgb_mean, img2 - rgb_mean, rgb_mean

    def forward(self, img1, img2):
        assert img1.size() == img2.size()

        # img1 = torch.from_numpy(cv2.imread(img1_path)).float().permute(2, 0, 1).unsqueeze(0)/255.0
        # img2 = torch.from_numpy(cv2.imread(img2_path)).float().permute(2, 0, 1).unsqueeze(0)/255.0
        img1, img2, _ = self.centralize(img1, img2)

        height, width = img1.shape[-2:]
        orig_size = (int(height), int(width))

        if height % self.div_size != 0 or width % self.div_size != 0:
            input_size = (
                int(self.div_size * np.ceil(height / self.div_size)),
                int(self.div_size * np.ceil(width / self.div_size))
            )
            img1 = F.interpolate(img1, size=input_size, mode='bilinear', align_corners=False)
            img2 = F.interpolate(img2, size=input_size, mode='bilinear', align_corners=False)
        else:
            input_size = orig_size

        input_t = torch.cat([img1, img2], 1).cuda()

        output = self.model(input_t) # .data
        # print('output',output.shape)

        flow = self.div_flow * F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)

        if input_size != orig_size:
            scale_h = orig_size[0] / input_size[0]
            scale_w = orig_size[1] / input_size[1]
            flow = F.interpolate(flow, size=orig_size, mode='bilinear', align_corners=False)
            flow[:, 0, :, :] *= scale_w
            flow[:, 1, :, :] *= scale_h

        # flow = flow[0].cpu().permute(1, 2, 0).numpy()

        return flow





########################## Shifted Mamba Block  ##########################


class LC_Mamba_Block(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            shift_size=0,
            expand: float = 2.,
            window_size=8,
            **kwargs,
    ):
        super().__init__()
        print('LC_Mamba_Block')

        self.window_size=window_size
        self.shift_size=shift_size

        if not isinstance(self.window_size, (tuple, list)):
            self.window_size = to_2tuple(window_size)

        if not isinstance(self.shift_size, (tuple, list)):
            self.shift_size = to_2tuple(shift_size)

        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SW_HSS3D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate,shift_size=self.shift_size, window_size=self.window_size , **kwargs)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


    def forward(self, input_x):
        # input_x = torch.cat(input_x, 0)
        shortcut=input_x.permute(0, 2, 3, 1).contiguous()
        B,H,W,C= shortcut.shape
        input_x=shortcut
        x_pad, mask = pad_if_needed(input_x, input_x.size(), self.window_size)# # b,hw,ww,c , b,hw,ww,1
        _, Hw, Ww, C = x_pad.shape

        if self.shift_size[0] or self.shift_size[1]:
            _, H_p, W_p, C = x_pad.shape
            x_pad = torch.roll(x_pad, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) #
            shift_mask = torch.zeros((B, H_p, W_p, 1))  # 1 H W 1
            shift_mask[:,  : int(-self.shift_size[0]) , : int(-self.shift_size[1])  , :]  = 1.# B H_p W_p 1
            if mask is not None:
                    mask = torch.roll(mask, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) #
                    shift_mask = shift_mask*mask
            shift_mask = window_partition(shift_mask, self.window_size).permute(0,2,1)   #(b*nW),1(d),L

        else:
            if mask is not None:
                shift_mask= window_partition(mask, self.window_size).permute(0,2,1)
            else :
                shift_mask= None

        x_pad = self.ln_1(x_pad) # B,N,C
        x_back_win = self.self_attention(x_pad,shift_mask)

        if self.shift_size[0] or self.shift_size[1]:
            x = torch.roll(x_back_win, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

        x = depad_if_needed(x_back_win, shortcut.size(), self.window_size)

        x= shortcut * self.skip_scale + x  # B,H,W,C

        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        return x.permute(0, 3, 1, 2).contiguous()





class Shfited_Mamba_Block(nn.Module):
    def __init__(
            self,
            k_num: int = 3,
            hidden_dim: int = 0,
            norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            shift_size_1=0,
            shift_size_3=0,
            expand: float = 2.,
            window_size=8,
            **kwargs,
    ):
        super().__init__()
        print('Shfited_Mamba_Block')

        self.k_num = k_num
        self.window_size=window_size
        self.shift_size = [0,0]
        self.shift_size_1=shift_size_1
        self.shift_size_3=shift_size_3

        if not isinstance(self.window_size, (tuple, list)):
            self.window_size = to_2tuple(window_size)

        if not isinstance(self.shift_size_1, (tuple, list)):
            self.shift_size_1 = to_2tuple(shift_size_1)

        if not isinstance(self.shift_size_3, (tuple, list)):
            self.shift_size_3 = to_2tuple(shift_size_3)

        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention      = SW_HSS3D_Scan_1(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate,shift_size=0, window_size=self.window_size , **kwargs)
        self.shifted_attention_1 = SW_HSS3D_Scan_3(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate,shift_size=self.shift_size_1, window_size=self.window_size , **kwargs)
        self.shifted_attention_3 = SW_HSS3D_Scan_3(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate,shift_size=self.shift_size_3, window_size=self.window_size , **kwargs)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim*2)
        self.ln_2 = nn.LayerNorm(hidden_dim*2)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim*2))
        self.conv_layer = nn.Conv2d(hidden_dim*2, hidden_dim*2, 3, 1, 1, bias=True)
        self.fusion2 = nn.Conv2d((self.k_num+1) * hidden_dim, hidden_dim*2, 3, 1, 1, bias=True)
        self.conv_layer1 = nn.Conv2d(hidden_dim, hidden_dim*2, 3, 1, 1, bias=True)





    def forward(self, curr_feat0, input_x):
        n, c, h, w  = curr_feat0.shape
        # input_x = torch.cat(input_x, 0)
        shortcut=input_x.permute(0, 2, 3, 1).contiguous()
        B,H,W,C= shortcut.shape
        input_x=shortcut

        # [Shift 0]
        x_pad, mask = pad_if_needed(input_x, input_x.size(), self.window_size)# # b,hw,ww,c , b,hw,ww,1
        _, Hw, Ww, C = x_pad.shape

        if self.shift_size[0] or self.shift_size[1]:
            _, H_p, W_p, C = x_pad.shape
            x_pad = torch.roll(x_pad, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) #
            shift_mask = torch.zeros((B, H_p, W_p, 1))  # 1 H W 1
            shift_mask[:,  : int(-self.shift_size[0]) , : int(-self.shift_size[1])  , :]  = 1.# B H_p W_p 1
            if mask is not None:
                    mask = torch.roll(mask, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) #
                    shift_mask = shift_mask*mask
            shift_mask = window_partition(shift_mask, self.window_size).permute(0,2,1)   #(b*nW),1(d),L

        else:
            if mask is not None:
                shift_mask= window_partition(mask, self.window_size).permute(0,2,1)
            else :
                shift_mask= None

        x_ln = self.ln_1(x_pad) # B,N,C  Layernorm
        x_back_win = self.self_attention(x_ln,shift_mask)  #  Mamba block

        if self.shift_size[0] or self.shift_size[1]:
            x = torch.roll(x_back_win, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

        x = depad_if_needed(x_back_win, shortcut.size(), self.window_size)

        x_shift0= shortcut * self.skip_scale + x  # B,H,W,C

        # # [Shift 1,0]
        x_pad, mask = pad_if_needed(x_shift0, input_x.size(), self.window_size)# # b,hw,ww,c , b,hw,ww,1
        _, Hw, Ww, C = x_pad.shape
        if self.shift_size_1[0] or self.shift_size_1[1]:
            _, H_p, W_p, C = x_pad.shape
            x_pad = torch.roll(x_pad, shifts=(-self.shift_size_1[0], -self.shift_size_1[1]), dims=(1, 2)) #
            shift_mask = torch.zeros((B, H_p, W_p, 1))  # 1 H W 1
            shift_mask[:,  : int(-self.shift_size_1[0]) , : int(-self.shift_size_1[1])  , :]  = 1.# B H_p W_p 1
            if mask is not None:
                    mask = torch.roll(mask, shifts=(-self.shift_size_1[0], -self.shift_size_1[1]), dims=(1, 2)) #
                    shift_mask = shift_mask*mask
            shift_mask = window_partition(shift_mask, self.window_size).permute(0,2,1)   #(b*nW),1(d),L

        else:
            if mask is not None:
                shift_mask= window_partition(mask, self.window_size).permute(0,2,1)
            else :
                shift_mask= None

        x_ln = self.ln_1(x_pad) # B,N,C
        x_back_win = self.shifted_attention_1(x_ln,shift_mask)

        if self.shift_size_1[0] or self.shift_size_1[1]:
            x = torch.roll(x_back_win, shifts=(self.shift_size_1[0], self.shift_size_1[1]), dims=(1, 2))

        x = depad_if_needed(x_back_win, x_shift0.size(), self.window_size)

        x= x_shift0 * self.skip_scale + x  # B,H,W,C
        x_shift1 = x


        # [Shift 3,3]
        x_pad, mask = pad_if_needed(x_shift0, input_x.size(), self.window_size)# # b,hw,ww,c , b,hw,ww,1
        _, Hw, Ww, C = x_pad.shape
        if self.shift_size_3[0] or self.shift_size_3[1]:
            _, H_p, W_p, C = x_pad.shape
            x_pad = torch.roll(x_pad, shifts=(-self.shift_size_3[0], -self.shift_size_3[1]), dims=(1, 2)) #
            shift_mask = torch.zeros((B, H_p, W_p, 1))  # 1 H W 1
            shift_mask[:,  : int(-self.shift_size_3[0]) , : int(-self.shift_size_3[1])  , :]  = 1.# B H_p W_p 1
            if mask is not None:
                    mask = torch.roll(mask, shifts=(-self.shift_size_3[0], -self.shift_size_3[1]), dims=(1, 2)) #
                    shift_mask = shift_mask*mask
            shift_mask = window_partition(shift_mask, self.window_size).permute(0,2,1)   #(b*nW),1(d),L

        else:
            if mask is not None:
                shift_mask= window_partition(mask, self.window_size).permute(0,2,1)
            else :
                shift_mask= None

        x_ln = self.ln_1(x_pad) # B,N,C
        x_back_win = self.shifted_attention_3(x_ln,shift_mask)

        if self.shift_size_3[0] or self.shift_size_3[1]:
            x = torch.roll(x_back_win, shifts=(self.shift_size_3[0], self.shift_size_3[1]), dims=(1, 2))

        x = depad_if_needed(x_back_win, x_shift0.size(), self.window_size)

        x_shift3= x_shift0 * self.skip_scale + x  # B,H,W,C

        # fused = self.conv_layer(torch.cat([x_shift0, x_shift3], dim = 3).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        # fused = self.fusion2(fused.reshape(n, -1, h, w))

        # extra_feat = torch.cat([curr_feat0, fused], dim=1)
        # output = self.deformFusion(fused, curr_feat0, extra_feat, flow=None)
        # print('output',curr_feat0.shape, output.shape,fused.shape)
        # final_out = output
        # final_out = output * self.skip_scale2 + self.conv_blk(self.ln_2(output))

        # final_out = output.permute(0, 2, 3, 1).contiguous() * self.skip_scale2
        # final_out = final_out + self.conv_blk(self.ln_2(output.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        # final_out = final_out.permute(0, 3, 1, 2).contiguous()
        # print('output',final_out.shape)

        fused = self.conv_layer(torch.cat([x_shift1, x_shift3], dim = 3).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = fused * self.skip_scale2 + self.conv_blk(self.ln_2(fused).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        x = self.conv_layer1(x_shift3.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous() + x + curr_feat0.repeat(self.k_num+1, 1, 1, 1).permute(0, 2, 3, 1).contiguous()
        # x = self.conv_layer1(x_shift0.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()


        return x.permute(0, 3, 1, 2).contiguous()


        # return final_out




#################################


class SW_HSS3D_Scan_1(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            shift_size=0,
            window_size=[8,8],
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        # print('SW_HSS3D')
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.window_size=window_size
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.delta_softplus=True

        # ##### Scan 1  ##### [ 0  1  5  4  8 12 13  9 10 14 15 11  7  6  2  3]
        p = int(np.log2(window_size[0]))
        n = 2

        H,W=window_size[0],window_size[1]
        hilbert_curve = HilbertCurve(p, n)

        coords = []
        for y in range(H):
            for x in range(W):
                coords.append((x, y))

        # ##### Scan 2  #####  [ 0  4  5  1  2  3  7  6 10 11 15 14 13  9  8 12]
        # p = int(np.log2(window_size[0])) + 1
        # n = 2

        # H,W=window_size[0],window_size[1]
        # hilbert_curve = HilbertCurve(p, n)

        # coords = []
        # for y in range(H):
        #     for x in range(W):
        #         coords.append((x, y))

        # ##### Scan 3  #####  [15 14 10 11  7  3  2  6  5  1  0  4  8  9 13 12]
        # p = int(np.log2(window_size[0]))
        # n = 2

        # H,W=window_size[0],window_size[1]
        # hilbert_curve = HilbertCurve(p, n)

        # coords = []
        # for y in range(H-1,-1,-1):
        #     for x in range(W-1,-1,-1):
        #         coords.append((x, y))

        # ##### Scan 4  #####  [15 11 10 14 13 12  8  9  5  4  0  1  2  6  7  3]
        # p = int(np.log2(window_size[0])) + 1
        # n = 2

        # H,W=window_size[0],window_size[1]
        # hilbert_curve = HilbertCurve(p, n)

        # coords = []
        # for y in range(H-1,-1,-1):
        #     for x in range(W-1,-1,-1):
        #         coords.append((x, y))


        hilbert_indices = []
        for coord in coords:
            x, y = coord
            hilbert_index = hilbert_curve.distance_from_point([x, y])
            hilbert_indices.append(hilbert_index)

        hilbert_indices = np.array(hilbert_indices)
        self.sorted_indices = np.argsort(hilbert_indices)
        self.inverse_indices = np.argsort( self.sorted_indices)
        # print('self.sorted_indices',self.sorted_indices)
        # print('self.inverse_indices',self.inverse_indices)


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def merge_x(self, x):
        B, C, H, W = x.shape
        x= x.view(B,C,H*W).transpose(1,2) # B,H*W,C
        x = torch.cat([x[:B//2], x[B//2:]], dim=-1).reshape(B//2, 2 * H * W, C) # 프레임 0,1을 L에 합침.
        return x.transpose(1, 2).contiguous() # B//2,C,2L

    def unmerge_x(self, x ):
        B, C,L = x.shape  #B,C,2L , L=H*W

        odd_elements = x[:, :, 1::2]  # L 차원에서 홀수 인덱스 (1, 3, 5, ...)
        even_elements = x[:, :, 0::2]  # L 차원에서 짝수 인덱스 (0, 2, 4, ...)

        # 홀수, 짝수 순서로 배치
        x = torch.cat((even_elements, odd_elements), dim=0).view(B*2,C,L//2).contiguous()

        return x

    def forward_core(self, x: torch.Tensor,shift_mask=None):
        B, C, H, W = x.shape # BNW,C,W0,W1

        # print('self.sorted_indices',self.sorted_indices)

        h_ordered_tensor = apply_hilbert_curve_2d(x,self.sorted_indices)
        h_ordered_tensor_wh = apply_hilbert_curve_2d(torch.transpose(x, dim0=2, dim1=3).contiguous(),self.sorted_indices)


        x=h_ordered_tensor.view(B,C,H,W).contiguous()
        x_wh=h_ordered_tensor_wh.view(B,C,W,H).contiguous()

        L = 2 * H * W #inter
        K = 4 #

        if shift_mask != None:
            shift_mask=shift_mask.view(B,1,H,W)
            h_ordered_tensor_wh= apply_hilbert_curve_2d(shift_mask,self.sorted_indices)
            h_ordered_mask_wh= apply_hilbert_curve_2d(torch.transpose(shift_mask, dim0=2, dim1=3).contiguous(),self.sorted_indices)
            h_ordered_tensor_wh=h_ordered_tensor_wh.view(B,1,H,W).contiguous()
            h_ordered_mask_wh=h_ordered_mask_wh.view(B,1,W,H).contiguous()
            shift_mask_hwwh = torch.stack([self.merge_x(h_ordered_tensor_wh), self.merge_x(h_ordered_mask_wh)], dim=1).view(B//2, 2, 1, L) # B//2,2,C,2L,  horizon,vertical
            shift_mask_hwwh_reverse =  torch.flip(shift_mask_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,2L,
            shift_mask = torch.cat([shift_mask_hwwh,shift_mask_hwwh_reverse], dim=1) # reverse# B//2,4,C,2L,
            shift_mask = shift_mask.float().view(B//2, -1, L).unsqueeze(-2).to(x.device) # B//2 , K , 1, L


        B = B // 2


        x_hwwh = torch.stack([self.merge_x(x), self.merge_x(x_wh)], dim=1).view(B, 2, -1, L) # B//2,2,C,2L,  horizon,vertical
        x_hwwh_reverse =  torch.flip(x_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,2L,
        xs = torch.cat([x_hwwh,x_hwwh_reverse], dim=1) # reverse# B//2,4,C,2L,

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) ## projection
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L) # B, 4 *c , L
        Bs = Bs.float().view(B, K, -1, L) # B, 4, d_state , L

        Cs = Cs.float().view(B, K, -1, L)# B, 4, d_state , L
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float()

        if dt_projs_bias is not None:
            dts = dts + dt_projs_bias[...,None].float()
        if self.delta_softplus is True :
            dts = F.softplus(dts)
        if shift_mask != None:
            dts= dts*shift_mask #e 0이면 이전 스테이트에 유지 # (B, K, -1, L) * B , K , 1, L

        dts = dts.contiguous().float().view(B, -1, L) # B, 4 *d , L

        out_y = self.selective_scan(
            xs.contiguous(), #u
            dts.contiguous(),
            As.contiguous(),
            Bs.contiguous(),
            Cs.contiguous(),
            Ds.contiguous(),
            #z=None,
            delta_bias=None,
            delta_softplus=False,
            #return_last_state=False,
        ).view(B, K, -1, L)

        assert out_y.dtype == torch.float

        y= out_y[:, 0]
        wh_y= out_y[:, 1] #B//2,C,2L

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        invwh_y = inv_y[:, 1]#B//2,C,2L

        wh_y = wh_y+ invwh_y # B//2,C,2L

        wh_y= self.unmerge_x(wh_y)# B,C,L
        wh_y = reverse_hilbert_curve_2d(wh_y, self.inverse_indices , W,H)#.view(batch, channel, height, width)
        wh_y = torch.transpose(wh_y,2,3).contiguous()

        y=  out_y[:, 0]+ inv_y[:, 0]
        y= self.unmerge_x(y) # B,C,L
        y = reverse_hilbert_curve_2d(y,self.inverse_indices,H,W)#.view(batch, channel, height, width)

        y= y+wh_y
        y= y.permute(0,2,3,1).contiguous()

        return y


    def window_partition(self,x, window_size):
        B, C, H, W = x.shape
        x=x.permute(0,2,3,1)
        x = x.view(B,H // window_size[0], window_size[0], W // window_size[1], window_size[1],C)
        windows = (
            x.permute(0, 1, 3,2,4,5).contiguous().view(-1, window_size[0],window_size[1],C)
        )
        return windows.permute(0,3,1,2)
    def window_reverse(self,windows, window_size, H, W):
        nwB, w0,w1, C = windows.shape
        #windows = windows.view(-1, window_size[0], window_size[1], C)
        B = int(nwB / (H * W / window_size[0] / window_size[1]))
        x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x



    def forward(self, x: torch.Tensor,shift_mask=None, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        x = self.window_partition(x, self.window_size) # B*NW,C,H,W

        y = self.forward_core(x, shift_mask ) #b,h,w,c
        assert y.dtype == torch.float32

        y = self.out_norm(y)
        y= self.window_reverse(y,self.window_size,H,W)#B,H,W,C
        y = y * F.silu(z)

        out = self.out_proj(y)#B,H,W,C

        if self.dropout is not None:
            out = self.dropout(out)
        return out





class SW_HSS3D_Scan_3(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            shift_size=0,
            window_size=[8,8],
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        # print('SW_HSS3D')
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.window_size=window_size
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.delta_softplus=True

        # ##### Scan 1  ##### [ 0  1  5  4  8 12 13  9 10 14 15 11  7  6  2  3]
        # p = int(np.log2(window_size[0]))
        # n = 2

        # H,W=window_size[0],window_size[1]
        # hilbert_curve = HilbertCurve(p, n)

        # coords = []
        # for y in range(H):
        #     for x in range(W):
        #         coords.append((x, y))

        # ##### Scan 2  #####  [ 0  4  5  1  2  3  7  6 10 11 15 14 13  9  8 12]
        # p = int(np.log2(window_size[0])) + 1
        # n = 2

        # H,W=window_size[0],window_size[1]
        # hilbert_curve = HilbertCurve(p, n)

        # coords = []
        # for y in range(H):
        #     for x in range(W):
        #         coords.append((x, y))

        # ##### Scan 3  #####  [15 14 10 11  7  3  2  6  5  1  0  4  8  9 13 12]
        p = int(np.log2(window_size[0]))
        n = 2

        H,W=window_size[0],window_size[1]
        hilbert_curve = HilbertCurve(p, n)

        coords = []
        for y in range(H-1,-1,-1):
            for x in range(W-1,-1,-1):
                coords.append((x, y))

        # ##### Scan 4  #####  [15 11 10 14 13 12  8  9  5  4  0  1  2  6  7  3]
        # p = int(np.log2(window_size[0])) + 1
        # n = 2

        # H,W=window_size[0],window_size[1]
        # hilbert_curve = HilbertCurve(p, n)

        # coords = []
        # for y in range(H-1,-1,-1):
        #     for x in range(W-1,-1,-1):
        #         coords.append((x, y))


        hilbert_indices = []
        for coord in coords:
            x, y = coord
            hilbert_index = hilbert_curve.distance_from_point([x, y])
            hilbert_indices.append(hilbert_index)

        hilbert_indices = np.array(hilbert_indices)
        self.sorted_indices = np.argsort(hilbert_indices)
        self.inverse_indices = np.argsort( self.sorted_indices)
        # print('self.sorted_indices',self.sorted_indices)
        # print('self.inverse_indices',self.inverse_indices)


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def merge_x(self, x):
        B, C, H, W = x.shape
        x= x.view(B,C,H*W).transpose(1,2) # B,H*W,C
        x = torch.cat([x[:B//2], x[B//2:]], dim=-1).reshape(B//2, 2 * H * W, C) # 프레임 0,1을 L에 합침.
        return x.transpose(1, 2).contiguous() # B//2,C,2L

    def unmerge_x(self, x ):
        B, C,L = x.shape  #B,C,2L , L=H*W

        odd_elements = x[:, :, 1::2]  # L 차원에서 홀수 인덱스 (1, 3, 5, ...)
        even_elements = x[:, :, 0::2]  # L 차원에서 짝수 인덱스 (0, 2, 4, ...)

        # 홀수, 짝수 순서로 배치
        x = torch.cat((even_elements, odd_elements), dim=0).view(B*2,C,L//2).contiguous()

        return x

    def forward_core(self, x: torch.Tensor,shift_mask=None):
        B, C, H, W = x.shape # BNW,C,W0,W1

        # print('self.sorted_indices',self.sorted_indices)

        h_ordered_tensor = apply_hilbert_curve_2d(x,self.sorted_indices)
        h_ordered_tensor_wh = apply_hilbert_curve_2d(torch.transpose(x, dim0=2, dim1=3).contiguous(),self.sorted_indices)


        x=h_ordered_tensor.view(B,C,H,W).contiguous()
        x_wh=h_ordered_tensor_wh.view(B,C,W,H).contiguous()

        L = 2 * H * W #inter
        K = 4 #

        if shift_mask != None:
            shift_mask=shift_mask.view(B,1,H,W)
            h_ordered_tensor_wh= apply_hilbert_curve_2d(shift_mask,self.sorted_indices)
            h_ordered_mask_wh= apply_hilbert_curve_2d(torch.transpose(shift_mask, dim0=2, dim1=3).contiguous(),self.sorted_indices)
            h_ordered_tensor_wh=h_ordered_tensor_wh.view(B,1,H,W).contiguous()
            h_ordered_mask_wh=h_ordered_mask_wh.view(B,1,W,H).contiguous()
            shift_mask_hwwh = torch.stack([self.merge_x(h_ordered_tensor_wh), self.merge_x(h_ordered_mask_wh)], dim=1).view(B//2, 2, 1, L) # B//2,2,C,2L,  horizon,vertical
            shift_mask_hwwh_reverse =  torch.flip(shift_mask_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,2L,
            shift_mask = torch.cat([shift_mask_hwwh,shift_mask_hwwh_reverse], dim=1) # reverse# B//2,4,C,2L,
            shift_mask = shift_mask.float().view(B//2, -1, L).unsqueeze(-2).to(x.device) # B//2 , K , 1, L


        B = B // 2


        x_hwwh = torch.stack([self.merge_x(x), self.merge_x(x_wh)], dim=1).view(B, 2, -1, L) # B//2,2,C,2L,  horizon,vertical
        x_hwwh_reverse =  torch.flip(x_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,2L,
        xs = torch.cat([x_hwwh,x_hwwh_reverse], dim=1) # reverse# B//2,4,C,2L,

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) ## projection
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L) # B, 4 *c , L
        Bs = Bs.float().view(B, K, -1, L) # B, 4, d_state , L

        Cs = Cs.float().view(B, K, -1, L)# B, 4, d_state , L
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float()

        if dt_projs_bias is not None:
            dts = dts + dt_projs_bias[...,None].float()
        if self.delta_softplus is True :
            dts = F.softplus(dts)
        if shift_mask != None:
            dts= dts*shift_mask #e 0이면 이전 스테이트에 유지 # (B, K, -1, L) * B , K , 1, L

        dts = dts.contiguous().float().view(B, -1, L) # B, 4 *d , L

        out_y = self.selective_scan(
            xs.contiguous(), #u
            dts.contiguous(),
            As.contiguous(),
            Bs.contiguous(),
            Cs.contiguous(),
            Ds.contiguous(),
            #z=None,
            delta_bias=None,
            delta_softplus=False,
            #return_last_state=False,
        ).view(B, K, -1, L)

        assert out_y.dtype == torch.float

        y= out_y[:, 0]
        wh_y= out_y[:, 1] #B//2,C,2L

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        invwh_y = inv_y[:, 1]#B//2,C,2L

        wh_y = wh_y+ invwh_y # B//2,C,2L

        wh_y= self.unmerge_x(wh_y)# B,C,L
        wh_y = reverse_hilbert_curve_2d(wh_y, self.inverse_indices , W,H)#.view(batch, channel, height, width)
        wh_y = torch.transpose(wh_y,2,3).contiguous()

        y=  out_y[:, 0]+ inv_y[:, 0]
        y= self.unmerge_x(y) # B,C,L
        y = reverse_hilbert_curve_2d(y,self.inverse_indices,H,W)#.view(batch, channel, height, width)

        y= y+wh_y
        y= y.permute(0,2,3,1).contiguous()

        return y


    def window_partition(self,x, window_size):
        B, C, H, W = x.shape
        x=x.permute(0,2,3,1)
        x = x.view(B,H // window_size[0], window_size[0], W // window_size[1], window_size[1],C)
        windows = (
            x.permute(0, 1, 3,2,4,5).contiguous().view(-1, window_size[0],window_size[1],C)
        )
        return windows.permute(0,3,1,2)
    def window_reverse(self,windows, window_size, H, W):
        nwB, w0,w1, C = windows.shape
        #windows = windows.view(-1, window_size[0], window_size[1], C)
        B = int(nwB / (H * W / window_size[0] / window_size[1]))
        x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x



    def forward(self, x: torch.Tensor,shift_mask=None, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        x = self.window_partition(x, self.window_size) # B*NW,C,H,W

        y = self.forward_core(x, shift_mask ) #b,h,w,c
        assert y.dtype == torch.float32

        y = self.out_norm(y)
        y= self.window_reverse(y,self.window_size,H,W)#B,H,W,C
        y = y * F.silu(z)

        out = self.out_proj(y)#B,H,W,C

        if self.dropout is not None:
            out = self.dropout(out)
        return out







class SW_HSS3D_signal(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            shift_size=0,
            window_size=[8,8],
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        # print('SW_HSS3D')
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.window_size=window_size
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.delta_softplus=True

        # ##### Scan 1  ##### [ 0  1  5  4  8 12 13  9 10 14 15 11  7  6  2  3]
        p = int(np.log2(window_size[0]))
        n = 2

        H,W=window_size[0],window_size[1]
        hilbert_curve = HilbertCurve(p, n)

        coords = []
        for y in range(H):
            for x in range(W):
                coords.append((x, y))

        # ##### Scan 2  #####  [ 0  4  5  1  2  3  7  6 10 11 15 14 13  9  8 12]
        # p = int(np.log2(window_size[0])) + 1
        # n = 2

        # H,W=window_size[0],window_size[1]
        # hilbert_curve = HilbertCurve(p, n)

        # coords = []
        # for y in range(H):
        #     for x in range(W):
        #         coords.append((x, y))

        # ##### Scan 3  #####  [15 14 10 11  7  3  2  6  5  1  0  4  8  9 13 12]
        # p = int(np.log2(window_size[0]))
        # n = 2

        # H,W=window_size[0],window_size[1]
        # hilbert_curve = HilbertCurve(p, n)

        # coords = []
        # for y in range(H-1,-1,-1):
        #     for x in range(W-1,-1,-1):
        #         coords.append((x, y))

        # ##### Scan 4  #####  [15 11 10 14 13 12  8  9  5  4  0  1  2  6  7  3]
        # p = int(np.log2(window_size[0])) + 1
        # n = 2

        # H,W=window_size[0],window_size[1]
        # hilbert_curve = HilbertCurve(p, n)

        # coords = []
        # for y in range(H-1,-1,-1):
        #     for x in range(W-1,-1,-1):
        #         coords.append((x, y))


        hilbert_indices = []
        for coord in coords:
            x, y = coord
            hilbert_index = hilbert_curve.distance_from_point([x, y])
            hilbert_indices.append(hilbert_index)

        hilbert_indices = np.array(hilbert_indices)
        self.sorted_indices = np.argsort(hilbert_indices)
        self.inverse_indices = np.argsort( self.sorted_indices)
        print('self.sorted_indices',self.sorted_indices)
        print('self.inverse_indices',self.inverse_indices)


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def merge_x(self, x):
        B, C, H, W = x.shape
        x= x.view(B,C,H*W).transpose(1,2) # B,H*W,C
        x = torch.cat([x[:B//2], x[B//2:]], dim=-1).reshape(B//2, 2 * H * W, C) # 프레임 0,1을 L에 합침.
        return x.transpose(1, 2).contiguous() # B//2,C,2L

    def unmerge_x(self, x ):
        B, C,L = x.shape  #B,C,2L , L=H*W

        odd_elements = x[:, :, 1::2]  # L 차원에서 홀수 인덱스 (1, 3, 5, ...)
        even_elements = x[:, :, 0::2]  # L 차원에서 짝수 인덱스 (0, 2, 4, ...)

        # 홀수, 짝수 순서로 배치
        x = torch.cat((even_elements, odd_elements), dim=0).view(B*2,C,L//2).contiguous()

        return x

    def forward_core(self, x: torch.Tensor,shift_mask=None):
        B, C, H, W = x.shape # BNW,C,W0,W1

        h_ordered_tensor = apply_hilbert_curve_2d(x,self.sorted_indices)
        h_ordered_tensor_wh = apply_hilbert_curve_2d(torch.transpose(x, dim0=2, dim1=3).contiguous(),self.sorted_indices)


        x=h_ordered_tensor.view(B,C,H,W).contiguous()
        x_wh=h_ordered_tensor_wh.view(B,C,W,H).contiguous()

        L = 2 * H * W #inter
        K = 4 #

        if shift_mask != None:
            shift_mask=shift_mask.view(B,1,H,W)
            h_ordered_tensor_wh= apply_hilbert_curve_2d(shift_mask,self.sorted_indices)
            h_ordered_mask_wh= apply_hilbert_curve_2d(torch.transpose(shift_mask, dim0=2, dim1=3).contiguous(),self.sorted_indices)
            h_ordered_tensor_wh=h_ordered_tensor_wh.view(B,1,H,W).contiguous()
            h_ordered_mask_wh=h_ordered_mask_wh.view(B,1,W,H).contiguous()
            shift_mask_hwwh = torch.stack([self.merge_x(h_ordered_tensor_wh), self.merge_x(h_ordered_mask_wh)], dim=1).view(B//2, 2, 1, L) # B//2,2,C,2L,  horizon,vertical
            shift_mask_hwwh_reverse =  torch.flip(shift_mask_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,2L,
            shift_mask = torch.cat([shift_mask_hwwh,shift_mask_hwwh_reverse], dim=1) # reverse# B//2,4,C,2L,
            shift_mask = shift_mask.float().view(B//2, -1, L).unsqueeze(-2).to(x.device) # B//2 , K , 1, L


        B = B // 2


        x_hwwh = torch.stack([self.merge_x(x), self.merge_x(x_wh)], dim=1).view(B, 2, -1, L) # B//2,2,C,2L,  horizon,vertical
        x_hwwh_reverse =  torch.flip(x_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,2L,
        xs = torch.cat([x_hwwh,x_hwwh_reverse], dim=1) # reverse# B//2,4,C,2L,

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) ## projection
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L) # B, 4 *c , L
        Bs = Bs.float().view(B, K, -1, L) # B, 4, d_state , L

        Cs = Cs.float().view(B, K, -1, L)# B, 4, d_state , L
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float()

        if dt_projs_bias is not None:
            dts = dts + dt_projs_bias[...,None].float()
        if self.delta_softplus is True :
            dts = F.softplus(dts)
        if shift_mask != None:
            dts= dts*shift_mask #e 0이면 이전 스테이트에 유지 # (B, K, -1, L) * B , K , 1, L

        dts = dts.contiguous().float().view(B, -1, L) # B, 4 *d , L

        out_y = self.selective_scan(
            xs.contiguous(), #u
            dts.contiguous(),
            As.contiguous(),
            Bs.contiguous(),
            Cs.contiguous(),
            Ds.contiguous(),
            #z=None,
            delta_bias=None,
            delta_softplus=False,
            #return_last_state=False,
        ).view(B, K, -1, L)

        assert out_y.dtype == torch.float

        y= out_y[:, 0]
        wh_y= out_y[:, 1] #B//2,C,2L

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        invwh_y = inv_y[:, 1]#B//2,C,2L

        wh_y = wh_y+ invwh_y # B//2,C,2L

        wh_y= self.unmerge_x(wh_y)# B,C,L
        wh_y = reverse_hilbert_curve_2d(wh_y, self.inverse_indices , W,H)#.view(batch, channel, height, width)
        wh_y = torch.transpose(wh_y,2,3).contiguous()

        y=  out_y[:, 0]+ inv_y[:, 0]
        y= self.unmerge_x(y) # B,C,L
        y = reverse_hilbert_curve_2d(y,self.inverse_indices,H,W)#.view(batch, channel, height, width)

        y= y+wh_y
        y= y.permute(0,2,3,1).contiguous()

        return y


    def window_partition(self,x, window_size):
        B, C, H, W = x.shape
        x=x.permute(0,2,3,1)
        x = x.view(B,H // window_size[0], window_size[0], W // window_size[1], window_size[1],C)
        windows = (
            x.permute(0, 1, 3,2,4,5).contiguous().view(-1, window_size[0],window_size[1],C)
        )
        return windows.permute(0,3,1,2)
    def window_reverse(self,windows, window_size, H, W):
        nwB, w0,w1, C = windows.shape
        #windows = windows.view(-1, window_size[0], window_size[1], C)
        B = int(nwB / (H * W / window_size[0] / window_size[1]))
        x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x



    def forward(self, x: torch.Tensor,shift_mask=None, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        x = self.window_partition(x, self.window_size) # B*NW,C,H,W

        y = self.forward_core(x, shift_mask ) #b,h,w,c
        assert y.dtype == torch.float32

        y = self.out_norm(y)
        y= self.window_reverse(y,self.window_size,H,W)#B,H,W,C
        y = y * F.silu(z)

        out = self.out_proj(y)#B,H,W,C

        if self.dropout is not None:
            out = self.dropout(out)
        return out




###############


def _binary_repr(num: int, width: int) -> str:
    """Return a binary string representation of `num` zero padded to `width`
    bits."""
    return format(num, 'b').zfill(width)

class HilbertCurve:

    def __init__(
        self,
        p: Union[int, float],
        n: Union[int, float],
        n_procs: int=0,
    ) -> None:

        """Initialize a hilbert curve with,
        Args:
            p (int or float): iterations to use in constructing the hilbert curve.
                if float, must satisfy p % 1 = 0
            n (int or float): number of dimensions.
                if float must satisfy n % 1 = 0
            n_procs (int): number of processes to use
                0 = dont use multiprocessing
               -1 = use all available threads
                any other positive integer = number of processes to use
        """
        if (p % 1) != 0:
            raise TypeError("p is not an integer and can not be converted")
        if (n % 1) != 0:
            raise TypeError("n is not an integer and can not be converted")
        if (n_procs % 1) != 0:
            raise TypeError("n_procs is not an integer and can not be converted")

        self.p = int(p)
        self.n = int(n)

        if self.p <= 0:
            raise ValueError('p must be > 0 (got p={} as input)'.format(p))
        if self.n <= 0:
            raise ValueError('n must be > 0 (got n={} as input)'.format(n))

        # minimum and maximum distance along curve
        self.min_h = 0
        self.max_h = 2**(self.p * self.n) - 1

        # minimum and maximum coordinate value in any dimension
        self.min_x = 0
        self.max_x = 2**self.p - 1

        # set n_procs
        n_procs = int(n_procs)
        if n_procs == -1:
            self.n_procs = multiprocessing.cpu_count()
        elif n_procs == 0:
            self.n_procs = 0
        elif n_procs > 0:
            self.n_procs = n_procs
        else:
            raise ValueError(
                'n_procs must be >= -1 (got n_procs={} as input)'.format(n_procs))


    def _hilbert_integer_to_transpose(self, h: int) -> List[int]:
        """Store a hilbert integer (`h`) as its transpose (`x`).

        Args:
            h (int): integer distance along hilbert curve

        Returns:
            x (list): transpose of h
                (n components with values between 0 and 2**p-1)
        """
        h_bit_str = _binary_repr(h, self.p*self.n)
        x = [int(h_bit_str[i::self.n], 2) for i in range(self.n)]
        return x


    def _transpose_to_hilbert_integer(self, x: Iterable[int]) -> int:
        """Restore a hilbert integer (`h`) from its transpose (`x`).
        Args:
            x (list): transpose of h
                (n components with values between 0 and 2**p-1)
        Returns:
            h (int): integer distance along hilbert curve
        """
        x_bit_str = [_binary_repr(x[i], self.p) for i in range(self.n)]
        h = int(''.join([y[i] for i in range(self.p) for y in x_bit_str]), 2)
        return h


    def point_from_distance(self, distance: int) -> Iterable[int]:
        """Return a point in n-dimensional space given a distance along a hilbert curve.

        Args:
            distance (int): integer distance along hilbert curve

        Returns:
            point (iterable of ints): an n-dimensional vector of lengh n where
            each component value is between 0 and 2**p-1.
        """
        x = self._hilbert_integer_to_transpose(int(distance))
        z = 2 << (self.p-1)

        # Gray decode by H ^ (H/2)
        t = x[self.n-1] >> 1
        for i in range(self.n-1, 0, -1):
            x[i] ^= x[i-1]
        x[0] ^= t

        # Undo excess work
        q = 2
        while q != z:
            p = q - 1
            for i in range(self.n-1, -1, -1):
                if x[i] & q:
                    # invert
                    x[0] ^= p
                else:
                    # exchange
                    t = (x[0] ^ x[i]) & p
                    x[0] ^= t
                    x[i] ^= t
            q <<= 1

        return x


    def points_from_distances(
        self,
        distances: Iterable[int],
        match_type: bool=False,
    ) -> Iterable[Iterable[int]]:
        """Return points in n-dimensional space given distances along a hilbert curve.

        Args:
            distances (iterable of int): iterable of integer distances along hilbert curve
            match_type (bool): if True, make type(points) = type(distances)

        Returns:
            points (iterable of iterable of ints): an iterable of n-dimensional vectors
                where each vector has lengh n and component values between 0 and 2**p-1.
                if match_type=False will be list of lists else type(points) = type(distances)
        """
        for ii, dist in enumerate(distances):
            if (dist % 1) != 0:
                raise TypeError(
                    "all values in distances must be int or floats that are convertible to "
                    "int but found distances[{}]={}".format(ii, dist))
            if dist > self.max_h:
                raise ValueError(
                    "all values in distances must be <= 2**(p*n)-1={} but found "
                    "distances[{}]={} ".format(self.max_h, ii, dist))
            if dist < self.min_h:
                raise ValueError(
                    "all values in distances must be >= {} but found distances[{}]={} "
                    "".format(self.min_h, ii, dist))

        if self.n_procs == 0:
            points = []
            for distance in distances:
                x = self.point_from_distance(distance)
                points.append(x)
        else:
            with Pool(self.n_procs) as p:
                points = p.map(self.point_from_distance, distances)

        if match_type:
            if isinstance(distances, np.ndarray):
                points = np.array(points, dtype=distances.dtype)
            else:
                target_type = type(distances)
                points = target_type([target_type(vec) for vec in points])

        return points


    def distance_from_point(self, point: Iterable[int]) -> int:
        """Return distance along the hilbert curve for a given point.
        Args:
            point (iterable of ints): an n-dimensional vector where each component value
                is between 0 and 2**p-1.
        Returns:
            distance (int): integer distance along hilbert curve
        """
        point = [int(el) for el in point]

        m = 1 << (self.p - 1)

        # Inverse undo excess work
        q = m
        while q > 1:
            p = q - 1
            for i in range(self.n):
                if point[i] & q:
                    point[0] ^= p
                else:
                    t = (point[0] ^ point[i]) & p
                    point[0] ^= t
                    point[i] ^= t
            q >>= 1

        # Gray encode
        for i in range(1, self.n):
            point[i] ^= point[i-1]
        t = 0
        q = m
        while q > 1:
            if point[self.n-1] & q:
                t ^= q - 1
            q >>= 1
        for i in range(self.n):
            point[i] ^= t

        distance = self._transpose_to_hilbert_integer(point)
        return distance


    def distances_from_points(
        self,
        points: Iterable[Iterable[int]],
        match_type: bool=False,
    ) -> Iterable[int]:
        """Return distances along the hilbert curve for a given set of points.

        Args:
            points (iterable of iterable of ints): an iterable of n-dimensional vectors
                where each vector has lengh n and component values between 0 and 2**p-1.
            match_type (bool): if True, make type(distances) = type(points)

        Returns:
            distances (iterable of int): iterable of integer distances along hilbert curve
              the return type will match the type used for points.
        """
        for ii, point in enumerate(points):

            if len(point) != self.n:
                raise ValueError(
                    "all vectors in points must have length n={} "
                    "but found points[{}]={}".format(self.n, ii, point))

            if any(elx > self.max_x for elx in point):
                raise ValueError(
                    "all coordinate values in all vectors in points must be <= 2**p-1={} "
                    "but found points[{}]={}".format(self.max_x, ii, point))

            if any(elx < self.min_x for elx in point):
                raise ValueError(
                    "all coordinate values in all vectors in points must be > {} "
                    "but found points[{}]={}".format(self.min_x, ii, point))

            if any((elx % 1) != 0 for elx in point):
                raise TypeError(
                    "all coordinate values in all vectors in points must be int or floats "
                    "that are convertible to int but found points[{}]={}".format(ii, point))

        if self.n_procs == 0:
            distances = []
            for point in points:
                distance = self.distance_from_point(point)
                distances.append(distance)
        else:
            with Pool(self.n_procs) as p:
                distances = p.map(self.distance_from_point, points)

        if match_type:
            if isinstance(points, np.ndarray):
                distances = np.array(distances, dtype=points.dtype)
            else:
                target_type = type(points)
                distances = target_type(distances)

        return distances


    def __str__(self):
        return f"HilbertCruve(p={self.p}, n={self.n}, n_procs={self.n_procs})"


    def __repr__(self):
        return self.__str__()

def apply_hilbert_curve_2d(tensor,sorted_indices):
    """
    입력:
        tensor: (B, C, H, W) 형태의 텐서
    출력:
        hilbert_tensor: 힐베르트 곡선 순서로 재배열된 텐서, shape은 (B, C, N)
        inverse_indices: 원래 순서로 복원하기 위한 인덱스 배열
    """
    B, C, H, W = tensor.shape
    tensor_flat = tensor.view(B,  C, -1) # (B,K, C, H*W)
    hilbert_tensor = tensor_flat[: , : , sorted_indices] # (B,K, C,L)

    return hilbert_tensor.contiguous()

def reverse_hilbert_curve_2d(hilbert_tensor, inverse_indices, H, W):
    """
    입력:
        hilbert_tensor: 힐베르트 곡선 순서로 정렬된 텐서, shape은 (B, C, N)
        inverse_indices: 원래 순서로 복원하기 위한 인덱스 배열
        H, W: 원래 이미지의 높이와 너비
    출력:
        tensor: 원래의 (B, C, H, W) 형태의 텐서
    """
    B, C, N = hilbert_tensor.shape
    # 원래 순서로 재배열
    tensor_flat = hilbert_tensor[:, :,inverse_indices] # (B, K,C, N)
    # 원래의 이미지 형태로 변환
    tensor = tensor_flat.view(B, C, H, W)
    return tensor.contiguous()


##############


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=3,squeeze_factor=4):
        super(CAB, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(in_chans)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x).permute(0, 3, 1, 2).contiguous()
        x = self.proj(x)
        return x



def mlp(in_c,h_c,out_c,k=1,s=1,p=0):
    return nn.Sequential(
        nn.Conv2d(in_c,h_c,1,1,0),
        nn.PReLU(h_c),
        nn.Conv2d(h_c,out_c,1,1,0)
        )

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, depths=2,act_layer=nn.PReLU):
        super().__init__()
        layers = []
        for i in range(depths):
            if i == 0:
                layers.append(nn.Conv2d(in_dim, out_dim, 3,1,1))
            else:
                layers.append(nn.Conv2d(out_dim, out_dim, 3,1,1))
            layers.extend([
                act_layer(out_dim),
            ])
        self.conv = nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        return x




class Head(nn.Module):
    def __init__(self, in_planes, scale, c, in_else=17):
        super(Head, self).__init__()
        self.upsample = nn.Sequential(nn.PixelShuffle(2), nn.PixelShuffle(2))
        self.scale = scale
        self.conv = nn.Sequential(
                                  conv(in_planes*2 // (4*4) + in_else, c),
                                  conv(c, c),
                                  conv(c, 5),
                                  )

    def forward(self, motion_feature, x, flow): # /16 /8 /4
        motion_feature = self.upsample(motion_feature) #/4 /2 /1
        if self.scale != 4:
            x = F.interpolate(x, scale_factor = 4. / self.scale, mode="bilinear", align_corners=False)
        if flow != None:
            if self.scale != 4:
                flow = F.interpolate(flow, scale_factor = 4. / self.scale, mode="bilinear", align_corners=False) * 4. / self.scale
            x = torch.cat((x, flow), 1)

        x = self.conv(torch.cat([motion_feature, x], 1))
        if self.scale != 4:
            x = F.interpolate(x, scale_factor = self.scale // 4, mode="bilinear", align_corners=False)
            flow = x[:, :4] * (self.scale // 4)
        else:
            flow = x[:, :4]
        mask = x[:, 4:5]
        return flow, mask




def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0]*window_size[1], C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    nwB, N, C = windows.shape
    windows = windows.view(-1, window_size[0], window_size[1], C)
    B = int(nwB / (H * W / window_size[0] / window_size[1]))
    x = windows.view(
        B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def pad_if_needed(x, size, window_size):
    b, h, w, c = size
    pad_h = math.ceil(h / window_size[0]) * window_size[0] - h
    pad_w = math.ceil(w / window_size[1]) * window_size[1] - w
    if pad_h > 0 or pad_w > 0:  # center-pad the feature on H and W axes
        img_mask = torch.ones(b,h,w,1)  # 1 H W 1
        img_mask = nn.functional.pad(img_mask,
            (0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
        )
        return nn.functional.pad(
            x,
            (0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
        ), img_mask # b,h,w,1

    return x, None


def depad_if_needed(x, size, window_size):
    n, h, w, c = size
    pad_h = math.ceil(h / window_size[0]) * window_size[0] - h
    pad_w = math.ceil(w / window_size[1]) * window_size[1] - w
    if pad_h > 0 or pad_w > 0:  # remove the center-padding on feature
        return x[:, pad_h // 2 : pad_h // 2 + h, pad_w // 2 : pad_w // 2 + w, :].contiguous()
    return x





# def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
#     return nn.Sequential(
#         nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
#                   padding=padding, dilation=dilation, bias=True),
#         nn.PReLU(out_planes)
#         )

# class Conv2(nn.Module):
#     def __init__(self, in_planes, out_planes,kernel_size=3, stride=2,padding=1):
#         super(Conv2, self).__init__()
#         self.conv1 = conv(in_planes, out_planes, kernel_size, stride, padding)
#         self.conv2 = conv(out_planes, out_planes, 3, 1, padding)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         return x



# def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
#     return nn.Sequential(
#         torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1, bias=True),
#         nn.PReLU(out_planes)
#         )