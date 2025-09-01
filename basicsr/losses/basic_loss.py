import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss





@LOSS_REGISTRY.register()
class SobelLoss(nn.Module):
    """Sobel Loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(SobelLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """

        filter_x = torch.tensor([[1., 2. , 1.], [0., 0., 0.], [-1., -2. , -1.]]).to(target.device)
        filter_y = torch.tensor([[1., 0 , -1.], [2., 0., -2.], [1., 0. , -1.]]).to(target.device)
        filter_xy = torch.tensor([[0., 1. , 2.], [-1., 0., 1.], [-2., -1. , 0.]]).to(target.device)
        filter_yx = torch.tensor([[2., 1. , 0.], [1., 0., -1.], [0., -1. , -2.]]).to(target.device)

        filter_x = filter_x.view(1,1,3,3).repeat(1, 3, 1, 1)
        filter_y = filter_y.view(1,1,3,3).repeat(1, 3, 1, 1)
        filter_xy = filter_xy.view(1,1,3,3).repeat(1, 3, 1, 1)
        filter_yx = filter_yx.view(1,1,3,3).repeat(1, 3, 1, 1)

        B, T, C, H, W =  pred.shape
        sobel_x_img1 = pred.new_zeros((B,T,C,H,W))
        sobel_y_img1 = pred.new_zeros((B,T,C,H,W))
        sobel_xy_img1 = pred.new_zeros((B,T,C,H,W))
        sobel_yx_img1 = pred.new_zeros((B,T,C,H,W))
        sobel_x_img2 = pred.new_zeros((B,T,C,H,W))
        sobel_y_img2 = pred.new_zeros((B,T,C,H,W))
        sobel_xy_img2 = pred.new_zeros((B,T,C,H,W))
        sobel_yx_img2 = pred.new_zeros((B,T,C,H,W))

        for i in range(T):
            sobel_x_img1[:,i,:,:,:] = F.conv2d(pred[:,i,:,:,:], filter_x, stride=1, padding=1)
            sobel_y_img1[:,i,:,:,:] = F.conv2d(pred[:,i,:,:,:], filter_y, stride=1, padding=1)
            sobel_xy_img1[:,i,:,:,:] = F.conv2d(pred[:,i,:,:,:], filter_xy, stride=1, padding=1)
            sobel_yx_img1[:,i,:,:,:] = F.conv2d(pred[:,i,:,:,:], filter_yx, stride=1, padding=1)

            sobel_x_img2[:,i,:,:,:] = F.conv2d(target[:,i,:,:,:], filter_x, stride=1, padding=1)
            sobel_y_img2[:,i,:,:,:] = F.conv2d(target[:,i,:,:,:], filter_y, stride=1, padding=1)
            sobel_xy_img2[:,i,:,:,:] = F.conv2d(target[:,i,:,:,:], filter_xy, stride=1, padding=1)
            sobel_yx_img2[:,i,:,:,:] = F.conv2d(target[:,i,:,:,:], filter_yx, stride=1, padding=1)

        char_loss = self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)
        sobel_loss = self.loss_weight * torch.sum((torch.abs(sobel_x_img1 - sobel_x_img2) + torch.abs(sobel_y_img1 - sobel_y_img2) + \
                torch.abs(sobel_xy_img1 - sobel_xy_img2) + torch.abs(sobel_yx_img1 - sobel_yx_img2))) / 4.0
        loss = char_loss + 0.1 * sobel_loss
        return loss






@LOSS_REGISTRY.register()
class CharbonnierTrajectoryLoss(nn.Module):
    """Charbonnier trajectory loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierTrajectoryLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        # print('pred[:-2]',pred[:-2][0].shape, target.shape)
        pred_img = pred[:-2][0]
        pred_local = pred[-2] / torch.max(pred[-2])
        target_local = pred[-1] / torch.max(pred[-1])
        char_loss = self.loss_weight * charbonnier_loss(pred_img, target, weight, eps=self.eps, reduction=self.reduction)
        traj_loss = self.loss_weight * mse_loss(pred_local, target_local, weight, reduction=self.reduction) # + self.eps
        total_loss = char_loss + 0.1 * traj_loss
        return total_loss

