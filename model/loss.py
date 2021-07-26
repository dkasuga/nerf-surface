import torch
from torch import nn
from torch.nn import functional as F


class NeuSLoss(nn.Module):
    def __init__(self, eikonal_weight, mask_weight, alpha):
        super().__init__()
        self.eikonal_weight = eikonal_weight
        self.mask_weight = mask_weight
        self.alpha = alpha
        self.l1_loss = nn.L1Loss(reduction='mean')

    def get_rgb_loss(self, rgb_values, rgb_gt):
        rgb_values = rgb_values
        rgb_gt = rgb_gt.reshape(-1, 3)
        # 結局出力も[0.0,1.0]に収める
        rgb_gt = (rgb_gt + 1.0) / 2.0
        # rgb_gt = rgb_gt
        rgb_loss = self.l1_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def forward(self, model_outputs, ground_truth):
        rgb_gt = ground_truth['rgb'].squeeze().cuda()
        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt)
        eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])

        loss = rgb_loss + self.eikonal_weight * eikonal_loss

        return {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
        }
