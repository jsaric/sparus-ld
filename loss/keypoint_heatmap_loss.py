import torch
import torch.nn as nn
import torch.nn.functional as F
from data.keypoint_utils import generate_channel_heatmap


class KeypointHeatmapLoss(nn.Module):
    def __init__(self, sigma=4, top_k_percent_pixels=1.0, criterion='mse'):
        super().__init__()
        self.sigma = sigma
        self.top_k_percent_pixels = top_k_percent_pixels
        if criterion == 'mse':
            self.criterion = F.mse_loss
        elif criterion == 'bce':
            self.criterion = F.binary_cross_entropy_with_logits
        else:
            raise NotImplementedError(f'Criterion {criterion} not implemented')
        
    def forward(self, pred_heatmap, gt_keypoints):
        """
        :param pred_heatmap: (B, K, H, W)
        :param gt_keypoints: (B, K, 2)
        :return:
        """
        B, K, H, W = pred_heatmap.shape
        with torch.no_grad():
            gt_heatmap = generate_channel_heatmap((H, W), gt_keypoints, self.sigma, device=pred_heatmap.device)
        if self.top_k_percent_pixels == 1.0:
            return self.criterion(pred_heatmap, gt_heatmap)
        else:
            pixel_loss = self.criterion(pred_heatmap, gt_heatmap, reduction='none')
            num_pixels = H * W
            num_pixels_to_keep = int(num_pixels * self.top_k_percent_pixels)
            pixel_loss = pixel_loss.reshape(B, K, -1)
            pixel_loss, _ = torch.topk(pixel_loss, num_pixels_to_keep, dim=-1)
            return pixel_loss.mean()
