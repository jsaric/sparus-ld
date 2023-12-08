import torch
from typing import Tuple


def generate_channel_heatmap(
        image_size: Tuple[int, int],
        keypoints: torch.Tensor,
        sigma: float,
        device: torch.device
) -> torch.Tensor:
    """
    Generates heatmap with gaussian blobs for each keypoint, using the given sigma.
    Max operation is used to combine the heatpoints to avoid local optimum surpression.
    Origin is topleft corner and u goes right, v down.
    Args:
        image_size: Tuple(int,int) that specify (H,W) of the heatmap image
        keypoints: a 2D Tensor K x 2,  with K keypoints  (u,v).
        sigma: (float) std deviation of the blobs
        device: the device on which to allocate new tensors
    Returns:
         Torch.tensor:  A Tensor with the combined heatmaps of all keypoints.
    """

    # cast keypoints (center) to ints to make grid align with pixel raster.
    #  Otherwise, the AP metric for  d = 1 will not result in 1
    #  if the gt_heatmaps are used as input.

    assert isinstance(keypoints, torch.Tensor)

    if keypoints.numel() == 0:
        # special case for which there are no keypoints in this channel.
        return torch.zeros(image_size, device=device)

    keypoints = keypoints.clone()
    invalid_keypoints_mask = torch.logical_or(
        torch.logical_or(keypoints[..., 0] <=0, keypoints[..., 0] >= image_size[1]),
        torch.logical_or(keypoints[..., 1] <=0, keypoints[..., 1] >= image_size[0])
    )
    keypoints[invalid_keypoints_mask] = torch.inf

    u_axis = torch.linspace(0, image_size[1] - 1, image_size[1], device=device)
    v_axis = torch.linspace(0, image_size[0] - 1, image_size[0], device=device)
    # create grid values in 2D with x and y coordinate centered aroud the keypoint
    v_grid, u_grid = torch.meshgrid(v_axis, u_axis, indexing="ij")  # v-axis -> dim 0, u-axis -> dim 1

    u_grid = u_grid.unsqueeze(0) - keypoints[..., 0].unsqueeze(-1).unsqueeze(-1)
    v_grid = v_grid.unsqueeze(0) - keypoints[..., 1].unsqueeze(-1).unsqueeze(-1)

    ## create gaussian around the centered 2D grids $ exp ( -0.5 (x**2 + y**2) / sigma**2)$
    heatmap = torch.exp(
        -0.5 * (torch.square(u_grid) + torch.square(v_grid)) / torch.square(torch.tensor([sigma], device=device))
    )
    # heatmap = torch.max(heatmap, dim=0)[0]
    return heatmap


def get_keypoints_from_heatmap(heatmap):
    h, w = heatmap.shape[-2:]
    flattened = heatmap.flatten(-2)
    offsets = flattened.max(dim=-1).indices
    rows = offsets // w
    columns = offsets % w
    keypoint_coords = torch.stack((columns, rows), dim=-1)
    return keypoint_coords
