from typing import Any, Dict

import torch
import torch.nn as nn
import rootutils
from einops import rearrange

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from gaussiancar.render import GaussianRenderer


class GaussianCaR(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        depth_num: int,
        depth_min: int,
        depth_max: int,
        error_tolerance: float,
        opacity_filter: float,
        x_min: float = -50,
        x_max: float = 50,
        y_min: float = -50,
        y_max: float = 50,
        bev_h: int = 200,
        bev_w: int = 200,
        image_encoder: nn.Module = None,
        radar_encoder: nn.Module = None,
        fuser: nn.Module = None,
        decoder: nn.Module = None,
        head: nn.Module = None,
        aux_head: nn.Module = None,
    ) -> None:
        super().__init__()

        # Parameters.
        self.embed_dims = embed_dims
        self.depth_num = depth_num
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.error_tolerance = error_tolerance
        self.opacity_filter = opacity_filter

        # Modules.
        self.image_encoder = image_encoder
        self.radar_encoder = radar_encoder
        self.gs_render_image = GaussianRenderer(
            embed_dims,
            opacity_filter,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            bev_h=bev_h,
            bev_w=bev_w,
        )
        self.gs_render_radar = GaussianRenderer(
            embed_dims,
            opacity_filter,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            bev_h=bev_h,
            bev_w=bev_w,
        )
        self.fuser = fuser
        self.decoder = decoder
        self.head = head
        self.aux_head = aux_head

        # Geometry for PixelsToGaussians.
        bins = self._init_bin_centers()
        self.register_buffer("bins", bins, persistent=False)

    def _init_bin_centers(self):
        """
        Compute the centers of uniformly spaced depth bins within a specified range.

        The method divides the interval [self.depth_min, self.depth_max] into
        self.depth_num equal-width bins, builds the bin edges via a cumulative sum,
        and returns the midpoint of each adjacent pair of edges.

        Returns:
            torch.Tensor: A 1D tensor of length `self.depth_num` containing the
                center value of each uniform depth bin.

        Uses:
            - self.depth_min (float): The starting depth (lower bound of the range).
            - self.depth_max (float): The ending depth (upper bound of the range).
            - self.depth_num (int): The number of uniform bins to create.
        """
        depth_range = self.depth_max - self.depth_min
        interval = depth_range / self.depth_num
        interval = interval * torch.ones((self.depth_num+1))
        interval[0] = self.depth_min
        bin_edges = torch.cumsum(interval, 0)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return bin_centers
    
    @torch.no_grad()
    def _get_pixel_coords_3d(
        self,
        coords_d,
        depth,
        lidar2img,
        img_h=224,
        img_w=480,
    ):
        """
        Convert 2D image coordinates to 3D coordinates using depth information.

        The method creates a 3D coordinate grid by combining 2D image coordinates 
        with depth values, then transforms these coordinates from image space to 
        lidar coordinate system using the inverse transformation matrices.

        Args:
            coords_d (torch.Tensor): 1D tensor of depth values for each depth bin.
            depth (torch.Tensor): Depth tensor with shape containing spatial dimensions.
            lidar2img (torch.Tensor): Transformation matrices from lidar to image space 
                with shape (B, N, 4, 4).
            img_h (int, optional): Image height in pixels. Defaults to 224.
            img_w (int, optional): Image width in pixels. Defaults to 480.

        Returns:
            tuple: A tuple containing:
                - coords3d (torch.Tensor): 3D coordinates in lidar space with shape 
                (B, N, W, H, D, 3).
                - coords_d (torch.Tensor): Input depth coordinates (passed through).
        """
        eps = 1e-5
        
        B, N = lidar2img.shape[:2]
        H, W = depth.shape[-2:]
        coords_h = torch.linspace(0, 1, H, device=depth.device).float() * img_h
        coords_w = torch.linspace(0, 1, W, device=depth.device).float() * img_w

        D = coords_d.shape[0]
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0) # W, H, D, 3
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)
        dtype = lidar2img.dtype
        img2lidars = lidar2img.float().inverse().to(dtype) # b n 4 4

        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
        img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3] # B N W H D 3

        return coords3d, coords_d

    def _pred_depth(self, lidar2img, depth, img_h, img_w, coords_3d=None):
        """
        Predict 3D coordinates and covariance matrices from depth probability distributions.

        The method converts depth probability distributions into predicted 3D coordinates
        by computing weighted averages across depth bins, and estimates uncertainty by
        calculating covariance matrices based on the probability distributions.

        Args:
            lidar2img (torch.Tensor): Transformation matrices from lidar to image space
                with shape (B, N, 4, 4).
            depth (torch.Tensor): Depth logits tensor with shape (B*N, depth_num, H, W).
            img_h (int): Image height in pixels.
            img_w (int): Image width in pixels.
            coords_3d (torch.Tensor, optional): Pre-computed 3D coordinates. If None,
                coordinates will be computed using _get_pixel_coords_3d.

        Returns:
            tuple: A tuple containing:
                - pred_coords_3d (torch.Tensor): Predicted 3D coordinates with shape
                (B*N, H, W, 3).
                - cov (torch.Tensor): Covariance matrices with shape (B*N, H, W, 3, 3).
        """
        # b, n, c, h, w = depth.shape
        if coords_3d is None:
            coords_3d, coords_d = self._get_pixel_coords_3d(self.bins, depth, lidar2img, img_h=img_h, img_w=img_w) # b n w h d 3
            coords_3d = rearrange(coords_3d, 'b n w h d c -> (b n) d h w c')
            
        depth_prob = depth.softmax(1) # (b n) depth h w
        pred_coords_3d = (depth_prob.unsqueeze(-1) * coords_3d).sum(1)  # (b n) h w 3
        
        delta_3d = pred_coords_3d.unsqueeze(1) - coords_3d
        cov = (depth_prob.unsqueeze(-1).unsqueeze(-1) * (delta_3d.unsqueeze(-1) @ delta_3d.unsqueeze(-2))).sum(1)
        scale = (self.error_tolerance ** 2) / 9 
        cov = cov * scale

        return pred_coords_3d, cov

    def forward_features_camera(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        
        # Arrange the input images.
        B, N, _, H, W = batch["image"].shape
        images = batch["image"].flatten(0, 1).contiguous()
        lidar2img = batch["lidar2img"].contiguous()

        # Process images.
        feats = self.image_encoder(images)
        means_3d, covs_3d = self._pred_depth(lidar2img, feats["depth"], H, W)
        means_3d = means_3d + rearrange(feats["offsets"], "(b n) d h w -> (b n) h w d", b=B, n=N)
        covs_3d = covs_3d.flatten(-2, -1)
        covs_3d = torch.cat((covs_3d[..., 0:3], covs_3d[..., 4:6], covs_3d[..., 8:9]), dim=-1)

        feats_3d = rearrange(feats["features"], '(b n) d h w -> b (n h w) d', b=B, n=N)
        means_3d = rearrange(means_3d, '(b n) h w d-> b (n h w) d', b=B, n=N)
        covs_3d = rearrange(covs_3d, '(b n) h w d -> b (n h w) d',b=B, n=N)
        opacities_3d = rearrange(feats["opacity"], '(b n) d h w -> b (n h w) d', b=B, n=N)

        return {
            "features": feats_3d,
            "centers": means_3d,
            "covariances": covs_3d,
            "opacities": opacities_3d,
        }
    
    def forward_features_radar(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        radar_gaussians = self.radar_encoder(batch["radar_points"])
        return radar_gaussians

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:

        # Convert pixels and points to Gaussians.
        camera_gaussians = self.forward_features_camera(batch)
        radar_gaussians = self.forward_features_radar(batch)

        # Rasterize Gaussians to BEV features.
        camera_bev, num_gaussians_cam = self.gs_render_image(
            camera_gaussians["features"],
            camera_gaussians["centers"],
            camera_gaussians["covariances"],
            camera_gaussians["opacities"]
        )
        radar_bev, num_gaussians_radar = self.gs_render_radar(
            radar_gaussians["features"],
            radar_gaussians["centers"],
            radar_gaussians["covariances"],
            radar_gaussians["opacities"]
        )
        
        # Fuse BEV features.
        fused_bev = self.fuser(camera_bev, radar_bev)

        # Decode the fused BEV features.
        out = self.decoder(fused_bev)
        out = self.head(out)
        aux_out = self.aux_head(fused_bev[0])

        return {
            "output": out,
            "aux_output": aux_out,
            "num_gaussians_cam": num_gaussians_cam,
            "num_gaussians_radar": num_gaussians_radar,
            "features_cam": camera_bev,
            "features_radar": radar_bev,
            "features_fused": fused_bev,
        }