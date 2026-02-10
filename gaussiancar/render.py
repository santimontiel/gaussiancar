import math
import torch
import torch.nn as nn
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


class BEVCamera:
    def __init__(self, x_range=(-50, 50), y_range=(-50, 50), image_size=(200, 200)):
        # Orthographic projection parameters
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.image_width = image_size[1]
        self.image_height = image_size[0]

        # Set up FoV to cover the range [-50, 50] for both X and Y
        self.FoVx = (self.x_max - self.x_min)  # Width of the scene in world coordinates
        self.FoVy = (self.y_max - self.y_min)  # Height of the scene in world coordinates

        # Camera position: placed above the scene, looking down along Z-axis
        self.camera_center = torch.tensor([0, 0, 0], dtype=torch.float32)  # High above Z-axis

        # Orthographic projection matrix for BEV
        self.set_transform()
    
    def set_transform(self, h=200, w=200, h_meters=100, w_meters=100):
        """ Set up an orthographic projection matrix for BEV. """
        # Create an orthographic projection matrix
        sh = h / h_meters
        sw = w / w_meters
        self.world_view_transform = torch.tensor([
            [ 0.,  sh,  0.,         0.],
            [ sw,  0.,  0.,         0.],
            [ 0.,  0.,  0.,         0.],
            [ 0.,  0.,  0.,         0.],
        ], dtype=torch.float32)

        self.full_proj_transform = torch.tensor([
            [ 0., -sh,  0.,          h/2.],
            [-sw,   0.,  0.,         w/2.],
            [ 0.,  0.,  0.,           1.],
            [ 0.,  0.,  0.,           1.],
        ], dtype=torch.float32)

    def set_size(self, h, w):
        self.image_height = h
        self.image_width = w


class GaussianRenderer(nn.Module):
    def __init__(
        self,
        embed_dims,
        threshold=0.05,
        x_min: float = -50,
        x_max: float = 50,
        y_min: float = -50,
        y_max: float = 50,
        bev_h: int = 200,
        bev_w: int = 200,
    ):
        super().__init__()
        self.viewpoint_camera = BEVCamera(
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
            image_size=(bev_h, bev_w),
        )
        self.rasterizer = GaussianRasterizer()
        self.embed_dims = embed_dims
        self.threshold = threshold
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.bev_h = bev_h
        self.bev_w = bev_w

    def forward(self, features, means3D, cov3D, opacities):
        """
        features: b G d
        means3D: b G 3
        uncertainty: b G 6
        opacities: b G 1
        """ 
        b = features.shape[0]
        device = means3D.device
        
        bev_out = []
        mask = (opacities > self.threshold)
        mask = mask.squeeze(-1)
        self.set_render_scale(self.bev_h, self.bev_w)
        self.set_Rasterizer(device)
        for i in range(b):
            rendered_bev, _ = self.rasterizer(
                means3D=means3D[i][mask[i]],
                means2D=None,
                shs=None,  # No SHs used
                colors_precomp=features[i][mask[i]],
                opacities=opacities[i][mask[i]],
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D[i][mask[i]]
            )
            bev_out.append(rendered_bev)
            
        x = torch.stack(bev_out, dim=0) # b d h w
        num_gaussians = (mask.detach().float().sum(1)).mean().cpu()

        return x, num_gaussians
        
    @torch.no_grad()
    def set_Rasterizer(self, device):
        tanfovx = math.tan(self.viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(self.viewpoint_camera.FoVy * 0.5)

        bg_color = torch.zeros((self.embed_dims)).to(device) # self.embed_dims
        # bg_color[-1] = -4
        raster_settings = GaussianRasterizationSettings(
            image_height=int(self.viewpoint_camera.image_height),
            image_width=int(self.viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1,
            viewmatrix=self.viewpoint_camera.world_view_transform.to(device),
            projmatrix=self.viewpoint_camera.full_proj_transform.to(device),
            sh_degree=0,  # No SHs used 
            campos=self.viewpoint_camera.camera_center.to(device),
            prefiltered=False,
            debug=False
        )
        self.rasterizer.set_raster_settings(raster_settings)

    @torch.no_grad()
    def set_render_scale(self, h, w):
        self.viewpoint_camera.set_size(h, w)
        self.viewpoint_camera.set_transform(h, w)
