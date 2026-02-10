import torch
import torch.nn as nn
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from gaussiancar.modeling.components.cmx import FeatureFusionModule, FeatureRectifyModule
from gaussiancar.modeling.components.conv import Conv, ResBlock


class CMXFuser(nn.Module):
    def __init__(self, embed_dims: int = 256) -> None:
        super().__init__()
        self.embed_dims = embed_dims

        # Fuser blocks.
        self.frm_1 = FeatureRectifyModule(embed_dims, reduction=1)
        self.ffm_1 = FeatureFusionModule(embed_dims, reduction=1, num_heads=2, norm_layer=nn.BatchNorm2d)

        self.frm_2 = FeatureRectifyModule(embed_dims, reduction=1)
        self.ffm_2 = FeatureFusionModule(embed_dims, reduction=1, num_heads=2, norm_layer=nn.BatchNorm2d)

        self.frm_3 = FeatureRectifyModule(embed_dims, reduction=1)
        self.ffm_3 = FeatureFusionModule(embed_dims, reduction=1, num_heads=2, norm_layer=nn.BatchNorm2d)

        self.frm_4 = FeatureRectifyModule(embed_dims, reduction=1)
        self.ffm_4 = FeatureFusionModule(embed_dims, reduction=1, num_heads=2, norm_layer=nn.BatchNorm2d)

        # Downsample conv layers.
        # Radar branch.
        self.radar_down_conv1 = nn.Sequential(
            Conv(embed_dims, embed_dims, k=3, s=2, p=1),
            ResBlock(embed_dims, embed_dims, k=3, s=1, p=1),
            ResBlock(embed_dims, embed_dims, k=3, s=1, p=1)
        )
        self.radar_down_conv2 = nn.Sequential(
            Conv(embed_dims, embed_dims, k=3, s=2, p=1),
            ResBlock(embed_dims, embed_dims, k=3, s=1, p=1),
            ResBlock(embed_dims, embed_dims, k=3, s=1, p=1),
            ResBlock(embed_dims, embed_dims, k=3, s=1, p=1),
            ResBlock(embed_dims, embed_dims, k=3, s=1, p=1)
        )
        self.radar_down_conv3 = nn.Sequential(
            Conv(embed_dims, embed_dims, k=3, s=2, p=1),
            ResBlock(embed_dims, embed_dims, k=3, s=1, p=1),
            ResBlock(embed_dims, embed_dims, k=3, s=1, p=1),
            ResBlock(embed_dims, embed_dims, k=3, s=1, p=1),
            ResBlock(embed_dims, embed_dims, k=3, s=1, p=1)
        )

        # Camera branch.
        self.camera_down_conv1 = nn.Sequential(
            Conv(embed_dims, embed_dims, k=3, s=2, p=1),
            ResBlock(embed_dims, embed_dims, k=3, s=1, p=1),
            ResBlock(embed_dims, embed_dims, k=3, s=1, p=1)
        )
        self.camera_down_conv2 = nn.Sequential(
            Conv(embed_dims, embed_dims, k=3, s=2, p=1),
            ResBlock(embed_dims, embed_dims, k=3, s=1, p=1),
            ResBlock(embed_dims, embed_dims, k=3, s=1, p=1),
            ResBlock(embed_dims, embed_dims, k=3, s=1, p=1),
            ResBlock(embed_dims, embed_dims, k=3, s=1, p=1)
        )
        self.camera_down_conv3 = nn.Sequential(
            Conv(embed_dims, embed_dims, k=3, s=2, p=1),
            ResBlock(embed_dims, embed_dims, k=3, s=1, p=1),
            ResBlock(embed_dims, embed_dims, k=3, s=1, p=1),
            ResBlock(embed_dims, embed_dims, k=3, s=1, p=1),
            ResBlock(embed_dims, embed_dims, k=3, s=1, p=1)
        )


    def forward(
        self,
        camera_feats: torch.Tensor,
        radar_feats: torch.Tensor
    ) -> torch.Tensor:

        # Stage 0 of fuser.
        c1, r1 = self.frm_1(camera_feats, radar_feats)
        cr1 = self.ffm_1(c1, r1)

        # Stage 1 of fuser.
        c2 = self.camera_down_conv1(c1)
        r2 = self.radar_down_conv1(r1)
        c2, r2 = self.frm_2(c2, r2)
        cr2 = self.ffm_2(c2, r2)

        # Stage 2 of fuser.
        c3 = self.camera_down_conv2(c2)
        r3 = self.radar_down_conv2(r2)
        c3, r3 = self.frm_3(c3, r3)
        cr3 = self.ffm_3(c3, r3)

        # Stage 3 of fuser.
        c4 = self.camera_down_conv3(c3)
        r4 = self.radar_down_conv3(r3)
        c4, r4 = self.frm_4(c4, r4)
        cr4 = self.ffm_4(c4, r4)

        return [cr1, cr2, cr3, cr4]
    

if __name__ == "__main__":
    fuser = CMXFuser(embed_dims=256)
    cam_feats = torch.randn(2, 256, 200, 200)
    rad_feats = torch.randn(2, 256, 200, 200)
    out = fuser(cam_feats, rad_feats)
    for i, v in enumerate(out):
        print(f"stage_{i}", v.shape)