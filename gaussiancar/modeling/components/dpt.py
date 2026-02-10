from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck

BottleneckBlock = lambda x: Bottleneck(x, x//4)


class ResidualConvUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.conv1 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.conv2 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out
    

class FeatureFusionBlock(nn.Module):
    def __init__(self, channels, expand: bool = False, align_corners: bool = True):
        super().__init__()
        self.channels = channels
        self.expand = expand
        self.align_corners = align_corners
        self.out_channels = channels // 2 if expand else channels

        self.res_conv_unit1 = ResidualConvUnit(self.channels)
        self.res_conv_unit2 = ResidualConvUnit(self.channels)
        self.projection = nn.Conv2d(
            self.channels, self.out_channels, kernel_size=1, bias=True,
        )

    def forward(self, *inputs):
        x = inputs[0]
        if len(inputs) == 2:
            if x.shape[2:] != inputs[1].shape[2:]:
                res = F.interpolate(
                    inputs[1],
                    size=(x.shape[2], x.shape[3]),
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
            else:
                res = inputs[1]
            x = x + self.res_conv_unit1(res)
        x = self.res_conv_unit2(x)
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=self.align_corners,
        )
        x = self.projection(x)
        return x


class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels: list[int, int, int, int] | int,
        post_process_channels: list[int, int, int, int],
        channels: int = 256,
        backbone_type: Literal["convnext", "vit"] = "vit",
        expand_channels=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.post_process_channels = [
            channel * (2**i) if expand_channels else channel
            for i, channel in enumerate(post_process_channels)
        ]
        self.channels = channels
        self.backbone_type = backbone_type
        self.expand_channels = expand_channels
        assert backbone_type in ["convnext", "vit"], "Unsupported backbone type"

        self.make_all_reassemble_blocks()
        self.fusion_4 = FeatureFusionBlock(channels)
        self.fusion_8 = FeatureFusionBlock(channels)
        self.fusion_16 = FeatureFusionBlock(channels)
        self.fusion_32 = FeatureFusionBlock(channels)
        self.output_conv = nn.Sequential(
            BottleneckBlock(channels),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def make_reassemble_block(self, stage_idx, backbone_type="vit"):
        """
        Create a reassemble block for a given stage.
        
        Args:
            stage_idx (int): Index of the stage (0 for _4, 1 for _8, 2 for _16, 3 for _32)
            backbone_type (str): Type of backbone ("vit" or other)
        
        Returns:
            nn.Sequential: The reassemble block
        """
        # Create the base sequential block
        block = nn.Sequential(
            nn.BatchNorm2d(self.in_channels[stage_idx]),
            nn.Conv2d(
                self.in_channels[stage_idx],
                self.post_process_channels[stage_idx],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Identity(),
            nn.Conv2d(
                self.post_process_channels[stage_idx],
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(self.channels),
            nn.GELU(),
        )
        
        # If backbone is vit, we need to create multi-scale features.
        if backbone_type == "vit":
            if stage_idx == 0:  # reassemble_4
                block[2] = nn.ConvTranspose2d(
                    self.post_process_channels[0],
                    self.post_process_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                )
            elif stage_idx == 1:  # reassemble_8
                block[2] = nn.ConvTranspose2d(
                    self.post_process_channels[1],
                    self.post_process_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                )
            elif stage_idx == 2:  # reassemble_16
                block[2] = nn.Identity()
            elif stage_idx == 3:  # reassemble_32
                block[2] = nn.Conv2d(
                    self.post_process_channels[3],
                    self.post_process_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
        
        return block

    def make_all_reassemble_blocks(self):
        """
        Create all reassemble blocks using the automated function.
        """
        stage_names = ['reassemble_4', 'reassemble_8', 'reassemble_16', 'reassemble_32']

        for i, stage_name in enumerate(stage_names):
            setattr(self, stage_name, self.make_reassemble_block(i, self.backbone_type))


    def forward(self, inputs):
        x1, x2, x3, x4 = inputs

        x_4 = self.reassemble_4(x1)
        x_8 = self.reassemble_8(x2)
        x_16 = self.reassemble_16(x3)
        x_32 = self.reassemble_32(x4)

        path_4 = self.fusion_32(x_32)
        path_3 = self.fusion_16(path_4, x_16)
        path_2 = self.fusion_8(path_3, x_8)
        path_1 = self.fusion_4(path_2, x_4)

        output = self.output_conv(path_1)
        return output


if __name__ == "__main__":

    in_shape = (128, 128, 128, 128)
    out_shape = 256
    model = DPTHead(
        in_channels=[384, 384, 384, 384],
        post_process_channels=[48, 96, 192, 384],
        channels=256,
        out_channels=1,
        backbone_type="vit",
    ).cuda()

    # Example input
    x1 = torch.randn(1, 384, 512 // 16, 512 // 16).cuda()
    x2 = torch.randn(1, 384, 512 // 16, 512 // 16).cuda()
    x3 = torch.randn(1, 384, 512 // 16, 512 // 16).cuda()
    x4 = torch.randn(1, 384, 512 // 16, 512 // 16).cuda()


    # # Example input
    # x1 = torch.randn(1, 384, 512 // 4, 512 // 4).cuda()
    # x2 = torch.randn(1, 384, 512 // 8, 512 // 8).cuda()
    # x3 = torch.randn(1, 384, 512 // 16, 512 // 16).cuda()
    # x4 = torch.randn(1, 384, 512 // 32, 512 // 32).cuda()

    print(f"Input shapes: {x1.shape}, {x2.shape}, {x3.shape}, {x4.shape}")

    output = model([x1, x2, x3, x4])
    print(output.shape)  # Expected output shape: (1, out_shape, H', W')
