import torch
import torch.nn as nn
import torch.nn.functional as F

from gaussiancar.ops.ptv3.model import PointTransformerV3


class PointsToGaussians(nn.Module):
    def __init__(
        self,
        in_channels: int = 51,
        hidden_channels: int = 256,
        out_channels: int = 128,
        max_points: int = 3500,
        opacity_bias_init: float = 3.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.max_points = max_points

        self.encoder = PointTransformerV3(
            in_channels=in_channels,
            enc_depths=(1, 1, 1, 1, 1),
            enc_num_head=(1, 2, 4, 8, 16),
            enc_patch_size=(64, 64, 64, 64, 64),
            enc_channels=(64, 64, 128, 128, self.hidden_channels),
            dec_depths=(1, 1, 1, 1),
            dec_channels=(self.hidden_channels, 128, 64, 64),
            dec_num_head=(4, 4, 4, 8),
            dec_patch_size=(64, 64, 64, 64),
            mlp_ratio=4,
            qkv_bias=True,
        )

        self.feats_mlp = self._create_mlp(
            self.hidden_channels,
            self.out_channels,
            hidden_channels=2 * self.hidden_channels,
            dropout=0.1,
        )
        self.offset_mlp = self._create_mlp(
            self.hidden_channels,
            3,
            hidden_channels=2 * self.hidden_channels,
            dropout=0.1,
        )
        self.covs_mlp = self._create_mlp(
            self.hidden_channels,
            6,
            hidden_channels=2 * self.hidden_channels,
            dropout=0.1,
        )
        self.opacities_mlp = self._create_mlp(
            self.hidden_channels,
            1,
            hidden_channels=2 * self.hidden_channels,
            dropout=0.1,
            final_bias_init=opacity_bias_init,
        )

    def _create_mlp(
        self,
        in_channels,
        out_channels,
        hidden_channels=None,
        dropout=0.1,
        final_bias_init=None,
    ) -> nn.Sequential:

        if hidden_channels is None:
            hidden_channels = in_channels * 2

        mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

        if final_bias_init is not None:
            nn.init.constant_(mlp[-1].bias, final_bias_init)

        return mlp

    def forward(self, radar_points: list[torch.Tensor]):

        B = len(radar_points)

        # Extract features from the backbone.
        offset = torch.tensor([i.shape[0] for i in radar_points]).cumsum(0)
        radar_points = torch.cat(radar_points, 0)
        radar_dict = {
            "feat": radar_points[:, 3:].float(),
            "coord": radar_points[:, :3].float(),
            "offset": offset.to(radar_points.device),
            "grid_size": 2.0,
        }
        radar_point_features = self.encoder(radar_dict)

        # Compute the means, covariances, and opacities.
        means = radar_point_features["coord"].float()
        offsets = self.offset_mlp(radar_point_features["feat"]).float()
        covs = self.covs_mlp(radar_point_features["feat"]).float()
        covs = self.make_valid_covariances(covs)
        opacities = torch.sigmoid(self.opacities_mlp(radar_point_features["feat"]).float())
        features = self.feats_mlp(radar_point_features["feat"]).float()

        # Recompute the means in 3D space.
        means = means + offsets

        # Reconvert to list of tensors.
        means_out = torch.zeros(B, self.max_points, 3, device=means.device)
        offsets_out = torch.zeros(B, self.max_points, 3, device=means.device)
        features_out = torch.zeros(B, self.max_points, 128, device=means.device)
        covs_out = torch.zeros(B, self.max_points, 6, device=means.device)
        opacities_out = torch.zeros(B, self.max_points, 1, device=means.device)

        # Fill the output tensors with the computed values.
        batch_offsets = [0] + radar_dict["offset"].tolist()
        for b, start, end in zip(range(B), batch_offsets[:-1], batch_offsets[1:]):
            means_out[b, :end - start] = means[start:end]
            offsets_out[b, :end - start] = offsets[start:end]
            features_out[b, :end - start] = features[start:end]
            covs_out[b, :end - start] = covs[start:end]
            opacities_out[b, :end - start] = opacities[start:end]

        return {
            "centers": means_out,
            "offsets": offsets_out,
            "features": features_out,
            "covariances": covs_out,
            "opacities": opacities_out,
        }
    
    def make_valid_covariances(self, covs_raw):
        """Convert raw MLP output to valid covariance matrices"""
        # Create individual components without in-place operations
        xx = F.softplus(covs_raw[:, 0]) + 1e-4
        yy = F.softplus(covs_raw[:, 3]) + 1e-4  
        zz = F.softplus(covs_raw[:, 5]) + 1e-4
        
        # Off-diagonal elements
        sqrt_xx_yy = torch.sqrt(xx * yy)
        sqrt_xx_zz = torch.sqrt(xx * zz)
        sqrt_yy_zz = torch.sqrt(yy * zz)
        
        xy = torch.tanh(covs_raw[:, 1]) * sqrt_xx_yy * 0.9
        xz = torch.tanh(covs_raw[:, 2]) * sqrt_xx_zz * 0.9
        yz = torch.tanh(covs_raw[:, 4]) * sqrt_yy_zz * 0.9
        
        # Stack all components at once instead of in-place assignment
        covs = torch.stack([xx, xy, xz, yy, yz, zz], dim=1)
        
        return covs
    

