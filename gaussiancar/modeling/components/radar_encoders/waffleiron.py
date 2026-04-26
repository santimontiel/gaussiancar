import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import knn

# ----------------------------------------------------------------------
# --- Data Transforms
# ----------------------------------------------------------------------
def voxelize(
    pc: torch.Tensor,
    dims=(0, 1, 2),
    voxel_size: float = 0.5,
) -> torch.Tensor:
    pc_dims = pc[:, dims]
    pc_shift = pc_dims - pc_dims.min(dim=0).values
    vox_coords = (pc_shift / voxel_size).long()

    max_per_dim = vox_coords.max(dim=0).values + 1
    strides = torch.ones(len(dims), dtype=torch.long, device=pc.device)
    for i in range(1, len(dims)):
        strides[i] = strides[i - 1] * max_per_dim[i - 1]
    keys = (vox_coords * strides).sum(dim=1)

    _, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    num_unique = int(inverse.max().item()) + 1

    arange = torch.arange(pc.shape[0], device=pc.device)
    sentinel = torch.full((num_unique,), pc.shape[0], dtype=torch.long, device=pc.device)
    ind = torch.scatter_reduce(sentinel, dim=0, index=inverse, src=arange, reduce="amin")

    return pc[ind]

def crop(
    pc: torch.Tensor,
    dims=(0, 1, 2),
    fov=((-5, -5, -5), (5, 5, 5)),
    eps: float = 1e-4,
) -> torch.Tensor:
    mask = None
    for i, d in enumerate(dims):
        temp = (pc[:, d] > fov[0][i] + eps) & (pc[:, d] < fov[1][i] - eps)
        mask = temp if mask is None else mask & temp
    return pc[mask]


def get_occupied_2d_cells(
    pc: torch.Tensor,
    dim_proj: list[int],
    grids_shape: list[tuple[int, int]],
    lut_axis_plane: dict[int, list[int]],
    fov_xyz: torch.Tensor,             # (2, N_dims)
) -> torch.Tensor:
    """Return mapping between 3D points and corresponding 2D cell indices."""
    cell_ind = []
    for dim, grid in zip(dim_proj, grids_shape):
        dims = lut_axis_plane[dim]
        grid_t = torch.tensor(grid, dtype=pc.dtype, device=pc.device)

        # Compute grid resolution
        res = (fov_xyz[1, dims] - fov_xyz[0, dims]) / grid_t

        # Shift and quantize point cloud
        pc_quant = ((pc[:, dims] - fov_xyz[0, dims]) / res).long()

        # Check that points fit on the grid
        min_q = pc_quant.min(dim=0).values
        max_q = pc_quant.max(dim=0).values
        assert min_q[0] >= 0 and min_q[1] >= 0, \
            f"Some points are outside the FOV (min): {pc[:, :3].min(dim=0).values}"
        assert max_q[0] < grid[0] and max_q[1] < grid[1], \
            f"Some points are outside the FOV (max): {pc[:, :3].max(dim=0).values}"

        # Quantized (row, col) → flat cell index
        temp = pc_quant[:, 0] * grid[1] + pc_quant[:, 1]
        cell_ind.append(temp.unsqueeze(0))

    return torch.cat(cell_ind, dim=0)   # (n_projections, N)

# ----------------------------------------------------------------------
# --- Helper functions
# ----------------------------------------------------------------------
def projection_3d_to_2d(feat, sp_mat, B, C, H, W):

    residual = torch.zeros(
        (B, C, H * W), 
        device=feat.device, 
        dtype=feat.dtype
    )
    residual.scatter_reduce_(
        2, 
        sp_mat["inflate"], 
        feat * sp_mat["mask_zero_padding"], 
        "sum", 
        include_self=False,
    )
    
    return residual

def get_all_projections(
    cell_ind, nb_feat, batch_size, num_points, occupied_cell, device, grids_shape, dtype,
):
    sp_mat = []
    B, C = batch_size, nb_feat
    occupied_cell = occupied_cell.unsqueeze(1).to(dtype)
    temp_mask = torch.ones_like(occupied_cell)
    for i in range(cell_ind.shape[1]):
        # Pre-compute number of points falling in each 2D cells
        temp_sp_mat = {"inflate": cell_ind[:, i:i+1], "mask_zero_padding": temp_mask}
        num_points_per_cell = projection_3d_to_2d(
            occupied_cell, temp_sp_mat, B, 1, *grids_shape[i],
        )
        num_points_per_cell = torch.gather(
            num_points_per_cell, 2, temp_sp_mat["inflate"],
        )
        # Store projection information
        sp_mat += [dict()]
        sp_mat[i]["mask_zero_padding"] = occupied_cell / (num_points_per_cell + 1e-6)
        sp_mat[i]["inflate"] = cell_ind[:, i:i+1].expand(-1, C, -1)
    
    return sp_mat

# ----------------------------------------------------------------------
# --- Layers
# ----------------------------------------------------------------------
class Embedding(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()

        #
        self.compressed = False
        self.channels_in, self.channels_out = channels_in, channels_out

        # Normalize inputs
        self.norm = nn.BatchNorm1d(channels_in)

        # Point Embedding
        self.conv1 = nn.Conv1d(channels_in, channels_out, 1)

        # Neighborhood embedding
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(channels_in),
            nn.Conv2d(channels_in, channels_out, 1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, channels_out, 1, bias=False),
        )

        # Merge point and neighborhood embeddings
        self.final = nn.Conv1d(2 * channels_out, channels_out, 1, bias=True, padding=0)

    def compress(self):
        # Recombine first batch norm and conv1
        first_norm_weight = self.norm.weight.data / torch.sqrt(
            self.norm.running_var.data + 1e-05
        )
        first_norm_bias = (
            self.norm.bias.data - first_norm_weight * self.norm.running_mean.data
        )
        conv1_weight = self.conv1.weight.data * first_norm_weight[None, :, None]
        conv1_bias = (
            self.conv1.weight.data[:, :, 0] @ first_norm_bias + self.conv1.bias.data
        )
        self.conv1.weight.data = conv1_weight
        self.conv1.bias.data = conv1_bias
        self.norm = nn.Identity()
        # Merge all batch norms and conv in local part
        # Trick in understanding the two line below is too realize that first_norm_bias has no influence because of
        # relative difference. Hence vector is just rescaled
        second_norm_weight = (
            first_norm_weight
            * self.conv2[0].weight.data
            / torch.sqrt(self.conv2[0].running_var.data + 1e-05)
        )
        second_norm_bias = (
            self.conv2[0].bias.data
            - (second_norm_weight / first_norm_weight) * self.conv2[0].running_mean
        )
        third_norm_weight = self.conv2[2].weight.data / torch.sqrt(
            self.conv2[2].running_var.data + 1e-05
        )
        third_norm_bias = (
            self.conv2[2].bias.data
            - third_norm_weight * self.conv2[2].running_mean.data
        )
        conv2_weight = (
            self.conv2[1].weight.data * second_norm_weight[None, :, None, None]
        )
        conv2_bias = self.conv2[1].weight.data[:, :, 0, 0] @ second_norm_bias
        conv2_weight = conv2_weight * third_norm_weight[:, None, None, None]
        conv2_bias = third_norm_weight * conv2_bias + third_norm_bias
        # Update layers
        self.new_conv2 = nn.Sequential(
            nn.Conv2d(self.channels_in, self.channels_out, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels_out, self.channels_out, 1, bias=False),
        )
        self.new_conv2[0].weight.data = conv2_weight
        self.new_conv2[0].bias.data = conv2_bias
        self.new_conv2[2].weight.data = self.conv2[4].weight.data
        self.conv2 = self.new_conv2
        # Flag
        self.compressed = True

    def forward(self, x, neighbors):
        """x: B x C_in x N. neighbors: B x K x N. Output: B x C_out x N"""
        if self.compressed:
            assert not self.training

        # Normalize input
        x = self.norm(x)

        # Point embedding
        point_emb = self.conv1(x)

        # Neighborhood embedding
        gather = []
        # Gather neighbors around each center point
        for ind_nn in range(
            1, neighbors.shape[1]
        ):  # Remove first neighbors which is the center point
            temp = neighbors[:, ind_nn : ind_nn + 1, :].expand(-1, x.shape[1], -1)
            gather.append(torch.gather(x, 2, temp).unsqueeze(-1))
        # Relative coordinates
        neigh_emb = torch.cat(gather, -1) - x.unsqueeze(-1)  # Size: (B x C x N) x K
        # Embedding
        neigh_emb = self.conv2(neigh_emb).max(-1)[0]

        # Merge both embeddings
        return self.final(torch.cat((point_emb, neigh_emb), dim=1))
    

class myLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x.transpose(1, -1)).transpose(1, -1)

NORM_OPTIONS = {
	"batchnorm": nn.BatchNorm1d,
	"layernorm": myLayerNorm,
}


class DropPath(nn.Module):
    """
    Stochastic Depth

    Original code of this module is at:
    https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
    """

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob

    def extra_repr(self):
        return f"prob={self.drop_prob}"

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        # work with diff dim tensors, not just 2D ConvNets
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = self.keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()  # binarize
        output = x.div(self.keep_prob) * random_tensor
        return output


class ChannelMix(nn.Module):
    def __init__(self, channels, drop_path_prob, which_norm="batchnorm"):
        super().__init__()
        self.compressed = False
        self.which_norm = which_norm
        self.norm = NORM_OPTIONS[which_norm](channels)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, 1),
        )
        self.scale = nn.Conv1d(
            channels, channels, 1, bias=False, groups=channels
        )  # Implement LayerScale
        self.drop_path = DropPath(drop_path_prob)

    def compress(self):
        if self.which_norm == "layernorm":
            warnings.warn(
                "Compression of ChannelMix layer has not been implemented " +
                "with layer norm. Skipping compression."
            )
            return
        # Join Batch norm and first conv
        norm_weight = self.norm.weight.data / torch.sqrt(
            self.norm.running_var.data + 1e-05
        )
        norm_bias = self.norm.bias.data - norm_weight * self.norm.running_mean.data
        # Careful the order of the two lines below should not be changed
        self.mlp[0].bias.data = (
            self.mlp[0].weight.data[:, :, 0] @ norm_bias + self.mlp[0].bias.data
        )
        self.mlp[0].weight.data = self.mlp[0].weight.data * norm_weight[None, :, None]
        # Join scale and last conv
        self.mlp[-1].weight.data = self.mlp[-1].weight.data * self.scale.weight.data
        self.mlp[-1].bias.data = (
            self.mlp[-1].bias.data * self.scale.weight.data[:, 0, 0]
        )
        # Flag
        self.compressed = True

    def forward(self, tokens):
        """tokens <- tokens + LayerScale( MLP( BN(tokens) ) )"""
        if self.compressed:
            assert not self.training
            return tokens + self.drop_path(self.mlp(tokens))
        else:
            return tokens + self.drop_path(self.scale(self.mlp(self.norm(tokens))))


class SpatialMix(nn.Module):
    def __init__(self, channels, grid_shape, drop_path_prob, which_norm="batchnorm"):
        super().__init__()
        self.compressed = False
        self.H, self.W = grid_shape
        self.norm = NORM_OPTIONS[which_norm](channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
        )
        self.scale = nn.Conv1d(
            channels, channels, 1, bias=False, groups=channels
        )  # Implement LayerScale
        self.grid_shape = grid_shape
        self.drop_path = DropPath(drop_path_prob)

    def extra_repr(self):
        return f"(grid): [{self.grid_shape[0]}, {self.grid_shape[1]}]"

    def compress(self):
        # Join scale and last conv
        self.ffn[-1].weight.data = (
            self.ffn[-1].weight.data * self.scale.weight.data[..., None]
        )
        self.ffn[-1].bias.data = (
            self.ffn[-1].bias.data * self.scale.weight.data[:, 0, 0]
        )
        # Flag
        self.compressed = True

    def forward_compressed(self, tokens, sp_mat):
        """tokens <- tokens + LayerScale( Inflate( FFN( Flatten( BN(tokens) ) ) )"""
        # Make sure we are not in training mode
        assert not self.training
        # Forward pass
        B, C, N = tokens.shape
        residual = self.norm(tokens)
        # Flatten
        residual = projection_3d_to_2d(residual, sp_mat, B, C, self.H, self.W)
        residual = residual.reshape(B, C, self.H, self.W)
        # FFN
        residual = self.ffn(residual)
        # Inflate
        residual = residual.reshape(B, C, self.H * self.W)
        residual = torch.gather(residual, 2, sp_mat["inflate"])
        return tokens + self.drop_path(residual)

    def forward(self, tokens, sp_mat):
        """tokens <- tokens + LayerScale( Inflate( FFN( Flatten( BN(tokens) ) ) )"""
        if self.compressed:
            return self.forward_compressed(tokens, sp_mat)
        #
        B, C, N = tokens.shape
        residual = self.norm(tokens)
        # Flatten
        residual = projection_3d_to_2d(residual, sp_mat, B, C, self.H, self.W)
        residual = residual.reshape(B, C, self.H, self.W)
        # FFN
        residual = self.ffn(residual)
        # LayerScale
        residual = residual.reshape(B, C, self.H * self.W)
        residual = self.scale(residual)
        # Inflate
        residual = torch.gather(residual, 2, sp_mat["inflate"])
        return tokens + self.drop_path(residual)


class WaffleIron(nn.Module):
    def __init__(self, channels, depth, grids_shape, drop_path_prob, which_norm="batchnorm"):
        super().__init__()
        self.depth = depth
        self.grids_shape = grids_shape
        self.channel_mix = nn.ModuleList(
            [ChannelMix(channels, drop_path_prob, which_norm) for _ in range(depth)]
        )
        self.spatial_mix = nn.ModuleList(
            [
                SpatialMix(channels, grids_shape[d % len(grids_shape)], drop_path_prob, which_norm)
                for d in range(depth)
            ]
        )

    def compress(self):
        for d in range(self.depth):
            self.channel_mix[d].compress()
            self.spatial_mix[d].compress()

    def forward(self, tokens, cell_ind, occupied_cell):
        # Build all 3D to 2D projection matrices
        batch_size, nb_feat, num_points = tokens.shape
        sp_mat = get_all_projections(
            cell_ind, nb_feat, batch_size, num_points, 
            occupied_cell, tokens.device, self.grids_shape, tokens.dtype,
        )

        # Actual backbone
        for d, (smix, cmix) in enumerate(zip(self.spatial_mix, self.channel_mix)):
            tokens = smix(tokens, sp_mat[d % len(sp_mat)])
            tokens = cmix(tokens)

        return tokens

# ----------------------------------------------------------------------
# --- Full Model
# ----------------------------------------------------------------------
class WafflesToGaussians(nn.Module):

    def __init__(
        self,
        in_channels: int = 51,
        waffle_channels: int = 384,
        num_layers: int = 48,
        num_neighbors: int = 16,
        voxel_size: float = 0.5,
        dims: tuple[int] = (0, 1, 2),
        dim_proj: tuple[int] = (2, 1, 0),
        grids_size: list[tuple[int, int]] = [(200, 200), (200, 16), (200, 16)],
        fov_xyz: tuple[tuple[float, float, float], tuple[float, float, float]] = ((-50, -50, -5), (50, 50, 3)),
        hidden_channels: int = 256,
        out_channels: int = 128,
        opacity_bias_init: float = 3.0,
        max_points: int = 3500,
    ) -> None:
        """
        Args:
            dim_proj (tuple[int]): Define the sequence of projection
                (which is then repeated sequentially until \ell = L)
        """
        super().__init__()
        self.voxel_size = voxel_size
        self.dims = dims
        self.dim_proj = dim_proj
        self.grids_size = grids_size
        self.fov_xyz = fov_xyz
        self.lut_axis_plane = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
        self.num_neighbors = num_neighbors
        self.max_points = max_points
        self.waffle_channels = waffle_channels
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.opacities_bias_init = opacity_bias_init

        self.embedding = Embedding(
            channels_in=in_channels,
            channels_out=self.waffle_channels,
        )
        self.backbone = WaffleIron(
            channels=self.waffle_channels,
            depth=self.num_layers,
            grids_shape=self.grids_size,
            drop_path_prob=0.1,
            which_norm="batchnorm",
        )

        self.feats_mlp = self._create_mlp(
            self.waffle_channels,
            self.out_channels,
            hidden_channels=self.hidden_channels,
            dropout=0.1,
        )
        self.offset_mlp = self._create_mlp(
            self.waffle_channels,
            3,
            hidden_channels=self.hidden_channels,
            dropout=0.1,
        )
        self.covs_mlp = self._create_mlp(
            self.waffle_channels,
            6,
            hidden_channels=self.hidden_channels,
            dropout=0.1,
        )
        self.opacities_mlp = self._create_mlp(
            self.waffle_channels,
            1,
            hidden_channels=self.hidden_channels,
            dropout=0.1,
            final_bias_init=opacity_bias_init,
        )


    def forward(self, radar_points: list[torch.Tensor]):

        # Prepare inputs for WaffleIron.
        # From radar_points to (feats, cell_ind, occupied_cell, neighbors).
        radar_dict = {
            "xyz": [],
            "feats": [],
            "cell_ind": [],
            "occupied_cell": [],
            "neighbors": [],
        }
        for r in radar_points:
            vox_points = voxelize(r, dims=self.dims, voxel_size=self.voxel_size)
            crop_points = crop(vox_points, dims=self.dims, fov=self.fov_xyz)
            cell_ind = get_occupied_2d_cells(                           # Shape (N, P)
                crop_points,
                dim_proj=self.dim_proj,
                grids_shape=self.grids_size,
                lut_axis_plane=self.lut_axis_plane,
                fov_xyz=torch.tensor(self.fov_xyz, device=crop_points.device),
            ).T
            edge_index = knn(crop_points[:, :3], crop_points[:, :3], k=self.num_neighbors + 1)
            neighbors_emb = edge_index[0].view(crop_points.shape[0], self.num_neighbors + 1)

            radar_dict["xyz"].append(crop_points[:, :3])  # Shape (N, 3)
            radar_dict["feats"].append(crop_points)  # Shape (N, C)
            radar_dict["cell_ind"].append(cell_ind)
            radar_dict["occupied_cell"].append(
                torch.ones(crop_points.shape[0], device=crop_points.device, dtype=torch.long)
            )
            radar_dict["neighbors"].append(neighbors_emb)

        # Do zero padding to max_points and convert lists to tensors.
        for key in radar_dict:
            radar_dict[key] = [
                torch.cat(
                    (
                        x,
                        torch.zeros(
                            self.max_points - x.shape[0],
                            *x.shape[1:],
                            device=x.device,
                            dtype=x.dtype,
                        ),
                    ),
                    dim=0,
                )
                if x.shape[0] < self.max_points else x[:self.max_points]
                for x in radar_dict[key]
            ]
            radar_dict[key] = torch.stack(radar_dict[key], dim=0)

        radar_dict["xyz"] = radar_dict["xyz"].permute(0, 2, 1)
        radar_dict["feats"] = radar_dict["feats"].permute(0, 2, 1)
        radar_dict["cell_ind"] = radar_dict["cell_ind"].permute(0, 2, 1)
        radar_dict["neighbors"] = radar_dict["neighbors"].permute(0, 2, 1)

        # Pass the prepared inputs to WaffleIron.
        tokens = self.embedding(radar_dict["feats"], radar_dict["neighbors"])
        tokens = self.backbone(tokens, radar_dict["cell_ind"], radar_dict["occupied_cell"])

        # Convert from Waffles to Gaussians.
        tokens = tokens.permute(0, 2, 1)
        offsets = self.offset_mlp(tokens)
        means = radar_dict["xyz"].permute(0, 2, 1) + offsets
        covs = self.covs_mlp(tokens)
        covs = self.make_valid_covariances(covs)
        opacities = torch.sigmoid(self.opacities_mlp(tokens))
        features = self.feats_mlp(tokens)

        return {
            "centers": means,
            "offsets": offsets,
            "covariances": covs,
            "opacities": opacities,
            "features": features,
        }
    
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
    
    def make_valid_covariances(self, covs_raw: torch.Tensor) -> torch.Tensor:
        """
        Convert raw MLP output to valid covariance matrices.

        Args:
            covs_raw: Raw covariance parameters, shape [P, 6] or [B, P, 6].

        Returns:
            Valid covariance parameters, same shape as input.
        """
        # Handle both [P, 6] and [B, P, 6] inputs transparently
        unbatched = covs_raw.dim() == 2
        if unbatched:
            covs_raw = covs_raw.unsqueeze(0)  # [1, P, 6]

        # Diagonal elements — guaranteed positive via softplus
        xx = F.softplus(covs_raw[..., 0]) + 1e-4
        yy = F.softplus(covs_raw[..., 3]) + 1e-4
        zz = F.softplus(covs_raw[..., 5]) + 1e-4

        # Off-diagonal elements — bounded by 0.9 * sqrt(diag_i * diag_j)
        # to ensure positive semi-definiteness
        xy = torch.tanh(covs_raw[..., 1]) * torch.sqrt(xx * yy) * 0.9
        xz = torch.tanh(covs_raw[..., 2]) * torch.sqrt(xx * zz) * 0.9
        yz = torch.tanh(covs_raw[..., 4]) * torch.sqrt(yy * zz) * 0.9

        covs = torch.stack([xx, xy, xz, yy, yz, zz], dim=-1)  # [..., 6]

        return covs.squeeze(0) if unbatched else covs
    

def build_dummy_inputs(device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
    return [
        torch.randn(2000, 51, device=device, dtype=dtype),
        torch.randn(3000, 51, device=device, dtype=dtype),
        torch.randn(3000, 51, device=device, dtype=dtype),
        torch.randn(3000, 51, device=device, dtype=dtype),
    ]


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def warmup(model: torch.nn.Module, inputs: list[torch.Tensor], n: int = 5) -> None:
    with torch.no_grad():
        for _ in range(n):
            model(inputs)


def time_inference(
    model: torch.nn.Module,
    inputs: list[torch.Tensor],
    n_runs: int = 100,
) -> dict[str, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    latencies_ms = []

    with torch.no_grad():
        for _ in range(n_runs):
            start.record()
            model(inputs)
            end.record()
            torch.cuda.synchronize()
            latencies_ms.append(start.elapsed_time(end))

    return {
        "mean_ms":   sum(latencies_ms) / n_runs,
        "min_ms":    min(latencies_ms),
        "max_ms":    max(latencies_ms),
        "std_ms":    torch.tensor(latencies_ms).std().item(),
        "fps":       1000.0 / (sum(latencies_ms) / n_runs),
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = WafflesToGaussians(num_layers=12).to(device).to(dtype).eval()
    params = count_parameters(model)    
    inputs = build_dummy_inputs(device, dtype)

    print("Warming up...")
    warmup(model, inputs)

    print("Timing inference...")
    stats = time_inference(model, inputs, n_runs=100)

    print(f"\n{'─' * 30}")
    print(f"  Mean latency : {stats['mean_ms']:.2f} ms")
    print(f"  Std          : {stats['std_ms']:.2f} ms")
    print(f"  Min / Max    : {stats['min_ms']:.2f} / {stats['max_ms']:.2f} ms")
    print(f"  Throughput   : {stats['fps']:.1f} FPS")
    print(f"  Total params : {params['total']:,}")
    print(f"{'─' * 30}\n")