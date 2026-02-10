import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale


def extract_pca_features(
    features: torch.Tensor,
    n_components: int = 3,
) -> torch.Tensor:
    # Ensure the input is on CPU and converted to numpy
    B, C, H, W = features.shape
    
    # Reshape features to (Batch, Channels, H*W)
    projected_heatmap = features.view(B, C, -1).cpu().numpy()
    
    # Initialize output list
    output = []
    
    # Process each batch item
    for heatmap in projected_heatmap:
        # Transpose to make it (H*W, Channels) for PCA
        heatmap_transposed = heatmap.T
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(heatmap_transposed)
        
        # Normalize to 0-255 range
        pca_features = minmax_scale(pca_features, feature_range=(0, 255)).astype(np.uint8)
        
        # Reshape back to (n_components, H, W)
        pca_features = pca_features.T.reshape(n_components, H, W)
        
        # Convert to torch tensor
        pca_features = torch.from_numpy(pca_features)
        output.append(pca_features)
    
    # Stack the batch
    pca_features = torch.stack(output, dim=0)
    
    return pca_features


def extract_pca_features_batched(
    features: torch.Tensor,
    n_components: int = 3,
) -> torch.Tensor:
    """Extract PCA features using shared principal components across batch.

    Args:
        features: Input tensor of shape (B, C, H, W).
        n_components: Number of PCA components.

    Returns:
        Tensor of shape (B, n_components, H, W) with uint8 values.
    """
    # Get shapes
    bsz, ch, hgt, wid = features.shape

    # Reshape to (B * H * W, C)
    flat = (
        features
        .permute(0, 2, 3, 1)
        .reshape(-1, ch)
        .cpu()
        .numpy()
    )

    # Fit PCA once on all samples
    pca = PCA(n_components=n_components)
    flat_pca = pca.fit_transform(flat)

    # Normalize globally to keep colors consistent across batch
    flat_pca = minmax_scale(
        flat_pca,
        feature_range=(0, 255),
    ).astype(np.uint8)

    # Reshape back to (B, n_components, H, W)
    pca_features = (
        flat_pca
        .reshape(bsz, hgt, wid, n_components)
        .transpose(0, 3, 1, 2)
    )

    return torch.from_numpy(pca_features)