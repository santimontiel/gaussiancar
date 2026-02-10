from functools import partial

import logging
import hydra
import lightning as L
import rootutils
import torch
from nuscenes.nuscenes import NuScenes
from tqdm.auto import tqdm
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = logging.getLogger(__name__)

from gaussiancar.utils.config import register_new_resolvers
from gaussiancar.utils.consts import VALIDATION_DRN_SPLITS


def create_square_mask(grid_size, cell_resolution, start_meters, end_meters):
    """
    Create a square mask for a grid where elements between start_meters and end_meters
    from the center are marked as True.
    
    Args:
        grid_size: Size of the square grid (e.g., 200 for a 200x200 grid)
        cell_resolution: Resolution of each cell in meters (e.g., 0.5m)
        start_meters: Inner boundary of the mask in meters
        end_meters: Outer boundary of the mask in meters
        
    Returns:
        torch.Tensor: Boolean mask where True indicates positions within the specified range
    """
    # Create a tensor filled with zeros
    mask = torch.zeros((grid_size, grid_size), dtype=torch.bool)
    
    # Calculate the center of the grid
    center = grid_size // 2
    
    # Calculate how many cells correspond to the start and end distances
    start_cells = int(start_meters / cell_resolution)
    end_cells = int(end_meters / cell_resolution)
    
    # Calculate the square boundaries for the outer square
    outer_start = center - end_cells
    outer_end = center + end_cells
    
    # Calculate the square boundaries for the inner square
    inner_start = center - start_cells
    inner_end = center + start_cells
    
    # Set the outer square to True
    mask[outer_start:outer_end, outer_start:outer_end] = True
    
    # If there's an inner boundary, set it to False (creating a ring)
    if start_meters > 0:
        mask[inner_start:inner_end, inner_start:inner_end] = False
    
    return mask


def get_scene_name(nusc: NuScenes, sample_token: str) -> str:
    """Return scene name given a sample token.
    
    Args:
        nusc (NuScenes): Loaded NuScenes object.
        sample_token (str): Token of the sample.
    
    Returns:
        str: Scene name.
    """
    # Get sample and its scene token
    sample = nusc.get('sample', sample_token)
    scene_token = sample['scene_token']
    
    # Get scene metadata
    scene = nusc.get('scene', scene_token)
    return scene['name']


def load_from_checkpoint(
    module: L.LightningModule,
    checkpoint_path: str,
    device: str = "cpu",
) -> L.LightningModule:
    """Load model weights from a checkpoint file.

    Args:
        module (L.LightningModule): The LightningModule instance to load weights into.
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        L.LightningModule: The LightningModule instance with loaded weights.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    module.load_state_dict(checkpoint["state_dict"])
    module.to(device)
    return module


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main function to make a prediction in a selected sample using a trained model.
    """
    register_new_resolvers()

    log.info(f"Loading datamodule <{cfg.data._target_}>...")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")

    log.info(f"Loading module <{cfg.module._target_}> with weights from <{cfg.checkpoint_path}>...")
    module: L.LightningModule = hydra.utils.instantiate(cfg.module, cfg=cfg)
    if cfg.checkpoint_path:
        module = load_from_checkpoint(module, cfg.checkpoint_path, cfg.device)
        module.eval()

    log.info(f"Loading metrics for {cfg.task.key} task...")
    # General metrics.
    metrics: torch.nn.ModuleDict = torch.nn.ModuleDict(hydra.utils.instantiate(cfg.task.metrics))
    metrics.to(cfg.device)

    # Range-specific metrics.
    metrics_0_20: torch.nn.ModuleDict = torch.nn.ModuleDict(hydra.utils.instantiate(cfg.task.metrics))
    metrics_0_20.to(cfg.device)
    mask = partial(create_square_mask, grid_size=200, cell_resolution=0.5)
    mask_0_20 = mask(start_meters=0, end_meters=20).to(cfg.device).unsqueeze(0).unsqueeze(0)
    metrics_20_35: torch.nn.ModuleDict = torch.nn.ModuleDict(hydra.utils.instantiate(cfg.task.metrics))
    metrics_20_35.to(cfg.device)
    mask_20_35 = mask(start_meters=20, end_meters=35).to(cfg.device).unsqueeze(0).unsqueeze(0)
    metrics_35_50: torch.nn.ModuleDict = torch.nn.ModuleDict(hydra.utils.instantiate(cfg.task.metrics))
    metrics_35_50.to(cfg.device)
    mask_35_50 = mask(start_meters=35, end_meters=50).to(cfg.device).unsqueeze(0).unsqueeze(0)

    # Weather-specific metrics.
    nusc: NuScenes = NuScenes(
        version=cfg.data.data_config.version,
        dataroot=cfg.data.data_config.dataset_dir,
        verbose=False,
    )
    metrics_day: torch.nn.ModuleDict = torch.nn.ModuleDict(hydra.utils.instantiate(cfg.task.metrics))
    metrics_day.to(cfg.device)
    metrics_night: torch.nn.ModuleDict = torch.nn.ModuleDict(hydra.utils.instantiate(cfg.task.metrics))
    metrics_night.to(cfg.device)
    metrics_rain: torch.nn.ModuleDict = torch.nn.ModuleDict(hydra.utils.instantiate(cfg.task.metrics))
    metrics_rain.to(cfg.device)

    log.info("Evaluating on the validation dataset...")
    pbar = tqdm(datamodule.val_dataloader(), desc="Performing evaluation...", colour="blue")
    pbar.set_postfix_str(f"Task: {cfg.task.key}, mIoU: --")

    for sample in pbar:
        # Move sample data to device.
        for k, v in sample.items():
            if torch.is_tensor(v):
                sample[k] = v.to(cfg.device)
            elif isinstance(v, list) and torch.is_tensor(v[0]):
                sample[k] = [item.to(cfg.device) for item in v]
            else:
                continue

        # Forward pass through the model.
        with torch.no_grad():
            outputs = module(sample)

        # Calculate metrics.
        ious = []
        ious_0_20 = []
        ious_20_35 = []
        ious_35_50 = []
        ious_day = []
        ious_night = []
        ious_rain = []

        for k, v in metrics.items():
            # General metrics.
            v.update(outputs["output"], sample)
            res = v.compute()
            ious.append(res[k].item())

            # Range-specific metrics.
            metrics_0_20[k].update(outputs["output"], sample, ignore_mask=~mask_0_20)
            res = metrics_0_20[k].compute()
            ious_0_20.append(res[k].item())
            metrics_20_35[k].update(outputs["output"], sample, ignore_mask=~mask_20_35)
            res = metrics_20_35[k].compute()
            ious_20_35.append(res[k].item())
            metrics_35_50[k].update(outputs["output"], sample, ignore_mask=~mask_35_50)
            res = metrics_35_50[k].compute()
            ious_35_50.append(res[k].item())

            # Weather-specific metrics.
            token = sample["token"][0]
            scene_name = get_scene_name(nusc, token)
            if scene_name in VALIDATION_DRN_SPLITS["day"]:
                metrics_day[k].update(outputs["output"], sample)
            elif scene_name in VALIDATION_DRN_SPLITS["night"]:
                metrics_night[k].update(outputs["output"], sample)
            elif scene_name in VALIDATION_DRN_SPLITS["rain"]:
                metrics_rain[k].update(outputs["output"], sample)
                
        miou = torch.tensor(ious).mean().item()
        miou_0_20 = torch.tensor(ious_0_20).mean().item()
        miou_20_35 = torch.tensor(ious_20_35).mean().item()
        miou_35_50 = torch.tensor(ious_35_50).mean().item()

        pbar.set_postfix_str(f"Task: {cfg.task.key}, mIoU: {miou:.4f}")

    keys = list(metrics.keys())
    ious_day, ious_night, ious_rain = [], [], []
    for k in keys:
        ious_day.append(metrics_day[k].compute()[k].item())
        ious_night.append(metrics_night[k].compute()[k].item())
        ious_rain.append(metrics_rain[k].compute()[k].item())
    miou_day = torch.tensor(ious_day).mean().item() if ious_day else "--"
    miou_night = torch.tensor(ious_night).mean().item() if ious_night else "--"
    miou_rain = torch.tensor(ious_rain).mean().item() if ious_rain else "--"

    # Print metrics in a rich table.
    from rich.table import Table
    from rich.console import Console

    metrics_table = Table(title="Evaluation Metrics", show_lines=True)
    metrics_table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
    metrics_table.add_column("Value", justify="right", style="magenta")

    for k, v in metrics.items():
        metrics_table.add_row(k, f"{v.compute()[k].item():.4f}")
    metrics_table.add_row("mIoU", f"{miou:.4f}", style="bold green")

    range_table = Table(title="Evaluation Metrics by Distance", show_lines=True)
    range_table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
    range_table.add_column("0-20m", justify="right", style="magenta")
    range_table.add_column("20-35m", justify="right", style="magenta")
    range_table.add_column("35-50m", justify="right", style="magenta")

    for k, v in metrics_0_20.items():
        range_table.add_row(k, f"{v.compute()[k].item():.4f}", f"{metrics_20_35[k].compute()[k].item():.4f}", f"{metrics_35_50[k].compute()[k].item():.4f}")
    range_table.add_row("mIoU", f"{miou_0_20:.4f}", f"{miou_20_35:.4f}", f"{miou_35_50:.4f}", style="bold green")

    weather_table = Table(title="Evaluation Metrics by Weather", show_lines=True)
    weather_table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
    weather_table.add_column("Day", justify="right", style="magenta")
    weather_table.add_column("Night", justify="right", style="magenta")
    weather_table.add_column("Rain", justify="right", style="magenta")

    for k, v in metrics_day.items():
        day_value = f"{v.compute()[k].item():.4f}" if ious_day else "--"
        night_value = f"{metrics_night[k].compute()[k].item():.4f}" if ious_night else "--"
        rain_value = f"{metrics_rain[k].compute()[k].item():.4f}" if ious_rain else "--"
        weather_table.add_row(k, day_value, night_value, rain_value)

    miou_day_value = f"{miou_day:.4f}" if miou_day != "--" else "--"
    miou_night_value = f"{miou_night:.4f}" if miou_night != "--" else "--"
    miou_rain_value = f"{miou_rain:.4f}" if miou_rain != "--" else "--"
    weather_table.add_row("mIoU", f"{miou_day_value}", f"{miou_night_value}", f"{miou_rain_value}", style="bold green")

    console = Console()
    console.rule("[bold red]Evaluation Results")
    console.print(metrics_table)
    console.print(range_table)
    console.print(weather_table)
    console.rule("[bold red]...")

    log.info("Evaluation completed. Final mIoU: {:.4f} ✅".format(miou))


if __name__ == "__main__":
    main()