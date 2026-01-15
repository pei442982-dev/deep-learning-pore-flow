import argparse
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from data import NPZTwoPhaseDataset
from model import UNetTwoPhase


@dataclass
class TrainConfig:
    data_dir: str
    batch_size: int = 4
    epochs: int = 20
    lr: float = 1e-3
    weight_vel: float = 1.0
    weight_sat: float = 2.0
    val_split: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a two-phase flow predictor.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-vel", type=float, default=1.0)
    parser.add_argument("--weight-sat", type=float, default=2.0)
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()
    return TrainConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_vel=args.weight_vel,
        weight_sat=args.weight_sat,
        val_split=args.val_split,
    )


def weighted_loss(pred: torch.Tensor, target: torch.Tensor, weight_vel: float, weight_sat: float) -> torch.Tensor:
    vel_pred, sat_pred = pred[:, :2], pred[:, 2:3]
    vel_target, sat_target = target[:, :2], target[:, 2:3]
    mse = nn.MSELoss()
    return weight_vel * mse(vel_pred, vel_target) + weight_sat * mse(sat_pred, sat_target)


def split_dataset(dataset: NPZTwoPhaseDataset, val_split: float) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size])


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, config: TrainConfig) -> float:
    model.train()
    running = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(config.device)
        targets = targets.to(config.device)
        optimizer.zero_grad()
        preds = model(inputs)
        loss = weighted_loss(preds, targets, config.weight_vel, config.weight_sat)
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / max(1, len(loader))


def eval_epoch(model: nn.Module, loader: DataLoader, config: TrainConfig) -> float:
    model.eval()
    running = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)
            preds = model(inputs)
            loss = weighted_loss(preds, targets, config.weight_vel, config.weight_sat)
            running += loss.item()
    return running / max(1, len(loader))


def main() -> None:
    config = parse_args()
    dataset = NPZTwoPhaseDataset(config.data_dir)
    train_set, val_set = split_dataset(dataset, config.val_split)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size)

    sample_inputs, _ = train_set[0]
    in_channels = sample_inputs.shape[0]

    model = UNetTwoPhase(in_channels=in_channels).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, config)
        val_loss = eval_epoch(model, val_loader, config)
        print(f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}")

    torch.save(model.state_dict(), "two_phase_unet.pt")


if __name__ == "__main__":
    main()
