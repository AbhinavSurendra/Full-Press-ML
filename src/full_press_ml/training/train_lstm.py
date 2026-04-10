"""Train the PyTorch LSTM possession classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from full_press_ml.data.tracking_dataset import PossessionSequenceDataset
from full_press_ml.models.lstm_model import PossessionLSTM


def train_epoch(
    model: PossessionLSTM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / max(len(dataloader), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--label-column", default="label_id")
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    frame_table = pd.read_csv(args.data)
    feature_columns = [
        col
        for col in frame_table.columns
        if col not in {"game_id", "possession_id", "frame_idx", "possession_frame_idx", args.label_column}
    ]
    dataset = PossessionSequenceDataset(
        frame_table=frame_table,
        feature_columns=feature_columns,
        label_column=args.label_column,
        max_len=args.max_len,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PossessionLSTM(
        input_size=len(feature_columns),
        hidden_size=128,
        num_layers=2,
        num_classes=5,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        loss = train_epoch(model, dataloader, optimizer, criterion, device)
        print(f"epoch={epoch + 1} loss={loss:.4f}")


if __name__ == "__main__":
    main()
