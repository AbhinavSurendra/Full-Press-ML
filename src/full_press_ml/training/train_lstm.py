"""Train the PyTorch LSTM possession classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
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


@torch.no_grad()
def evaluate(
    model: PossessionLSTM,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    labels = []
    for features, batch_labels in dataloader:
        features = features.to(device)
        logits = model(features)
        predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(predictions), np.concatenate(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--label-column", default="label_id")
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-split", choices=["train", "val", "test"], default="test")
    args = parser.parse_args()

    frame_table = pd.read_csv(args.data)
    if "possession_is_usable" in frame_table.columns:
        frame_table = frame_table[frame_table["possession_is_usable"] == 1].copy()
    frame_table = frame_table[frame_table["possession_id"].notna()].copy()

    label_encoder: LabelEncoder | None = None
    if args.label_column == "label_id" and "label_id" not in frame_table.columns:
        if "terminal_label" not in frame_table.columns:
            raise ValueError("Expected either label_id or terminal_label in the frame table.")
        label_encoder = LabelEncoder()
        frame_table["label_id"] = label_encoder.fit_transform(frame_table["terminal_label"])
        args.label_column = "label_id"
    elif args.label_column == "terminal_label":
        label_encoder = LabelEncoder()
        frame_table["label_id"] = label_encoder.fit_transform(frame_table["terminal_label"])
        args.label_column = "label_id"

    train_table = frame_table[frame_table["split"] == "train"].copy()
    eval_table = frame_table[frame_table["split"] == args.eval_split].copy()
    if train_table.empty or eval_table.empty:
        raise ValueError("Training or evaluation split is empty after filtering.")

    feature_columns = [
        col
        for col in train_table.columns
        if col
        not in {
            "game_id",
            "event_id",
            "possession_id",
            "frame_idx",
            "possession_frame_idx",
            "split",
            "terminal_label",
            "pbp_join_status",
            args.label_column,
        }
    ]
    feature_columns = [col for col in feature_columns if pd.api.types.is_numeric_dtype(train_table[col])]
    train_table[feature_columns] = train_table[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    eval_table[feature_columns] = eval_table[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    train_dataset = PossessionSequenceDataset(
        frame_table=train_table,
        feature_columns=feature_columns,
        label_column=args.label_column,
        max_len=args.max_len,
    )
    eval_dataset = PossessionSequenceDataset(
        frame_table=eval_table,
        feature_columns=feature_columns,
        label_column=args.label_column,
        max_len=args.max_len,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PossessionLSTM(
        input_size=len(feature_columns),
        hidden_size=128,
        num_layers=2,
        num_classes=int(train_table[args.label_column].nunique()),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"epoch={epoch + 1} loss={loss:.4f}")

    predictions, labels = evaluate(model, eval_loader, device)
    print(f"train_sequences={len(train_dataset)} eval_sequences={len(eval_dataset)} eval_split={args.eval_split}")
    print(f"accuracy={accuracy_score(labels, predictions):.4f}")
    target_names = label_encoder.classes_ if label_encoder is not None else None
    print(classification_report(labels, predictions, target_names=target_names, zero_division=0))


if __name__ == "__main__":
    main()
