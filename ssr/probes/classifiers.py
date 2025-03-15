from typing import Dict, List, Tuple

import torch as t
from jaxtyping import Float
from torch.utils.data import DataLoader, TensorDataset

from ssr.types import Loss, Optimizer


def activations_to_dataloader(
    hf_layer_act: Float[t.Tensor, "batch_size d_model"],
    hl_layer_act: Float[t.Tensor, "batch_size d_model"],
    batch_size: int = 32,
    train_size: float = 0.7,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    a, b = hf_layer_act.float(), hl_layer_act.float()
    x = t.cat([a, b], dim=0)
    y = t.cat([t.zeros(a.shape[0]), t.ones(b.shape[0])], dim=0)

    dataset = TensorDataset(x, y)
    train_dataset, test_dataset = t.utils.data.random_split(
        dataset, [train_size, 1 - train_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, full_loader


class LinearClassifier(t.nn.Module):
    def __init__(self, d_model: int, xavier_gain: float = 0.1):
        super().__init__()
        self.linear = t.nn.Linear(d_model, 1)
        self.sigmoid = t.nn.Sigmoid()
        t.nn.init.xavier_uniform_(self.linear.weight, gain=xavier_gain)
        t.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


def train_model(
    model: t.nn.Module,
    train_loader: DataLoader,
    criterion: Loss,
    optimizer: Optimizer,
    epochs: int = 10,
    verbose: bool = False,
):
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if verbose:
            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}"
            )

    return model


def test_model(model: t.nn.Module, test_loader: DataLoader, criterion):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with t.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            preds = (outputs >= 0.5).float()

            all_preds.extend(preds.numpy())
            all_labels.extend(batch_y.numpy())

    metrics = {
        "loss": total_loss / len(test_loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "f1_score": f1_score(all_labels, all_preds),
    }

    return metrics


def train_and_test_classifier(
    train_loader: DataLoader,
    test_loader: DataLoader,
    d_model: int,
    loss_name: str,
    optimizer_name: str,
    lr: float,
    epochs: int,
    verbose: bool = False,
) -> Tuple[t.nn.Module, Loss, Dict[str, float]]:
    model = LinearClassifier(d_model=d_model)

    match optimizer_name:
        case "SGD":
            optimizer: Optimizer = t.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        case _:
            optimizer = t.optim.Adam(model.parameters(), lr=lr)

    match loss_name:
        case "BCE":
            loss_fn: Loss = t.nn.BCELoss()
        case _:
            loss_fn = t.nn.MSELoss()

    trained_model = train_model(model, train_loader, loss_fn, optimizer, epochs=epochs)

    metrics = test_model(trained_model, test_loader, loss_fn)
    if verbose:
        print(metrics)

    return trained_model, loss_fn, metrics


def compare_loss_and_optimizers(
    train_loader: DataLoader, test_loader: DataLoader, d_model: int
) -> Tuple[List[dict], int]:
    import numpy as np

    loss_functions = ["BCE", "MSE"]
    optimizers = ["SGD", "Adam"]
    lrs = [0.01, 0.001, 0.0001]
    test_epochs = [10, 20, 30, 40, 50, 70, 100, 150, 200]

    results = []
    accuracies = []

    for loss_name in loss_functions:
        for opt_name in optimizers:
            for lr in lrs:
                for epochs in test_epochs:
                    _, _, metrics = train_and_test_classifier(
                        train_loader,
                        test_loader,
                        d_model=d_model,
                        loss_name=loss_name,
                        optimizer_name=opt_name,
                        lr=lr,
                        epochs=epochs,
                        verbose=False,
                    )

                    accuracies.append(metrics["accuracy"])

                    results.append(
                        {
                            "loss_name": loss_name,
                            "optimizer": opt_name,
                            "lr": lr,
                            "epochs": epochs,
                        }
                        | metrics
                    )

    return results, int(np.argmax(accuracies))


def print_results(
    results: List[Dict[str, float | int | str]],
    best_accuracy_idx: int = -1,
    title: str = "Results",
):
    import rich

    metric_names = ["loss", "accuracy", "precision", "recall", "f1_score"]
    config_names = ["loss_name", "optimizer", "lr", "epochs"]

    table = rich.table.Table(title=title)  # type: ignore
    for name in config_names + metric_names:
        table.add_column(name)

    for i, result in enumerate(results):
        table.add_row(
            *[f"{x}" for x in [result[key] for key in config_names]],
            *[f"{x:.5f}" for x in [result[key] for key in metric_names]],
            style="dark_orange" if i == best_accuracy_idx else "black",
        )
    console = rich.console.Console()
    console.print(table)
