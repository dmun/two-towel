import torch
from torch._prims_common import DeviceLikeType
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm.notebook import tqdm
from util import evaluate_model, GameDataset
from model import TwoTowerModel


class WeightedLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, predictions, targets, weights):
        loss = self.bce_loss(predictions, targets)
        weighted_loss = loss * weights
        return torch.mean(weighted_loss)


def accuracy(out, y):
    return Accuracy(task="binary")(out, y)


def train(
    model: TwoTowerModel,
    datasets: dict[str, GameDataset],
    num_epochs=10,
    batch_size=256,
    lr=0.01,
    optimizer=torch.optim.Adam,
    loss_fn: nn.Module = torch.nn.BCELoss(),
    device: DeviceLikeType = "cpu",
):
    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size, pin_memory=True)

    model = model.to(device)
    optimizer = optimizer(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for user_ids, game_ids, y in tqdm(train_loader, leave=False):
            user_ids = user_ids.to(device)
            game_ids = game_ids.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            out = model(user_ids, game_ids)
            loss = loss_fn(out, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        with torch.no_grad():
            out, y = evaluate_model(model, test_loader, device=device)
            test_loss = loss_fn(out, y)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader):.2f}, Val Loss: {test_loss:.2f}")
