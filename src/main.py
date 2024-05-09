import torch
import torch.nn as nn

from tqdm import tqdm
from dataclasses import dataclass

from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, AdamW

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

from kan import KAN_MNIST

@dataclass
class TrainConfig:
    epochs: int
    learning_rate: float
    opt: str
    device: str

def get_optimizer(opt: str):
    match opt:
        case "adam":
            return Adam
        case "adamw":
            return AdamW
        case _:
            return Adam

def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: TrainConfig,
):
    epochs = config.epochs
    learning_rate = config.learning_rate
    device = config.device
    opt = get_optimizer(config.opt)(model.parameters(), lr=learning_rate)

    avg_train_loss = 0
    for e in range(epochs):
        train_acc = 0
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)

            opt.zero_grad()

            y_pred = model(x)
            loss = nn.CrossEntropyLoss()(y_pred, y)
            
            loss.backward()
            opt.step()

            train_acc += (y_pred.argmax(1) == y).float().mean().item()
            avg_train_loss += loss.item()

        avg_train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        with torch.no_grad():
            acc = 0
            avg_loss = 0
            for x, y in tqdm(test_loader):
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = nn.CrossEntropyLoss()(y_pred, y)
                acc += (y_pred.argmax(1) == y).float().mean().item()
                avg_loss += loss.item()

            acc /= len(test_loader)
            avg_loss /= len(test_loader)

            print(f"""
                Epoch {e+1}/{epochs}
                Train Loss: {avg_train_loss}
                Val Loss: {avg_loss}
                Train Acc: {train_acc}
                Val Acc: {acc}
            """)

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.backends.openmp.is_available():
        device = "opencl"
    else:
        device = "cpu"

    torch.device = device

    split_ratio = 0.8
    batch_size = 64

    transform = Compose([ToTensor()])
    dataset = MNIST("../data", transform=transform, download=True)
    train_data, test_data = random_split(
        dataset,
        [
            int(len(dataset) * split_ratio),
            len(dataset) - int(len(dataset) * split_ratio),
        ]
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    model = KAN_MNIST().to(device)

    train_config = TrainConfig(
        epochs=5,
        learning_rate=0.002,
        opt="adam",
        device=device,
    )
    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=train_config,
    )

