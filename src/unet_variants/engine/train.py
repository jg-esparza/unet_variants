import torch
from torch.utils.data import DataLoader

def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    train_loader: torch.utils.data.DataLoader,
                    device:torch.device
                    ):
    """
    Trains the model for one epoch over the dataset.
    """
    model.train()
    train_loss = 0
    for batch in train_loader:
        images, masks = batch["image"].to(device), batch["mask"].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    return {"train/loss": train_loss}
