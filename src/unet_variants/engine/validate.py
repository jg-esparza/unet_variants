import torch
from torch.utils.data import DataLoader

def validate_one_epoch(model: torch.nn.Module,
                       criterion: torch.nn.Module,
                       val_loader: torch.utils.data.DataLoader,
                       device:torch.device
                       ):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images, masks = batch["image"].to(device), batch["mask"].to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    return val_loss