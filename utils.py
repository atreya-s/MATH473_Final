import torch
import torchvision
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader

from dataset import BCCDDataset  # Updated dataset class name

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """
    Save model checkpoint
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer=None):
    """
    Load model checkpoint
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    """
    Create train and validation data loaders
    """
    train_ds = BCCDDataset(
        images_dir=train_dir,
        masks_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = BCCDDataset(
        images_dir=val_dir,
        masks_dir=val_maskdir,  # Use masks_dir for validation
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # Ensure y has the right shape
            y = y.float().unsqueeze(0).to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    """
    Save model predictions as images
    """
    model.eval()
    os.makedirs(folder, exist_ok=True)
    
    for idx, (x, _) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds, _ = model(x)  # Updated to match new model signature
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()
        
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
    
    model.train()

def get_train_loader(
    train_dir,
    train_maskdir,
    batch_size,
    train_transform,
    num_workers=4,
    pin_memory=True,
):
    """
    Create training data loader
    """
    train_ds = BCCDDataset(
        images_dir=train_dir,
        masks_dir=train_maskdir,
        transform=train_transform,
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    
    return train_loader

def get_val_loader(
    val_dir,
    val_maskdir,
    batch_size,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    """
    Create validation data loader
    """
    val_ds = BCCDDataset(
        images_dir=val_dir,
        masks_dir=val_maskdir,  # Use masks_dir for validation
        transform=val_transform,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    
    return val_loader
