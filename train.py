import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import RegularizedUNet  # Updated model import
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_train_loader,
    get_val_loader,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 50
NUM_WORKERS = 2
IMAGE_HEIGHT = 512  
IMAGE_WIDTH = 512 
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = '/home/axs2220/Math473_Final/Dataset/BCCD/train/original'
TRAIN_MASK_DIR = '/home/axs2220/Math473_Final/Dataset/BCCD/train/mask'
VAL_IMG_DIR = '/home/axs2220/Math473_Final/Dataset/BCCD/test/original'
VAL_MASK_DIR = '/home/axs2220/Math473_Final/Dataset/BCCD/test/mask'

def train_fn(loader, model, optimizer, scaler, device):
    loop = tqdm(loader)
    total_loss = 0.0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.to(device)

        # Forward pass with AMP
        with torch.cuda.amp.autocast():
            predictions, batch_loss = model(data, targets)

        # Debugging: Check prediction values
        print(f"Predictions min: {predictions.min()}, max: {predictions.max()}")
        print(f"Targets min: {targets.min()}, max: {targets.max()}")

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(batch_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += batch_loss.item()
        loop.set_postfix(loss=batch_loss.item())

    avg_loss = total_loss / len(loader)
    print(f"Epoch Average Loss: {avg_loss}")

    return avg_loss


def main():
    # Data augmentation and preprocessing
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    # Initialize model
    model = RegularizedUNet(
        in_channels=3, 
        num_classes=1, 
        lambda_tv=0.1, 
        lambda_pd=0.1
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )

    train_loader = get_train_loader(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR, BATCH_SIZE, train_transform, NUM_WORKERS, PIN_MEMORY
    )
    
    val_loader = get_val_loader(
        VAL_IMG_DIR, VAL_MASK_DIR, BATCH_SIZE, val_transform, NUM_WORKERS, PIN_MEMORY
    )

    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        # Training phase
        model.train()
        train_loss = train_fn(train_loader, model, optimizer, scaler, DEVICE)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, targets in val_loader:
                data = data.to(DEVICE)
                if targets.dim() == 3:  # [N, H, W]
                    targets = targets.unsqueeze(1)
                targets = targets.float().to(DEVICE)

                _, batch_loss = model(data, targets)
                val_loss += batch_loss.item()
        
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=f"best_model_epoch_{epoch}.pth")

        save_predictions_as_imgs(
            val_loader, model, folder=f"saved_images/epoch_{epoch}/", device=DEVICE
        )

    print("Training complete!")

if __name__ == "__main__":
    main()
