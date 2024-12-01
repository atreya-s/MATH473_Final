import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import TVRegularizedUNet 
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    get_train_loader,
    get_val_loader,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = '/home/axs2220/Math473_Final/Dataset/BCCD/train/original'
TRAIN_MASK_DIR = '/home/axs2220/Math473_Final/Dataset/BCCD/train/mask'
VAL_IMG_DIR = '/home/axs2220/Math473_Final/Dataset/BCCD/test/original'

def train_fn(loader, model, optimizer, loss_fn):  # Removed scaler
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # Forward pass
        predictions = model(data)
        loss = loss_fn(torch.sigmoid(predictions), targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update tqdm loop
        loop.set_postfix(loss=loss.item())


        
def main():
    train_transform = A.Compose(
        [
         A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
         A.Rotate(limit=35, p=1.0),
         A.HorizontalFlip(p=0.5),
         A.VerticalFlip(p=0.1),
         A.Normalize(
             mean=[0.5, 0.5, 0.5],  # Adjusted mean
             std=[0.5, 0.5, 0.5],    # Adjusted std
             max_pixel_value=255.0,
             ),
             ToTensorV2(),
             ],
             )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = TVRegularizedUNet(in_channels=3, out_channels=1).to(DEVICE)
    
    # Define loss function
    loss_fn = nn.BCEWithLogitsLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    train_loader = get_train_loader(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    
    val_loader = get_val_loader(
        VAL_IMG_DIR,
        BATCH_SIZE,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )


    
    
    print(f"Total training samples: {len(train_loader.dataset)}")
    print(f"Total validation samples: {len(val_loader.dataset)}")

    # Verify a batch of data
    for batch_idx, (images, masks) in enumerate(train_loader):
        print(f"Batch {batch_idx}")
        print(f"Images shape: {images.shape}")
        print(f"Masks shape: {masks.shape}")
        print(f"Images dtype: {images.dtype}")
        print(f"Masks dtype: {masks.dtype}")
        
        # Check for any NaNs or infs
        print(f"Images NaNs: {torch.isnan(images).any()}")
        print(f"Masks NaNs: {torch.isnan(masks).any()}")
        print(f"Images infs: {torch.isinf(images).any()}")
        print(f"Masks infs: {torch.isinf(masks).any()}")
        
        break  # Just check the first batch

    # Training loop
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn)

        # Save model checkpoint
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # Generate and save predictions for validation images
        save_predictions_as_imgs(
            val_loader, model, folder=f"saved_images/epoch_{epoch}/", device=DEVICE
        )
        

if __name__ == "__main__":
    main()
