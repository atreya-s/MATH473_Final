import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class BCCDDataset(Dataset):
    def __init__(self, 
                 images_dir, 
                 masks_dir, 
                 transform=None, 
                 binary_threshold=128):
        """
        Custom Dataset for BCCD (Blood Cell Count and Detection) Dataset.
        
        Args:
            images_dir (str): Path to directory containing input images.
            masks_dir (str): Path to directory containing segmentation masks.
            transform (callable, optional): Data augmentation pipeline using Albumentations.
            binary_threshold (int): Threshold for binarizing masks.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.binary_threshold = binary_threshold
        
        # Get list of image filenames
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

        # Validate dataset
        self._validate_dataset()

    def _validate_dataset(self):
        """
        Ensure that images and masks match up.
        """
        mask_files = os.listdir(self.masks_dir)
        for img_file in self.image_files:
            mask_file = img_file.replace('.jpg', '.png').replace('.jpeg', '.png')
            if mask_file not in mask_files:
                print(f"Warning: Mask for image {img_file} not found in {self.masks_dir}!")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        """
        Fetch a single data sample.
        
        Args:
            index (int): Index of the sample to fetch.
            
        Returns:
            torch.Tensor: Preprocessed image.
            torch.Tensor: Preprocessed mask.
        """
        # Load image
        img_path = os.path.join(self.images_dir, self.image_files[index])
        image = Image.open(img_path).convert("RGB")  # Convert to RGB
        image = np.array(image)  # Albumentations expects NumPy arrays
    
        # Load mask
        mask_path = os.path.join(
            self.masks_dir,
            self.image_files[index].replace('.jpg', '.png').replace('.jpeg', '.png')
        )
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale
        mask = np.array(mask)  # Convert to NumPy array
        mask = (mask > self.binary_threshold).astype(np.float32)  # Binarize
    
        # Apply transformations
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
    
        # Add channel dimension to mask
        mask = np.expand_dims(mask, axis=0)  # Shape becomes [1, H, W]
        mask = torch.FloatTensor(mask)
    
        return image, mask



# Example usage of the BCCDDataset class
if __name__ == "__main__":
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from torch.utils.data import DataLoader

    # Define transformations
    train_transform = A.Compose([
        A.Resize(512, 512),
        A.Rotate(limit=35, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ToTensorV2(),
    ])

    # Initialize dataset
    train_dataset = BCCDDataset(
        images_dir='/home/axs2220/Math473_Final/Dataset/BCCD/train/original', 
        masks_dir='/home/axs2220/Math473_Final/Dataset/BCCD/train/mask',
        transform=train_transform,
    )

    val_dataset = BCCDDataset(
        images_dir='/home/axs2220/Math473_Final/Dataset/BCCD/test/original', 
        masks_dir='/home/axs2220/Math473_Final/Dataset/BCCD/test/mask',
        transform=val_transform,
    )

    # Test dataset and DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    # Print dataset information
    for i, (image, mask) in enumerate(train_loader):
        print(f"Batch {i}:")
        print(f"Image shape: {image.shape}")  # Expected [B, C, H, W]
        print(f"Mask shape: {mask.shape}")    # Expected [B, 1, H, W]
        break
