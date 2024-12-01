import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS

def total_variation_loss(A):
    """
    Compute Total Variation (TV) loss for a set of activation maps.
    
    Args:
    A (torch.Tensor): Activation maps of shape (B, C, H, W)
    
    Returns:
    torch.Tensor: Total Variation loss
    """
    # Compute vertical and horizontal differences
    diff_vert = torch.abs(A[:, :, 1:, :] - A[:, :, :-1, :]).sum()
    diff_horz = torch.abs(A[:, :, :, 1:] - A[:, :, :, :-1]).sum()
    
    # Return total variation
    return (diff_vert + diff_horz) / (A.shape[0] * A.shape[1])

def regularized_softmax(omega, lam):
    """
    Compute the Regularized Softmax activation function.
    
    Args:
    omega (torch.Tensor): Input logits
    lam (float): Regularization parameter
    
    Returns:
    torch.Tensor: Regularized Softmax activations
    """
    # Solve the optimization problem in Equation (3.6)
    A = omega.clone().detach().requires_grad_()
    optimizer = LBFGS([A], lr=1e-3)

    def closure():
        optimizer.zero_grad()
        loss = -torch.sum(A * torch.log(F.softmax(A, dim=1) + 1e-8)) + lam * torch.norm(A, p=1)
        loss.backward()
        return loss

    optimizer.step(closure)
    return F.softmax(A, dim=1)

class TVRegularizedUNet(nn.Module):
    """
    U-Net architecture with integrated Total Variation and Regularized Softmax regularization
    """
    def __init__(self, in_channels=1, out_channels=1, tv_weight=0.1, softmax_weight=0.1):
        """
        Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        tv_weight (float): Weight for Total Variation regularization
        softmax_weight (float): Weight for Regularized Softmax regularization
        """
        super(TVRegularizedUNet, self).__init__()
        
        # Regularization weights
        self.tv_weight = tv_weight
        self.softmax_weight = softmax_weight
        
        # Encoder (Downsampling)
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)
        
        # Bridge
        self.bridge = self._block(512, 1024)
        
        # Decoder (Upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._block(128, 64)
        
        # Final convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def _block(self, in_channels, out_channels):
        """
        Standard convolutional block with batch normalization and ReLU
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Forward pass with Total Variation and Regularized Softmax regularization
        
        Args:
        x (torch.Tensor): Input image tensor
        
        Returns:
        torch.Tensor: Segmentation output
        """
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        
        # Bridge
        bridge = self.bridge(F.max_pool2d(enc4, 2))
        
        # Decoder
        dec4 = self.upconv4(bridge)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        # Final convolution and Regularized Softmax
        logits = self.final_conv(dec1)
        output = regularized_softmax(logits, self.softmax_weight)
        
        return output
    
    def compute_loss(self, output, target, criterion=nn.MSELoss()):
        """
        Compute loss with Total Variation and Regularized Softmax regularization
        
        Args:
        output (torch.Tensor): Network prediction
        target (torch.Tensor): Ground truth
        criterion (nn.Module): Base loss function
        
        Returns:
        torch.Tensor: Total loss
        """
        # Base loss
        base_loss = criterion(output, target)
        
        # Total Variation regularization loss
        tv_loss = total_variation_loss(output)
        
        # Combined loss
        total_loss = base_loss + self.tv_weight * tv_loss + self.softmax_weight * tv_loss
        
        return total_loss

def train_unet(model, train_loader, optimizer, epochs=10):
    """
    Training function for the TV-regularized U-Net
    
    Args:
    model (TVRegularizedUNet): U-Net model
    train_loader (torch.utils.data.DataLoader): Training data loader
    optimizer (torch.optim.Optimizer): Optimizer
    epochs (int): Number of training epochs
    """
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Compute loss (including regularization)
            loss = model.compute_loss(output, target)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print average loss per epoch
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')


def test():
    x = torch.randn((3, 1, 160, 160))
    model = TVRegularizedUNet(in_channels=1, out_channels=1)
    
    # Forward pass
    preds = model(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", preds.shape)
    
    # Check if the output shape matches the input shape
    assert preds.shape == x.shape, f"Shape mismatch: input {x.shape} vs output {preds.shape}"
    
    print("Test passed successfully!")

if __name__ == "__main__":
    test()