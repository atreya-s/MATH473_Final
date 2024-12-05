import torch
import torch.nn as nn
import torch.nn.functional as F

class TVRegularization(nn.Module):
    """
    Total Variation (TV) Regularization module
    Implements the TV regularization as described in the paper
    """
    def __init__(self, weight=1.0):
        super(TVRegularization, self).__init__()
        self.weight = weight

    def forward(self, x):
        """
        Compute Total Variation loss
        x: Input tensor (typically network predictions)
        """
        # Horizontal TV
        diff_h = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        
        # Vertical TV
        diff_v = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        
        # Sum of absolute differences
        tv_loss = torch.sum(diff_h) + torch.sum(diff_v)
        
        return self.weight * tv_loss

class PrimalDualRegularization(nn.Module):
    """
    Primal-Dual Regularization module
    Implements the primal-dual optimization as described in the paper
    """
    def __init__(self, lambda_pd=1.0, num_iterations=10):
        super(PrimalDualRegularization, self).__init__()
        self.lambda_pd = lambda_pd
        self.num_iterations = num_iterations

    def forward(self, x):
        """
        Apply primal-dual optimization
        x: Input tensor (network predictions)
        """
        # Clone the original input as the initial primal variable
        primal = x.clone()
        
        # Initialize dual variable
        dual = torch.zeros_like(x)
        
        for _ in range(self.num_iterations):
            # Compute gradient of primal variable
            grad_primal = primal - x
            
            # Update dual variable with projection
            dual = torch.clamp(dual + grad_primal, 
                               min=-self.lambda_pd, 
                               max=self.lambda_pd)
            
            # Update primal variable
            primal = x - dual
        
        return primal

class RegularizedSegmentationLoss(nn.Module):
    """
    Combined loss function with data fidelity and regularization terms
    """
    def __init__(self, lambda_tv=1.0, lambda_pd=1.0):
        super(RegularizedSegmentationLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.tv_regularization = TVRegularization(weight=lambda_tv)
        self.primal_dual_reg = PrimalDualRegularization(lambda_pd)

    def forward(self, predictions, targets):
        """
        Compute the regularized loss
        predictions: Network output logits
        targets: Ground truth segmentation masks
        """
        # Resize predictions to match targets if needed
        if predictions.shape != targets.shape:
            predictions = F.interpolate(
                predictions, 
                size=targets.shape[1:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Data fidelity term (Cross-Entropy Loss)
        data_loss = self.cross_entropy(predictions, targets)
        
        # Total Variation regularization
        tv_loss = self.tv_regularization(predictions)
        
        # Combine losses
        total_loss = data_loss + tv_loss
        
        return total_loss

class RegularizedUNet(nn.Module):
    """
    U-Net architecture with TV and Primal-Dual regularization
    """
    def __init__(self, 
                 in_channels=3, 
                 num_classes=2, 
                 base_channels=64,
                 lambda_tv=1.0, 
                 lambda_pd=1.0):
        super(RegularizedUNet, self).__init__()
        
        # Encoder path
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        
        # Decoder path
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        
        # Regularization modules
        self.tv_regularization = TVRegularization(weight=lambda_tv)
        self.primal_dual_reg = PrimalDualRegularization(lambda_pd)
        
        # Loss function
        self.loss_fn = RegularizedSegmentationLoss(lambda_tv, lambda_pd)

    def forward(self, x, targets=None):
        # Encoder
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        
        # Decoder
        x = self.decoder1(x2)
        x = self.final_conv(x)
        
        # Primal-dual regularization
        x = self.primal_dual_reg(x)
        
        # Compute loss if targets are provided
        if targets is not None:
            loss = self.loss_fn(x, targets)
            return x, loss
        
        return x

# Example usage
if __name__ == "__main__":
    # Initialize the regularized U-Net
    model = RegularizedUNet(
        in_channels=3, 
        num_classes=2, 
        lambda_tv=0.1, 
        lambda_pd=0.1
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Dummy data
    inputs = torch.rand(1, 3, 256, 256)
    targets = torch.randint(0, 2, (1, 256, 256))
    
    # Training step
    model.train()
    optimizer.zero_grad()
    outputs, loss = model(inputs, targets)
    loss.backward()
    optimizer.step()
    
    
    print(f"Loss: {loss.item()}")
