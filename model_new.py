import torch
import torch.nn as nn
import torch.nn.functional as F

class TVRegularization(nn.Module):
    def __init__(self, weight=1.0):
        super(TVRegularization, self).__init__()
        self.weight = weight

    def forward(self, x):
        # Ensure tensor has 4 dimensions
        if x.dim() != 4:
            raise ValueError(f"Input tensor must have 4 dimensions, got {x.shape}")

        # Debugging: Print shape and sample values
        print(f"Input tensor shape: {x.shape}")
        print(f"Input tensor min: {x.min()}, max: {x.max()}")

        # Clamp extreme values to prevent NaN/Inf
        x = torch.clamp(x, min=-1e6, max=1e6)

        # Horizontal TV
        diff_h = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        
        # Vertical TV
        diff_v = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        
        # Sum of absolute differences
        tv_loss = torch.sum(diff_h) + torch.sum(diff_v)
        
        return self.weight * tv_loss


class PrimalDualRegularization(nn.Module):
    def __init__(self, lambda_pd=1.0, num_iterations=10):
        super(PrimalDualRegularization, self).__init__()
        self.lambda_pd = lambda_pd
        self.num_iterations = num_iterations

    def forward(self, x):
        primal = x.clone()
        dual = torch.zeros_like(x)
        for _ in range(self.num_iterations):
            grad_primal = primal - x
            dual = torch.clamp(dual + grad_primal, min=-self.lambda_pd, max=self.lambda_pd)
            primal = x - dual
        return primal

class RegularizedSegmentationLoss(nn.Module):
    def __init__(self, lambda_tv=1.0, lambda_pd=1.0):
        super(RegularizedSegmentationLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.tv_regularization = TVRegularization(weight=lambda_tv)
        self.primal_dual_reg = PrimalDualRegularization(lambda_pd)

    def forward(self, predictions, targets):
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        targets = targets.long()
        
        if predictions.shape[2:] != targets.shape[1:]:
            predictions = F.interpolate(predictions, size=targets.shape[1:], mode='bilinear', align_corners=False)
        data_loss = self.cross_entropy(predictions, targets)
        tv_loss = self.tv_regularization(predictions)
        total_loss = data_loss + tv_loss
        return total_loss

class RegularizedUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, base_channels=64, lambda_tv=1.0, lambda_pd=1.0):
        super(RegularizedUNet, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        self.loss_fn = RegularizedSegmentationLoss(lambda_tv, lambda_pd)

    def forward(self, x, targets=None):
        # Encoder
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        
        # Decoder
        x = self.decoder1(x2)
        x = self.final_conv(x)
    
        # Normalize predictions
        x = torch.clamp(x, min=-1.0, max=1.0)  # Restrict values to a reasonable range
    
        # Compute loss if targets are provided
        if targets is not None:
            loss = self.loss_fn(x, targets)
            return x, loss
            
        return x
    
    def initialize_weights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

    # Apply initialization in the UNet constructor
    self.apply(initialize_weights)


