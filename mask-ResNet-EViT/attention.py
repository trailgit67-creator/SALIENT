import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskGuidedAttention(nn.Module):
    def __init__(self, in_channels, out_channels=None, spatial_reduction=8, channel_reduction=8, dropout_rate=0.2):
        super(MaskGuidedAttention, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        
        # Spatial attention layers
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // spatial_reduction, 1),
            nn.BatchNorm2d(in_channels // spatial_reduction),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate * 0.5),  # Lower dropout rate (0.1)

            nn.Conv2d(in_channels // spatial_reduction, in_channels // (spatial_reduction ** 2), 1),
            nn.BatchNorm2d(in_channels // (spatial_reduction ** 2)),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // (spatial_reduction ** 2), 1, 1),
            nn.BatchNorm2d(1)
        )

        # Channel attention layers
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // channel_reduction),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.75),  # Moderate dropout rate (0.15)

            nn.Linear(in_channels // channel_reduction, out_channels),
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        # Spatial attention
        spatial_att = self.spatial_attention(x)
        spatial_att = self.sigmoid(spatial_att) + 1
            
        # Channel attention
        channel_att = self.avg_pool(x).view(batch_size, self.in_channels)
        channel_att = self.channel_attention(channel_att)
        channel_att = channel_att.view(batch_size, self.out_channels, 1, 1)

        # Combine attentions
        att = spatial_att * channel_att
        return att, spatial_att
    
    def get_attention_loss(self, spatial_att, mask):
        """RMSE between per-sample normalized attention and mask."""
        if mask is None:
            return torch.tensor(0.0, device=spatial_att.device)

        B = spatial_att.size(0)

        # --- per-sample min/max normalization ---
        # spatial_att: [B, 1, H, W] (or [B, C, H, W] but here it's 1 channel)
        flat = spatial_att.view(B, -1)                  # [B, H*W]
        min_vals = flat.min(dim=1)[0].view(B, 1, 1, 1)  # [B,1,1,1]
        max_vals = flat.max(dim=1)[0].view(B, 1, 1, 1)  # [B,1,1,1]
        spatial_att_norm = (spatial_att - min_vals) / (max_vals - min_vals + 1e-8)

        # --- resize mask if needed ---
        if mask.shape[-2:] != spatial_att.shape[-2:]:
            mask = F.interpolate(mask.float(), size=spatial_att.shape[-2:], mode='nearest')

        # binarize mask
        mask = (mask > 0.5).float()

        # --- RMSE loss ---
        mse = F.mse_loss(spatial_att_norm, mask, reduction='none')
        attention_loss = torch.sqrt(mse.mean())
        return attention_loss

    '''
    def get_attention_loss(self, spatial_att, mask):
        """Calculate the attention loss using MSE between normalized attention and mask"""
        if mask is None:
            return torch.tensor(0.0, device=spatial_att.device)
            
        # Normalize spatial attention to 0-1 range
        spatial_att_norm = (spatial_att - spatial_att.min()) / (spatial_att.max() - spatial_att.min() + 1e-8)
        
        # Resize mask to match attention size if needed
        if mask.shape[-2:] != spatial_att.shape[-2:]:
            mask = F.interpolate(mask.float(), size=spatial_att.shape[-2:], mode='nearest')
        
        # Convert mask to binary (0s and 1s)
        mask = (mask > 0.5).float()
        
        # Calculate RMSE loss
        mse = F.mse_loss(spatial_att_norm, mask, reduction='none')  # Keep per-element loss
        attention_loss = torch.sqrt(mse.mean())  # Take mean inside sqrt for true RMSE
        return attention_loss
    '''