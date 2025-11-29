import torch
import torch.nn.functional as F
import torch.nn as nn
from .layers import MLP, FC


class Predictor(nn.Module):
    """Simple regression predictor for cardinality estimation."""
    
    def __init__(self, in_ch, num_mlp_hid):
        super(Predictor, self).__init__()
        self.mlp = MLP(in_ch=in_ch, hid_ch=num_mlp_hid, out_ch=1)

    def forward(self, x):
        """
        Args:
            x: Input features [batch_size, in_ch]
        
        Returns:
            output: Predicted cardinality [batch_size, 1]
            hid: Hidden representation (same as input for compatibility)
        """
        output = self.mlp(x)
        
        # Handle NaN/Inf values
        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.tensor(0.0, device=output.device)
        
        return output, x
