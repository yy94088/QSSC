import torch
import torch.nn.functional as F
import torch.nn as nn
from .layers import MLP, FC

from .improved_memory import MemoryAugmentedPredictor


class Predictor(nn.Module):
    def __init__(self, in_ch, num_mlp_hid, num_classes, multi_task, memory_bank=None):
        super(Predictor, self).__init__()
        self.multi_task = multi_task
        self.mlp = MLP(in_ch=in_ch, hid_ch=num_mlp_hid, out_ch=1)
        self.fc_hid = FC(in_ch=in_ch, out_ch=num_mlp_hid)
        self.fc_reg = FC(in_ch=num_mlp_hid, out_ch=1)
        self.fc_cla = FC(in_ch=num_mlp_hid, out_ch=num_classes)

        # Optional memory augmentation
        self.memory_bank = memory_bank
        if self.memory_bank is not None:
            # Create a refiner that can combine base predictions with memory
            self.memory_refiner = MemoryAugmentedPredictor(base_predictor=None, memory_bank=self.memory_bank)
        else:
            self.memory_refiner = None

    def forward(self, x):
        # x is expected shape [1, in_ch]
        if self.multi_task:
            hid_g = F.relu(self.fc_hid(x))
            if torch.isnan(hid_g).any() or torch.isinf(hid_g).any():
                hid_g = torch.zeros_like(hid_g)
            output = self.fc_reg(hid_g)
            output_cla = F.log_softmax(self.fc_cla(hid_g), dim=1)
            if torch.isnan(output).any() or torch.isinf(output).any():
                output = torch.tensor(0.0, device=output.device)
            base_pred = output
            hid = hid_g
        else:
            output = self.mlp(x)
            if torch.isnan(output).any() or torch.isinf(output).any():
                output = torch.tensor(0.0, device=output.device)
            base_pred = output
            output_cla = None
            hid = x

        # If memory refiner exists, use it to refine the base prediction
        if self.memory_refiner is not None:
            try:
                refined = self.memory_refiner(x, base_pred)
                # Ensure shape/device consistency
                refined = refined.to(base_pred.device)
                return refined, output_cla, hid
            except Exception:
                # If refinement fails, fall back to base prediction
                return base_pred, output_cla, hid

        return base_pred, output_cla, hid
