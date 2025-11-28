import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import MLP, FC
from .encoder import CardEncoder
from .predictor import Predictor


class CardNet(nn.Module):
    """Thin wrapper that composes a CardEncoder and a Predictor.

    This preserves the previous `CardNet` external API while making
    encoding and prediction separable for future refactors.
    """
    def __init__(self, args, num_node_feat, num_edge_feat, memory_bank=None):
        super(CardNet, self).__init__()
        self.num_node_feat = num_node_feat
        self.num_edge_feat = num_edge_feat
        self.num_expert = args.num_expert
        self.out_g_ch = args.out_g_ch
        self.num_att_hid = args.num_att_hid
        self.num_mlp_hid = args.num_mlp_hid
        self.num_classes = args.max_classes
        self.multi_task = args.multi_task
        self.pool_type = args.pool_type

        # Encoder handles graph convs, pooling/attention and returns a single feature vector
        self.encoder = CardEncoder(args, self.num_node_feat, self.num_edge_feat)
        predictor_in_ch = self.encoder.mlp_in_ch

        # Keep reference to optional memory bank and pass it to the predictor
        self.memory_bank = memory_bank

        # Predictor handles regression/classification heads (optionally memory-augmented)
        self.predictor = Predictor(in_ch=predictor_in_ch, num_mlp_hid=self.num_mlp_hid,
                       num_classes=self.num_classes, multi_task=self.multi_task,
                       memory_bank=self.memory_bank)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            nn.init.uniform_(module, -0.1, 0.1)

    def forward(self, decomp_x, decomp_edge_index, decomp_edge_attr, card=None):
        # Encode
        x = self.encoder(decomp_x, decomp_edge_index, decomp_edge_attr)

        # Predict
        output, output_cla, hid = self.predictor(x)

        return output, output_cla, x

