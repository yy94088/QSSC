import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import MLP, FC
from .encoder import CardEncoder
from .predictor import Predictor


class CardNet(nn.Module):
    """Graph neural network for query cardinality estimation.
    
    Composes a CardEncoder and a Predictor for end-to-end learning.
    """
    def __init__(self, args, num_node_feat, num_edge_feat):
        super(CardNet, self).__init__()
        self.num_node_feat = num_node_feat
        self.num_edge_feat = num_edge_feat
        self.num_expert = args.num_expert
        self.out_g_ch = args.out_g_ch
        self.num_att_hid = args.num_att_hid
        self.num_mlp_hid = args.num_mlp_hid
        self.pool_type = args.pool_type

        # Encoder: graph convolutions + pooling
        self.encoder = CardEncoder(args, self.num_node_feat, self.num_edge_feat)
        predictor_in_ch = self.encoder.mlp_in_ch

        # Predictor: regression head only
        self.predictor = Predictor(in_ch=predictor_in_ch, num_mlp_hid=self.num_mlp_hid)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            nn.init.uniform_(module, -0.1, 0.1)

    def forward(self, decomp_x, decomp_edge_index, decomp_edge_attr):
        """
        Forward pass for cardinality estimation.
        
        Args:
            decomp_x: List of node feature tensors for decomposed subgraphs
            decomp_edge_index: List of edge index tensors
            decomp_edge_attr: List of edge attribute tensors
        
        Returns:
            output: Predicted log cardinality
            hid: Hidden representation from encoder
        """
        # Encode graph structure
        x = self.encoder(decomp_x, decomp_edge_index, decomp_edge_attr)

        # Predict cardinality
        output, hid = self.predictor(x)

        return output, x

