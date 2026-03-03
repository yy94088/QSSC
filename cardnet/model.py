import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import MLP, FC
from .encoder import CardEncoder
from .predictor import Predictor
from .relation import SubgraphRelationNet, GraphBasedRelationNet


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

        # Subgraph Relation Network (optional)
        self.use_subgraph_relation = getattr(args, 'use_subgraph_relation', False)
        if self.use_subgraph_relation:
            relation_hidden_dim = getattr(args, 'relation_hidden_dim', 128)
            relation_type = getattr(args, 'relation_type', 'attention')  # 'attention' or 'graph'
            
            if relation_type == 'attention':
                self.relation_net = SubgraphRelationNet(
                    input_dim=self.out_g_ch,
                    hidden_dim=relation_hidden_dim,
                    num_heads=4,
                    dropout=0.1
                )
            else:  # graph-based
                self.relation_net = GraphBasedRelationNet(
                    input_dim=self.out_g_ch,
                    hidden_dim=relation_hidden_dim,
                    num_layers=2,
                    dropout=0.1
                )
            print(f"Subgraph Relation Network enabled: type={relation_type}, hidden_dim={relation_hidden_dim}")
        else:
            self.relation_net = None

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
        # Encode graph structure with optional subgraph embeddings
        if self.use_subgraph_relation:
            # Get individual subgraph embeddings before pooling
            x, subgraph_embeds = self.encoder(decomp_x, decomp_edge_index, decomp_edge_attr, 
                                              return_subgraph_embeds=True)
            
            # Apply relation network to enhance subgraph embeddings
            if subgraph_embeds is not None and subgraph_embeds.size(0) > 1:
                if isinstance(self.relation_net, SubgraphRelationNet):
                    enhanced_embeds, _ = self.relation_net(subgraph_embeds)
                else:  # GraphBasedRelationNet
                    enhanced_embeds = self.relation_net(subgraph_embeds)
                
                # Re-pool the enhanced embeddings
                if self.pool_type == "sum":
                    x = torch.sum(enhanced_embeds, dim=0).unsqueeze(dim=0)
                elif self.pool_type == "mean":
                    x = torch.mean(enhanced_embeds, dim=0).unsqueeze(dim=0)
                elif self.pool_type == "max":
                    x, _ = torch.max(enhanced_embeds, dim=0, keepdim=True)
                else:  # attention pooling
                    att_weights = self.encoder.att_layer(enhanced_embeds)
                    enhanced_embeds_pooled = att_weights.matmul(enhanced_embeds)
                    x = enhanced_embeds_pooled.view((1, self.num_expert * self.out_g_ch))
        else:
            # Standard encoding without relation network
            x = self.encoder(decomp_x, decomp_edge_index, decomp_edge_attr)

        # Predict cardinality
        output, hid = self.predictor(x)

        return output, x

