"""
Subgraph Relation Network for learning interactions between decomposed subgraphs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SubgraphRelationNet(nn.Module):
    """
    Network to model relationships between decomposed subgraphs.
    
    Uses self-attention mechanism to capture dependencies and interactions
    between subgraphs, allowing the model to understand how different parts
    of the query graph relate to each other.
    """
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, dropout=0.1):
        """
        Args:
            input_dim (int): Dimension of subgraph embeddings
            hidden_dim (int): Hidden dimension for relation modeling
            num_heads (int): Number of attention heads
            dropout (float): Dropout rate
        """
        super(SubgraphRelationNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Project input embeddings to hidden dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head self-attention for subgraph interactions
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output projection back to input dimension
        self.output_projection = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, subgraph_embeddings):
        """
        Args:
            subgraph_embeddings (Tensor): [num_subgraphs, embedding_dim]
                Embeddings of decomposed subgraphs
        
        Returns:
            relation_enhanced (Tensor): [num_subgraphs, embedding_dim]
                Subgraph embeddings enhanced with relation information
            attention_weights (Tensor): [num_heads, num_subgraphs, num_subgraphs]
                Attention weights showing subgraph relationships
        """
        # Handle single subgraph case
        if subgraph_embeddings.dim() == 1:
            subgraph_embeddings = subgraph_embeddings.unsqueeze(0)
        
        num_subgraphs = subgraph_embeddings.size(0)
        
        # Project to hidden dimension
        x = self.input_projection(subgraph_embeddings)  # [num_subgraphs, hidden_dim]
        
        # Add batch dimension for attention (batch_first=True)
        x = x.unsqueeze(0)  # [1, num_subgraphs, hidden_dim]
        
        # Self-attention to capture subgraph relationships
        # Each subgraph attends to all other subgraphs
        attn_output, attn_weights = self.attention(x, x, x)
        
        # Residual connection + layer norm
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        
        # Residual connection + layer norm
        x = self.norm2(x + self.dropout(ffn_output))
        
        # Remove batch dimension
        x = x.squeeze(0)  # [num_subgraphs, hidden_dim]
        
        # Project back to input dimension
        relation_enhanced = self.output_projection(x)  # [num_subgraphs, input_dim]
        
        # Add residual connection from input
        relation_enhanced = relation_enhanced + subgraph_embeddings
        
        return relation_enhanced, attn_weights.squeeze(0)


class GraphBasedRelationNet(nn.Module):
    """
    Alternative graph-based approach to model subgraph relationships.
    
    Treats subgraphs as nodes in a meta-graph and applies GNN to learn
    their relationships. Can be used when subgraphs have explicit connections.
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.1):
        """
        Args:
            input_dim (int): Dimension of subgraph embeddings
            hidden_dim (int): Hidden dimension for GNN layers
            num_layers (int): Number of GNN layers
            dropout (float): Dropout rate
        """
        super(GraphBasedRelationNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer
        self.convs.append(nn.Linear(input_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(nn.Linear(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
    
    def build_complete_graph(self, num_nodes):
        """
        Build a complete graph (all-to-all connections) for subgraphs.
        
        Args:
            num_nodes (int): Number of subgraphs
        
        Returns:
            edge_index (Tensor): [2, num_edges] edge list
        """
        # Create fully connected graph
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # No self-loops
                    edges.append([i, j])
        
        if len(edges) == 0:
            # Single node case
            return torch.zeros((2, 0), dtype=torch.long)
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        return edge_index
    
    def forward(self, subgraph_embeddings, edge_index=None):
        """
        Args:
            subgraph_embeddings (Tensor): [num_subgraphs, embedding_dim]
            edge_index (Tensor, optional): [2, num_edges] edge connections
                If None, assumes fully connected meta-graph
        
        Returns:
            relation_enhanced (Tensor): [num_subgraphs, embedding_dim]
        """
        # Handle single subgraph case
        if subgraph_embeddings.dim() == 1:
            subgraph_embeddings = subgraph_embeddings.unsqueeze(0)
        
        num_subgraphs = subgraph_embeddings.size(0)
        device = subgraph_embeddings.device
        
        # Build edge connections if not provided (fully connected)
        if edge_index is None:
            edge_index = self.build_complete_graph(num_subgraphs).to(device)
        
        # Apply GNN layers with message passing
        x = subgraph_embeddings
        
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # Transform features
            x_transformed = conv(x)
            
            # Message passing: aggregate from neighbors
            if edge_index.size(1) > 0:
                # Simple aggregation: mean of neighbor features
                src, dst = edge_index
                num_edges = edge_index.size(1)
                
                # Aggregate messages
                messages = x_transformed[src]  # [num_edges, hidden_dim]
                
                # Sum aggregation
                aggregated = torch.zeros_like(x_transformed)
                aggregated.index_add_(0, dst, messages)
                
                # Count neighbors for averaging
                degree = torch.zeros(num_subgraphs, device=device)
                degree.index_add_(0, dst, torch.ones(num_edges, device=device))
                degree = degree.clamp(min=1).unsqueeze(1)
                
                # Average aggregation
                x_transformed = aggregated / degree
            
            # Add self-connection (residual)
            if i == 0:
                # First layer: project input dimension
                x = x_transformed
            else:
                x = x + x_transformed
            
            # Normalization and activation
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Project back to input dimension
        relation_enhanced = self.output_projection(x)
        
        # Residual connection from input
        relation_enhanced = relation_enhanced + subgraph_embeddings
        
        return relation_enhanced
