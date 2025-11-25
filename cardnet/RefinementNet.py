import torch
import torch.nn as nn

class RefinementNet(nn.Module):
    """
    A network to refine a cardinality prediction using similar queries.
    Input:
    - Initial prediction from the main model.
    - Embedding of the current query.
    - Embeddings and cardinalities of k similar queries.
    """
    def __init__(self, input_dim, hidden_dim=64):
        super(RefinementNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # The output is a delta to be added to the initial prediction
        delta = self.net(x)
        return delta
