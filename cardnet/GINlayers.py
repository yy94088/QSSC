import torch
from typing import Union, Tuple, Callable
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size

from torch import Tensor
from torch.nn import Parameter, Linear, Sequential, ReLU
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset, uniform, zeros
from torch_geometric.utils import softmax


class NNGINConv(MessagePassing):
	"""
	Add the node embedding with edge embedding
	"""
	def __init__(self, edge_nn: Callable, node_nn: Callable,
				 eps: float = 0.,train_eps: bool = False, aggr: str = 'add', **kwargs):
		super(NNGINConv, self).__init__(aggr=aggr, **kwargs)
		self.edge_nn = edge_nn
		self.node_nn = node_nn
		self.aggr = aggr
		self.initial_eps = eps
		if train_eps:
			self.eps = torch.nn.Parameter(torch.Tensor([eps]))
		else:
			self.register_buffer('eps', torch.Tensor([eps]))


	def reset_parameters(self):
		reset(self.node_nn)
		reset(self.edge_nn)
		self.eps.data.fill_(self.initial_eps)

	def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
		if isinstance(x, Tensor):
			x: OptPairTensor = (x, x)

		# propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
		out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
		x_r = x[1]
		if x_r is not None:
			out += (1 + self.eps) * x_r

		out = self.node_nn(out)
		#print(out.shape)
		return out

	def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
		edge_attr = self.edge_nn(edge_attr)
		return F.relu(x_j + edge_attr)


	def __repr__(self):
		return '{}({}, {})'.format(self.__class__.__name__, self.edge_nn,
                                   self.node_nn)

class NNGINConcatConv(MessagePassing):
	"""
	Concatenate the node embedding with edge embedding
	no self loop
	"""
	def __init__(self, edge_nn: Callable, node_nn: Callable,
				 eps: float = 0.,train_eps: bool = False, aggr: str = 'add', **kwargs):
		super(NNGINConcatConv, self).__init__(aggr=aggr, **kwargs)
		self.edge_nn = edge_nn
		self.node_nn = node_nn
		self.aggr = aggr
		self.initial_eps = eps
		if train_eps:
			self.eps = torch.nn.Parameter(torch.Tensor([eps]))
		else:
			self.register_buffer('eps', torch.Tensor([eps]))


	def reset_parameters(self):
		reset(self.node_nn)
		reset(self.edge_nn)
		self.eps.data.fill_(self.initial_eps)

	def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
		if isinstance(x, Tensor):
			x: OptPairTensor = (x, x)

		# propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
		out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

		out = self.node_nn(out)
		#print(out.shape)
		return out

	def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
		edge_attr = self.edge_nn(edge_attr)
		return torch.cat((x_j, edge_attr), dim= -1)


	def __repr__(self):
		return '{}({}, {})'.format(self.__class__.__name__, self.edge_nn,
                                   self.node_nn)

class NNGINETransformerConv(MessagePassing):
    """
    Graph Transformer-style layer that incorporates edge features,
    addressing limitations of simpler GNNs.
    """
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int,
                 heads: int = 4, dropout: float = 0.1, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(NNGINETransformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.lin_key = Linear(in_channels, heads * out_channels)
        self.lin_query = Linear(in_channels, heads * out_channels)
        self.lin_value = Linear(in_channels, heads * out_channels)
        self.lin_edge = Linear(edge_dim, heads * out_channels)
        
        # Use a single linear layer for the final projection
        self.lin_out = Linear(heads * out_channels, out_channels)
        
        # Separate skip connection projection if in_channels != out_channels
        if in_channels != out_channels:
            self.lin_skip = Linear(in_channels, out_channels, bias=False)
        else:
            self.lin_skip = torch.nn.Identity()
        
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_edge.reset_parameters()
        self.lin_out.reset_parameters()
        if isinstance(self.lin_skip, Linear):
            self.lin_skip.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        # propagate_type: (x: Tensor, edge_attr: Tensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Project multi-head output to out_channels
        out = self.lin_out(out)
        
        # Add residual connection
        out += self.lin_skip(x)
        
        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: OptTensor, size_i: int) -> Tensor:
        
        query = self.lin_query(x_i).view(-1, self.heads, self.out_channels)
        key = self.lin_key(x_j).view(-1, self.heads, self.out_channels)
        value = self.lin_value(x_j).view(-1, self.heads, self.out_channels)
        
        # Incorporate edge features into attention mechanism
        edge_embedding = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        key += edge_embedding

        alpha = (query * key).sum(dim=-1) / (self.out_channels ** 0.5)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value * alpha.unsqueeze(-1)
        
        return out.view(-1, self.heads * self.out_channels)