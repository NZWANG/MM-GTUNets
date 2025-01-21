from typing import Callable, List, Union
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import TopKPooling, TransformerConv
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor
from torch_geometric.utils.repeat import repeat


class GTUNet(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            depth: int,
            edge_dim: int = 1,
            pool_ratios: Union[float, List[float]] = 0.5,
            dropout: float = 0.3,
            sum_res: bool = True,
            act: Union[str, Callable] = 'relu',
    ):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.dropout = dropout
        self.act = activation_resolver(act)
        self.sum_res = sum_res

        self.down_in_hid_conv = TransformerConv(in_channels, hidden_channels, edge_dim=edge_dim, heads=1, beta=True)
        self.down_hid_conv = TransformerConv(hidden_channels, hidden_channels, edge_dim=edge_dim, heads=1, beta=True)

        channels = hidden_channels
        in_channels = channels if sum_res else 2 * channels

        self.up_in_hid_conv = TransformerConv(in_channels, hidden_channels, edge_dim=edge_dim, heads=1, beta=True)
        self.up_in_out_conv = TransformerConv(in_channels, out_channels, edge_dim=edge_dim, heads=1, beta=True)

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(self.down_in_hid_conv)
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(self.down_hid_conv)

        self.up_convs = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(self.up_in_hid_conv)
        self.up_convs.append(self.up_in_out_conv)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_weight: OptTensor = None,
                batch: OptTensor = None) -> Tensor:
        """"""  # noqa: D419
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        if edge_weight is None:
            edge_weight = x.new_ones(edge_index.size(1)).view(-1, 1)

        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        self.perms = []

        for i in range(1, self.depth + 1):
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            self.perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = self.perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index, edge_weight)
            x = self.act(x) if i < self.depth - 1 else x

        x = F.dropout(x, p=self.dropout)

        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, {self.out_channels}, '
                f'depth={self.depth}, pool_ratios={self.pool_ratios})')
