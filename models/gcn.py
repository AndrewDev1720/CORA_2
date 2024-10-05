import numpy as np
from dgl.nn.pytorch import GraphConv
import dgl
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


# class GCNGraphNew(torch.nn.Module):
#     def __init__(self, in_feats, h_feats):
#         super(GCNGraphNew, self).__init__()
#         self.conv1 = GraphConv(in_feats, h_feats)
#         self.conv2 = GraphConv(h_feats, h_feats)
#         self.conv3 = GraphConv(h_feats, h_feats)
#         self.dense = torch.nn.Linear(h_feats, 1)
#         self.maxpool = dgl.nn.pytorch.glob.MaxPooling()

#     def forward(self, g, in_feat, e_weight):
#         h = self.conv1(g, in_feat, e_weight)
#         h = torch.nn.functional.relu(h)
#         h = self.conv2(g, h, e_weight)
#         h = torch.nn.functional.relu(h)
#         h = self.conv3(g, h, e_weight)
#         h = torch.nn.functional.relu(h)
#         g.ndata['h'] = h
#         h = self.maxpool(g, h)  # pooling
#         h = self.dense(h)
#         h = torch.nn.functional.sigmoid(h)
#         return h

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        return x  # Return logits without softmax


# This function will return the initialized GCN model
def get_model(input_dim, hidden_dim, output_dim):
    return GCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

class GCNGraph(torch.nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GCNGraph, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)
        self.conv3 = GraphConv(h_feats, h_feats)
        self.dense1 = torch.nn.Linear(h_feats, 16)
        self.dense2 = torch.nn.Linear(16, 8)
        self.dense3 = torch.nn.Linear(8, 1)

    def forward(self, g, in_feat, e_weight):
        h = self.conv1(g, in_feat, e_weight)
        h = torch.nn.functional.relu(h)
        h = self.conv2(g, h, e_weight)
        h = torch.nn.functional.relu(h)
        h = self.conv3(g, h, e_weight)
        h = torch.nn.functional.relu(h)
        g.ndata['h'] = h
        h = dgl.mean_nodes(g, 'h')  # pooling
        h = self.dense1(h)
        h = torch.nn.functional.relu(h)
        h = self.dense2(h)
        h = torch.nn.functional.relu(h)
        h = self.dense3(h)
        h = torch.nn.functional.sigmoid(h)
        return h


class GCNNodeBAShapes(torch.nn.Module):
    # TODO
    def __init__(self, in_feats, h_feats, num_classes, device, if_exp=False):
        super(GCNNodeBAShapes, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)
        self.conv3 = GraphConv(h_feats, num_classes)
        self.if_exp = if_exp
        self.device = device

    def forward(self, g, in_feat, e_weight, target_node):
        # map target node index
        x = torch.cat((torch.tensor([0]).to(self.device), torch.cumsum(g.batch_num_nodes(), dim=0)), dim=0)[:-1]
        target_node = target_node + x
        h = self.conv1(g, in_feat, e_weight)
        h = torch.nn.functional.relu(h)
        h = self.conv2(g, h, e_weight)
        h = torch.nn.functional.relu(h)
        h = self.conv3(g, h, e_weight)
        if self.if_exp:  # if in the explanation mod, should add softmax layer
            h = torch.nn.functional.softmax(h)
        g.ndata['h'] = h
        return g.ndata['h'][target_node]


class GCNNodeTreeCycles(torch.nn.Module):
    # TODO
    def __init__(self, in_feats, h_feats, num_classes, if_exp=False):
        super(GCNNodeTreeCycles, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)
        self.conv3 = GraphConv(h_feats, num_classes)
        self.if_exp = if_exp

    def forward(self, g, in_feat, e_weight, target_node):
        # map target node index
        x = torch.cat((torch.tensor([0]), torch.cumsum(g.batch_num_nodes(), dim=0)), dim=0)[:-1]
        target_node = target_node + x

        h = self.conv1(g, in_feat, e_weight)
        h = torch.nn.functional.relu(h)
        h = self.conv2(g, h, e_weight)
        h = torch.nn.functional.relu(h)
        h = self.conv3(g, h, e_weight)
        if self.if_exp:  # if in the explanation mod, should add softmax layer
            h = torch.nn.functional.sigmoid(h)
        g.ndata['h'] = h
        return g.ndata['h'][target_node]


class GCNNodeCiteSeer(torch.nn.Module):
    # TODO
    def __init__(self, in_feats, h_feats, num_classes, if_exp=False):
        super(GCNNodeCiteSeer, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
        self.if_exp = if_exp

    def forward(self, g, in_feat, e_weight, target_node):
        # map target node index
        x = torch.cat((torch.tensor([0]), torch.cumsum(g.batch_num_nodes(), dim=0)), dim=0)[:-1]
        target_node = target_node + x

        h = self.conv1(g, in_feat, e_weight)
        h = torch.nn.functional.relu(h)
        h = self.conv2(g, h, e_weight)
        if self.if_exp:  # if in the explanation mod, should add softmax layer
            h = torch.nn.functional.softmax(h)
        g.ndata['h'] = h
        return g.ndata['h'][target_node]
