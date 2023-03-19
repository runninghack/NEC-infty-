import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import INECN_Layer, INECN_Layer_Context, IGNN_Layer
from torch.nn import Parameter


class INECN(nn.Module):
    def __init__(self, in_features_node, in_features_edge, nhid_node, nhid_edge, num_node, dropout, kappa=0.9):
        super(INECN, self).__init__()
        self.adj_rho = None

        #three layers and two MLP
        # self.ig1 = ImplicitNECGraph(Ds, Dr, nhid_node, nhid_edge, num_node, kappa)
        self.ig1 = INECN_Layer(in_features_node, in_features_edge, nhid_node, nhid_edge, num_node, kappa)
        # self.ig2 = ImplicitGraph(nhid, nhid, num_node, kappa)
        # self.ig3 = ImplicitGraph(nhid, nhid, num_node, kappa)
        self.dropout = dropout
        self.X_0 = None
        self.V_0 = nn.Linear(nhid_node, nhid_node)
        self.V_1 = nn.Linear(nhid_node, in_features_node)

    def forward(self, R, S, H, node_data, Ra_data):
        '''
        if adj is not self.adj:
            self.adj = adj
            self.adj_rho = get_spectral_rad(adj)
        '''
        self.adj_rho = 1

        #three layers and two MLP
        x, He, He_logits3 = self.ig1(self.X_0, R, S, H, Ra_data, node_data, F.relu, self.adj_rho)
        x = x.T
        # x = self.ig2(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig)
        # x = self.ig3(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig).T
        x = F.relu(self.V_0(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.V_1(x)
        return x, He, He_logits3


class IGNN(nn.Module):
    def __init__(self, in_features_node, in_features_edge, nhid_node, nhid_edge, num_node, dropout, kappa=0.9):
        super(IGNN, self).__init__()
        self.adj_rho = None

        #three layers and two MLP
        # self.ig1 = ImplicitNECGraph(Ds, Dr, nhid_node, nhid_edge, num_node, kappa)
        self.ig1 = IGNN_Layer(in_features_node, in_features_edge, nhid_node, nhid_edge, num_node, kappa)
        # self.ig2 = ImplicitGraph(nhid, nhid, num_node, kappa)
        # self.ig3 = ImplicitGraph(nhid, nhid, num_node, kappa)
        self.dropout = dropout
        self.X_0 = None
        self.V_0 = nn.Linear(nhid_node, nhid_node)
        self.V_1 = nn.Linear(nhid_node, in_features_node)

    def forward(self, R, S, H, node_data, Ra_data):
        '''
        if adj is not self.adj:
            self.adj = adj
            self.adj_rho = get_spectral_rad(adj)
        '''
        self.adj_rho = 1

        #three layers and two MLP
        x = self.ig1(self.X_0, R, S, H, Ra_data, node_data, F.relu, self.adj_rho)
        x = x.T
        # x = self.ig2(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig)
        # x = self.ig3(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig).T
        x = F.relu(self.V_0(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.V_1(x)
        return x

class INECN_Context(nn.Module):
    def __init__(self, in_features_node, in_features_edge, nhid_node, nhid_edge, num_node, dropout, kappa=0.9):
        super(INECN_Context, self).__init__()
        self.adj_rho = None

        #three layers and two MLP
        self.ig1 = INECN_Layer_Context(in_features_node, in_features_edge, nhid_node, nhid_edge, num_node, kappa)
        # self.dropout = dropout
        self.X_0 = None
        #self.V_0 = nn.Linear(20, 2)
        # self.V_1 = nn.Linear(nhid_node, in_features_node)
        #
        # self.V_2 = nn.Linear(nhid_edge, nhid_edge)
        # self.V_3 = nn.Linear(nhid_edge, in_features_edge)
        #
        # self.m1 = nn.Softmax(dim=1)

    def forward(self, R, S, H, node_data, Ra_data, X_data):
        '''
        if adj is not self.adj:
            self.adj = adj
            self.adj_rho = get_spectral_rad(adj)
        '''
        self.adj_rho = 1

        #three layers and two MLP
        Hv, Hv_digits, He, loss_EO_ER = self.ig1(self.X_0, R, S, H, Ra_data, node_data, X_data, F.relu, self.adj_rho)
        # x = x.T
        # # x = self.ig2(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig)
        # # x = self.ig3(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig).T
        # x = F.relu(self.V_0(x))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.V_1(x)
        # x = self.m1(x)
        #
        # # He = He.T
        # He = F.relu(self.V_2(He))
        # He = F.dropout(He, self.dropout, training=self.training)
        # He = self.V_3(He)
        # Hv = Hv.T
        # Hv = self.V_0(Hv)

        return Hv, Hv_digits, He, loss_EO_ER

