import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class Preprocessor(nn.Module):
    def __init__(self, in_features_node, in_features_edge, nhid_node, nhid_edge, num_node, inter=5):
        super(Preprocessor, self).__init__()
        self.in_features_node = in_features_node
        self.in_features_edge = in_features_edge
        self.p = nhid_node
        self.q = nhid_edge
        self.n = num_node
        self.inter = inter
        self.hidden_size_1 = 50
        self.hidden_size_2 = 50

        self.mlp_ve1 = nn.Sequential(
            nn.Linear(in_features_node * 2 + in_features_edge, self.hidden_size_1),
            nn.ReLU(),
            nn.Linear(self.hidden_size_1, self.hidden_size_1),
            nn.ReLU(),
            nn.Linear(self.hidden_size_1, self.q)
        )

        self.mlp_vv1 = nn.Sequential(
            nn.Linear(self.q + self.in_features_node, self.hidden_size_1),
            nn.ReLU(),
            nn.Linear(self.hidden_size_1, self.inter),
            nn.ReLU()
        )

        self.mlp_ee1 = nn.Sequential(
            nn.Linear(self.hidden_size_2, self.hidden_size_2),
            nn.ReLU(),
            nn.Linear(self.hidden_size_2, self.q)
        )

        self.mlp_ee2 = nn.Sequential(
            nn.Linear(self.q + self.in_features_edge, self.hidden_size_2),
            nn.ReLU(),
            nn.Linear(self.hidden_size_2, self.inter),
            nn.ReLU(),
        )

        self.mlp_ve2 = nn.Sequential(
            nn.Linear(inter * 3, self.hidden_size_1),
            nn.ReLU(),
            nn.Linear(self.hidden_size_1, self.hidden_size_1),
            nn.ReLU(),
            nn.Linear(self.hidden_size_1, self.q)
        )

        self.mlp_vv2 = nn.Sequential(
            nn.Linear(self.q + self.in_features_node, self.hidden_size_1),
            nn.ReLU(),
            # nn.Linear(self.hidden_size_1, self.Ds), # todo: to match the hidden dim
            nn.Linear(self.hidden_size_1, self.p),
            nn.ReLU()
        )

        self.mlp_ee3 = nn.Sequential(
            nn.Linear(self.hidden_size_2, self.hidden_size_2),
            nn.ReLU(),
            nn.Linear(self.hidden_size_2, self.q)
        )

        self.mlp_ee4 = nn.Sequential(
            nn.Linear(self.q + self.in_features_edge, self.hidden_size_2),
            nn.ReLU(),
            nn.Linear(self.hidden_size_2, self.in_features_edge)
        )

        self.m1 = nn.Softmax(dim=1)
        self.m2 = nn.Softmax(dim=1)

        self.w1_1 = Parameter(torch.empty((self.hidden_size_2, in_features_node)))
        self.w1_2 = Parameter(torch.empty((self.hidden_size_2, in_features_edge)))
        nn.init.trunc_normal_(self.w1_1, std=0.1)
        nn.init.trunc_normal_(self.w1_2, std=0.1)
        self.b1 = Parameter(torch.zeros(self.hidden_size_2))

        self.w2_1 = Parameter(torch.empty((self.hidden_size_2, inter)))
        self.w2_2 = Parameter(torch.empty((self.hidden_size_2, inter)))
        nn.init.trunc_normal_(self.w2_1, std=0.1)
        nn.init.trunc_normal_(self.w2_2, std=0.1)
        self.b2 = Parameter(torch.zeros(self.hidden_size_2))

    def forward(self, R, S, node_data, Ra_data):
        '''
        if adj is not self.adj:
            self.adj = adj
            self.adj_rho = get_spectral_rad(adj)
        '''
        self.adj_rho = 1

        # three layers and two MLP
        B1 = torch.cat((node_data.T @ R, node_data.T @ S, Ra_data.T))  # node to edge (m)
        He = self.mlp_ve1(B1.T)  # edge mlp (phi_E_O) influence-on-node TODO not permutation invariant?
        Hv = R @ He # + Rs @ He  # edge to node
        Hv = torch.cat((Hv, node_data), dim=1)  # node (a_O)
        Hv2 = self.mlp_vv1(Hv)  # (phi_U_O) # TODO inter=5?
        # updating the edge
        # He = self.c_mlp_1(B1.T)  # (phi_E_R) influence-on-edge TODO
        He = F.relu(F.linear(B1.T, torch.cat((self.w1_1, self.w1_2, self.w1_1), dim=1), self.b1))
        ER1 = self.mlp_ee1(He)
        He = torch.cat((ER1, Ra_data), dim=1)  # (a_R)
        He2 = self.mlp_ee2(He)  # (phi_U_R) TODO no edge updating?

        # H_map = Rr @ self.map(He2)

        # three layers and two MLP
        B2 = torch.cat((Hv2.T @ R, Hv2.T @ S, He2.T))  # node to edge
        He = self.mlp_ve2(B2.T)  # edge mlp edge mlp (phi_E_O)
        Hv = R @ He # + Rs @ He  # edge to node
        Hv = torch.cat((Hv, node_data), dim=1)  # node  (a_O)
        Hv3 = self.mlp_vv2(Hv)  # (phi_U_O)
        # Hv3 = F.relu(Hv3)
        # updating the edge
        # He = self.c_mlp_2(B1.T)  # (phi_E_R)
        He = F.relu(F.linear(B2.T, torch.cat((self.w2_1, self.w2_2, self.w2_1), dim=1), self.b2))
        ER2 = self.mlp_ee3(He)
        He = torch.cat((ER2, Ra_data), dim=1)  # (a_R)
        He_logits3 = self.mlp_ee4(He)  # (phi_U_R)
        He3 = self.m1(He_logits3)

        return Hv3, He3, He_logits3 # , H_map


class NEC_Cell(nn.Module):
    def __init__(self, p, q):
        super(NEC_Cell, self).__init__()
        self.p = p
        self.q = q
        self.W_ve = Parameter(torch.FloatTensor(self.q, self.p))
        self.W_ev = Parameter(torch.FloatTensor(self.p, self.q))
        self.m = nn.ReLU()

    def init(self):
        stdv = 1. / math.sqrt(self.p * self.q)
        # stdv = 0.1
        # stdv = 1.
        self.W_ve.data.uniform_(-stdv, stdv)
        self.W_ev.data.uniform_(-stdv, stdv)

    def forward(self, X, R, B):
        X = B if X is None else X
        temp = self.W_ve @ X
        support_He1 = temp @ R
        support_He2 = self.m(support_He1)
        support1 = self.W_ev @ support_He2
        support2 = support1 @ R.T  # support = torch.spmm(support, Rr.T)
        X_new = self.m(support2 + B)
        return X_new

class INECN_Layer(nn.Module):
    """
    An Implicit Graph Neural Network Layer (IGNN)
    """

    def __init__(self, in_features_node, in_features_edge, nhid_node, nhid_edge, num_node, kappa=0.99, b_direct=False):
        super(INECN_Layer, self).__init__()
        self.p = nhid_node
        self.q = nhid_edge
        self.n = num_node
        self.k = kappa      # if set kappa=0, projection will be disabled at forward feeding.
        self.b_direct = b_direct
        self.counter = 0

        self.cell = NEC_Cell(nhid_node, nhid_edge)

        self.B_encoder = Preprocessor(in_features_node, in_features_edge, nhid_node, nhid_edge, self.n)

    def forward(self, X_0, R, S, H, E0, F0, phi, A_rho=1.0, fw_mitr=50, bw_mitr=50, A_orig=None):
        """Allow one to use a different A matrix for convolution operation in equilibrium equ"""
        self.counter += 1
        M = H @ H.T
        L = torch.linalg.eig(M).eigenvalues.real
        pf = torch.max(L).item()

        # Hv, He, He_logits3 = self.B_encoder(Rr.T, Rs, F0, E0) # [1000, 64] [19000, 2]
        Hv, He, He_logits3 = self.B_encoder(R.T, S, F0, E0) # [1000, 64] [19000, 2] NECDGT_encoder
        # b_Omega = b_Omega.T  # [De_o,n_node]

        # solution = self.RNN(X_0, H, Hv.T)
        X = X_0
        for _ in range(50):
            X = self.cell(X, H, Hv.T)

        # fixpoint_solution = ImplicitNECFunction_Lite.apply(self.W_ev, self.W_ve, X_0, H, Hv.T, phi, self.counter, fw_mitr, bw_mitr)
        Hv = Hv.T + X
        # Hv = Hv.T
        return Hv, He, He_logits3


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

