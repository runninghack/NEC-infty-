import math
import torch
import torch.sparse
from torch.nn import Parameter
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from utils import projection_norm_inf, projection_NEC_DGT, SparseDropout
from functions import ImplicitFunction, ImplicitNECFunction, NECFunction, ImplicitNECFunction_Lite
# from models import NECDGT


class INECN_Layer(Module):
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

        # self.W_ve = Parameter(torch.FloatTensor(self.De_r, 2 * self.De_o))
        self.W_ve = Parameter(torch.FloatTensor(self.q, self.p))
        self.W_ev = Parameter(torch.FloatTensor(self.p, self.q))
        self.Omega_1 = Parameter(torch.FloatTensor(in_features_node, self.p))
        self.Omega_2 = Parameter(torch.FloatTensor(in_features_edge, self.p))
        self.B_encoder = Preprocessor(in_features_node, in_features_edge, nhid_node, nhid_edge, self.n)
        # self.B_encoder = NECDGT_encoder(Ds=1, Dr=2, nhid_node=20, nhid_edge=20, num_node=20)
        # self.bias = Parameter(torch.FloatTensor(self.m, 1))
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.p * self. q)
        # stdv = 0.1
        # stdv = 1.
        self.W_ve.data.uniform_(-stdv, stdv)
        self.W_ev.data.uniform_(-stdv, stdv)


    def forward(self, X_0, R, S, H, E0, F0, phi, A_rho=1.0, fw_mitr=50, bw_mitr=50, A_orig=None):
        """Allow one to use a different A matrix for convolution operation in equilibrium equ"""
        self.counter += 1
        M = H @ H.T
        L = torch.linalg.eig(M).eigenvalues.real
        pf = torch.max(L).item()
        kappa = self.k/pf
        norm_XY = torch.norm(self.W_ev @ self.W_ve, float('inf'))
        if norm_XY > kappa:
            self.W_ev, self.W_ve = projection_NEC_DGT(self.W_ev, self.W_ve, kappa)

        # Hv, He, He_logits3 = self.B_encoder(Rr.T, Rs, F0, E0) # [1000, 64] [19000, 2]
        Hv, He, He_logits3 = self.B_encoder(R.T, S, F0, E0) # [1000, 64] [19000, 2] NECDGT_encoder
        # b_Omega = b_Omega.T  # [De_o,n_node]
        fixpoint_solution = ImplicitNECFunction_Lite.apply(self.W_ev, self.W_ve, X_0, H, Hv.T, phi, self.counter, fw_mitr, bw_mitr)
        Hv = Hv.T + fixpoint_solution
        # Hv = Hv.T
        return Hv, He, He_logits3
        # return ImplicitNECFunction.apply(self.W_ev, self.W_ve, X_0, Rr, Rs, b_Omega, phi, fw_mitr, bw_mitr)

class IGNN_Layer(Module):
    """
    An Implicit Graph Neural Network Layer (IGNN)
    """

    def __init__(self, in_features_node, in_features_edge, nhid_node, nhid_edge, num_node, kappa=0.99, b_direct=False):
        super(IGNN_Layer, self).__init__()
        self.p = nhid_node
        self.q = nhid_edge
        self.n = num_node
        self.k = kappa      # if set kappa=0, projection will be disabled at forward feeding.
        self.b_direct = b_direct
        self.counter = 0

        # self.W_ve = Parameter(torch.FloatTensor(self.De_r, 2 * self.De_o))
        self.W = Parameter(torch.FloatTensor(self.q, self.q))
        # self.W_ev = Parameter(torch.FloatTensor(self.p, self.q))
        # self.Omega_1 = Parameter(torch.FloatTensor(in_features_node, self.p))
        # self.Omega_2 = Parameter(torch.FloatTensor(in_features_edge, self.p))
        self.B_encoder = nn.Linear(in_features_node, self.p)
        # self.B_encoder = NECDGT_encoder(Ds=1, Dr=2, nhid_node=20, nhid_edge=20, num_node=20)
        # self.bias = Parameter(torch.FloatTensor(self.m, 1))
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.p * self. q)
        # stdv = 0.1
        # stdv = 1.
        self.W.data.uniform_(-stdv, stdv)
        # self.W_ev.data.uniform_(-stdv, stdv)


    def forward(self, X_0, R, S, H, E0, F0, phi, A_rho=1.0, fw_mitr=50, bw_mitr=50, A_orig=None):
        """Allow one to use a different A matrix for convolution operation in equilibrium equ"""
        self.counter += 1
        # M = H @ H.T
        # L = torch.linalg.eig(M).eigenvalues.real
        # pf = torch.max(L).item()
        # kappa = self.k/pf
        # norm_XY = torch.norm(self.W_ev @ self.W_ve, float('inf'))
        # if norm_XY > kappa:
        #     self.W_ev, self.W_ve = projection_NEC_DGT(self.W_ev, self.W_ve, kappa)

        # Hv, He, He_logits3 = self.B_encoder(Rr.T, Rs, F0, E0) # [1000, 64] [19000, 2]
        if self.k is not None:  # when self.k = 0, A_rho is not required
            self.W = projection_norm_inf(self.W, kappa=0.9 / 1)
        Hv = self.B_encoder(F0) # [1000, 64] [19000, 2] NECDGT_encoder
        # b_Omega = b_Omega.T  # [De_o,n_node]
        fixpoint_solution = ImplicitFunction.apply(self.W,  X_0, H, Hv.T, phi, fw_mitr, bw_mitr)
        Hv = Hv.T + fixpoint_solution
        # Hv = Hv.T
        return Hv
        # return ImplicitNECFunction.apply(self.W_ev, self.W_ve, X_0, Rr, Rs, b_Omega, phi, fw_mitr, bw_mitr)

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


class INECN_Layer_Context(Module):
    """
    An Implicit Graph Neural Network Layer (IGNN)
    """

    def __init__(self, in_features_node, in_features_edge, nhid_node, nhid_edge, num_node, kappa=0.99, b_direct=False):
        super(INECN_Layer_Context, self).__init__()
        self.p = nhid_node
        self.q = nhid_edge
        self.n = num_node
        self.k = kappa      # if set kappa=0, projection will be disabled at forward feeding.
        self.b_direct = b_direct
        self.counter = 0

        # self.W_ve = Parameter(torch.FloatTensor(self.De_r, 2 * self.De_o))
        self.W_ve = Parameter(torch.FloatTensor(self.q, self.p))
        self.W_ev = Parameter(torch.FloatTensor(self.p, self.q))
        self.Omega_1 = Parameter(torch.FloatTensor(in_features_node, self.p))
        self.Omega_2 = Parameter(torch.FloatTensor(in_features_edge, self.p))
        self.B_encoder = Preprocessor_Context(in_features_node=in_features_node, in_features_edge=in_features_edge,
                                              nhid_node=nhid_node, nhid_edge=nhid_edge,
                                              num_node=self.n)
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.p * self. q)
        # stdv = 0.1
        # stdv = 1.
        self.W_ve.data.uniform_(-stdv, stdv)
        self.W_ev.data.uniform_(-stdv, stdv)

    def forward(self, X_0, R, S, H, E0, F0, X0, phi, A_rho=1.0, fw_mitr=50, bw_mitr=50, A_orig=None):
        """Allow one to use a different A matrix for convolution operation in equilibrium equ"""
        self.counter += 1
        M = H @ H.T
        L = torch.linalg.eig(M).eigenvalues.real
        pf = torch.max(L).item()
        kappa = self.k/pf
        norm_XY = torch.norm(self.W_ev @ self.W_ve, float('inf'))
        if norm_XY > kappa:
            self.W_ev, self.W_ve = projection_NEC_DGT(self.W_ev, self.W_ve, kappa)

        O_3, O_3_logits, Ra_3, loss_EO_ER= self.B_encoder(R.T, S, H, F0, E0, X0)

        # b_Omega = b_Omega.T  # [De_o,n_node]
        # fixpoint_solution = ImplicitNECFunction_Lite.apply(self.W_ev, self.W_ve, X_0, H, O_3.T, phi, self.counter, fw_mitr, bw_mitr)
        # O_3 = O_3.T # + fixpoint_solution
        # Hv = Hv.T
        return O_3, O_3_logits, Ra_3, loss_EO_ER
        # return ImplicitNECFunction.apply(self.W_ev, self.W_ve, X_0, Rr, Rs, b_Omega, phi, fw_mitr, bw_mitr)


class Preprocessor_Context(nn.Module):
    def __init__(self, in_features_node, in_features_edge, nhid_node, nhid_edge, num_node, inter=2):
        super(Preprocessor_Context, self).__init__()
        self.Ds = in_features_node
        self.Dr = in_features_edge
        self.De_o = nhid_node
        self.De_r = nhid_edge
        self.n = num_node
        self.nr = num_node * num_node - num_node
        self.inter = inter
        self.hidden_size_1 = 50
        self.hidden_size_2 = 100
        self.Dx = 3

        self.phi_E_O_1 = nn.Sequential(
            nn.Linear(self.Ds * 2 + self.Dr, self.hidden_size_1),
            nn.ReLU(),
            nn.Linear(self.hidden_size_1, self.hidden_size_1),
            nn.ReLU(),
            nn.Linear(self.hidden_size_1, self.De_r)
        )

        self.phi_U_O_1 = nn.Sequential(
            nn.Linear(self.Ds + self.De_r + self.Dx, self.hidden_size_1),
            nn.ReLU(),
            nn.Linear(self.hidden_size_1, self.Ds),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )

        self.mlp_ee1 = nn.Sequential(
            nn.Linear(self.hidden_size_2, self.De_r),
            # nn.ReLU(),
            # nn.Linear(self.hidden_size_2, self.De_r)
        )

        self.phi_U_R_1 = nn.Sequential(
            nn.Linear(self.De_r + self.Dr, self.hidden_size_2),
            nn.ReLU(),
            nn.Linear(self.hidden_size_2, self.Dr),
            nn.ReLU(),
        )

        self.phi_E_O_2 = nn.Sequential(
            nn.Linear(self.Ds * 2 + self.Dr, self.hidden_size_1),
            nn.ReLU(),
            nn.Linear(self.hidden_size_1, self.hidden_size_1),
            nn.ReLU(),
            nn.Linear(self.hidden_size_1, self.De_r)
        )

        self.phi_U_O_2 = nn.Sequential(
            nn.Linear(self.De_r + self.Ds + self.Dx, self.hidden_size_1),
            nn.ReLU(),
            nn.Linear(self.hidden_size_1, self.Ds)
        )

        self.vv = nn.Sequential(
            nn.Linear(20, self.Ds)
        )

        self.mlp_ee3 = nn.Sequential(
            nn.Linear(self.hidden_size_2, self.De_r),
            # nn.ReLU(),
            # nn.Linear(self.hidden_size_2, self.De_r)
        )

        self.phi_U_R_2 = nn.Sequential(
            nn.Linear(self.De_r + self.Dr, self.hidden_size_2),
            nn.ReLU(),
            nn.Linear(self.hidden_size_2, self.Dr),
            nn.ReLU()
        )

        self.w1_1 = Parameter(torch.empty((self.hidden_size_2, self.Ds)))
        self.w1_2 = Parameter(torch.empty((self.hidden_size_2, self.Dr)))
        nn.init.trunc_normal_(self.w1_1, std=0.1)
        nn.init.trunc_normal_(self.w1_2, std=0.1)
        self.b1 = Parameter(torch.zeros(self.hidden_size_2))

        self.w2_1 = Parameter(torch.empty((self.hidden_size_2, self.Ds)))
        self.w2_2 = Parameter(torch.empty((self.hidden_size_2, self.Dr)))
        nn.init.trunc_normal_(self.w2_1, std=0.1)
        nn.init.trunc_normal_(self.w2_2, std=0.1)
        self.b2 = Parameter(torch.zeros(self.hidden_size_2))

        self.m1 = nn.Softmax(dim=1)

    def forward(self, Rr, Rs, H, O_1, Ra_1, X0):
        '''
        if adj is not self.adj:
            self.adj = adj
            self.adj_rho = get_spectral_rad(adj)
        '''
        self.adj_rho = 1

        # three layers and two MLP
        B1 = torch.cat((O_1.T @ Rr, O_1.T @ Rs, Ra_1.T))  # node to edge (m)
        E_O_1 = self.phi_E_O_1(B1.T)  # edge mlp (phi_E_O) influence-on-node TODO not permutation invariant?
        # C_O_1 = torch.cat((O_1, Rr @ E_O_1, X0), dim=1)  # node (a_O) # todo
        C_O_1 = torch.cat((O_1, H @ E_O_1, X0), dim=1)  # node (a_O)
        O_2 = self.phi_U_O_1(C_O_1)  # (phi_U_O)
        # updating the edge
        E_R_1 = F.relu(F.linear(B1.T, torch.cat((self.w1_1, self.w1_1, self.w1_2), dim=1), self.b1)) # phi_E_R
        E_R_1 = self.mlp_ee1(E_R_1)
        h2_trans_bar = E_R_1.T @ H.T  # todo
        # h2_trans_bar = E_R_1.T @ H.T
        E_R_1 = h2_trans_bar @ Rr + h2_trans_bar @ Rs
        C_R_1 = torch.cat((Ra_1, E_R_1.T), dim=1)  # (a_R)
        Ra_2 = self.phi_U_R_1(C_R_1)  # (phi_U_R)

        # three layers and two MLP
        B2 = torch.cat((O_2.T @ Rr, O_2.T @ Rs, Ra_2.T))  # node to edge
        E_O_2 = self.phi_E_O_2(B2.T)  # edge mlp edge mlp (phi_E_O)
        # C_O_2 = torch.cat((O_1, Rr @ E_O_2, X0), dim=1)  # node  (a_O) # todo
        C_O_2 = torch.cat((O_1, H @ E_O_2, X0), dim=1)
        O_3 = self.phi_U_O_2(C_O_2)  # (phi_U_O)
        # O_3_logits = self.m1(self.vv(O_3))
        O_3_logits = self.m1(O_3)
        # updating the edge
        E_R_2 = F.relu(F.linear(B2.T, torch.cat((self.w2_1, self.w2_1, self.w2_2), dim=1), self.b2))
        E_R_2 = self.mlp_ee3(E_R_2)
        h2_trans_bar2 = E_R_2.T @ H.T
        E_R_2 = h2_trans_bar2 @ Rr + h2_trans_bar2 @ Rs
        C_R_2 = torch.cat((Ra_1, E_R_2.T), dim=1)  # (a_R)
        Ra_3 = self.phi_U_R_2(C_R_2)  # (phi_U_R)

        norm_weight = 0.001
        loss_EO_ER = norm_weight * torch.sum(E_O_1**2/2) + norm_weight * torch.sum(E_O_2**2/2) + norm_weight * torch.sum(E_R_1**2/2) + norm_weight * torch.sum(E_R_2**2/2)

        return O_3, O_3_logits, Ra_3, loss_EO_ER



#
# class ImplicitNECGraph(Module):
#     """
#     A Implicit Graph Neural Network Layer (IGNN)
#     """
#
#     def __init__(self, in_features_node, in_features_edge, nhid_node, nhid_edge, num_node, kappa=0.99, b_direct=False):
#         super(ImplicitNECGraph, self).__init__()
#         self.p = nhid_node
#         self.q = nhid_edge
#         self.n = num_node
#         self.k = kappa      # if set kappa=0, projection will be disabled at forward feeding.
#         self.b_direct = b_direct
#
#         # self.W_ve = Parameter(torch.FloatTensor(self.De_r, 2 * self.De_o))
#         self.W_ve = Parameter(torch.FloatTensor(self.q, self.p))
#         self.W_ev = Parameter(torch.FloatTensor(self.p, self.q))
#         self.Omega_1 = Parameter(torch.FloatTensor(in_features_node, self.p))
#         self.Omega_2 = Parameter(torch.FloatTensor(in_features_edge, self.p))
#         # self.bias = Parameter(torch.FloatTensor(self.m, 1))
#         self.init()
#
#     def init(self):
#         stdv = 1. / math.sqrt(self.p * self. q)
#         # stdv = 0.1
#         # stdv = 1.
#         self.W_ve.data.uniform_(-stdv, stdv)
#         self.W_ev.data.uniform_(-stdv, stdv)
#         self.Omega_1.data.uniform_(-stdv, stdv)
#         self.Omega_2.data.uniform_(-stdv, stdv)
#
#     def forward(self, X_0, R,  A, E0, F0, phi, A_rho=1.0, fw_mitr=300, bw_mitr=300, A_orig=None):
#         """Allow one to use a different A matrix for convolution operation in equilibrium equ"""
#         # if self.k is not None:  # when self.k = 0, A_rho is not required
#         self.W_ev, self.W_ve = projection_NEC_DGT(self.W_ev, self.W_ve)
#         support_1 = torch.spmm(F0, self.Omega_1)
#         support_1 = torch.spmm(A, support_1).T
#         support_2 = torch.spmm(R, E0)
#         support_2 = torch.spmm(support_2, self.Omega_2).T
#         b_Omega = support_1 + support_2
#         b_Omega = F.relu(b_Omega)
#         # b_Omega = b_Omega.T  # [De_o,n_node]
#         return ImplicitNECFunction_Lite.apply(self.W_ev, self.W_ve, X_0, R,  b_Omega, phi, fw_mitr, bw_mitr)
#         # return ImplicitNECFunction.apply(self.W_ev, self.W_ve, X_0, Rr, Rs, b_Omega, phi, fw_mitr, bw_mitr)


#
# class NECDGT_encoder(nn.Module):
#     def __init__(self, in_features_node, in_features_edge, nhid_node, nhid_edge, num_node, inter=5):
#         super(NECDGT_encoder, self).__init__()
#         self.in_features_node = in_features_node
#         self.in_features_edge = in_features_edge
#         self.p = nhid_node
#         self.q = nhid_edge
#         self.n = num_node
#         self.inter = inter
#         self.hidden_size_1 = 50
#         self.hidden_size_2 = 50
#
#         self.mlp_ve1 = nn.Sequential(
#             nn.Linear(in_features_node*2+in_features_edge, self.hidden_size_1),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size_1, self.hidden_size_1),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size_1, self.q)
#         )
#
#         self.mlp_vv1 = nn.Sequential(
#             nn.Linear(self.q+self.in_features_node, self.hidden_size_1),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size_1, self.inter), # TODO why Ds_inter=5 used only for b here?
#             nn.ReLU()
#         )
#
#         self.mlp_ee1 = nn.Sequential(
#             nn.Linear(self.hidden_size_2, self.hidden_size_2),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size_2, self.q)
#         )
#
#         self.mlp_ee2 = nn.Sequential(
#             nn.Linear(self.q + self.in_features_edge, self.hidden_size_2),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size_2, self.inter),
#             nn.ReLU(),
#         )
#
#         self.mlp_ve2 = nn.Sequential(
#             nn.Linear(inter*3, self.hidden_size_1),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size_1, self.hidden_size_1),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size_1, self.q)
#         )
#
#         self.mlp_vv2 = nn.Sequential(
#             nn.Linear(self.q+self.in_features_node, self.hidden_size_1),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size_1, 64),
#             nn.ReLU()
#         )
#
#         self.mlp_ee3 = nn.Sequential(
#             nn.Linear(self.hidden_size_2, self.hidden_size_2),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size_2, self.q)
#         )
#
#         self.mlp_ee4 = nn.Sequential(
#             nn.Linear(self.q + self.in_features_edge, self.hidden_size_2),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size_2, self.in_features_edge)
#         )
#         self.m1 = nn.Softmax(dim=1)
#
#         self.w1_1 = Parameter(torch.empty((self.hidden_size_2, in_features_node)))
#         self.w1_2 = Parameter(torch.empty((self.hidden_size_2, in_features_edge)))
#         nn.init.trunc_normal_(self.w1_1, std=0.1)
#         nn.init.trunc_normal_(self.w1_2, std=0.1)
#         self.b1 = Parameter(torch.zeros(self.hidden_size_2))
#
#         self.w2_1 = Parameter(torch.empty((self.hidden_size_2, inter)))
#         self.w2_2 = Parameter(torch.empty((self.hidden_size_2, inter)))
#         nn.init.trunc_normal_(self.w2_1, std=0.1)
#         nn.init.trunc_normal_(self.w2_2, std=0.1)
#         self.b2 = Parameter(torch.zeros(self.hidden_size_2))
#
#     def forward(self, R, S, node_data, Ra_data):
#         '''
#         if adj is not self.adj:
#             self.adj = adj
#             self.adj_rho = get_spectral_rad(adj)
#         '''
#         self.adj_rho = 1
#
#         # three layers and two MLP
#         B1 = torch.cat((torch.matmul(node_data.permute((0, 2, 1)), R),
#                         torch.matmul(node_data.permute((0, 2, 1)), S),
#                         Ra_data.permute((0, 2, 1))), dim=1)  # node to edge (m)
#         EO1 = self.mlp_ve1(B1.permute((0, 2, 1)))  # edge mlp (phi_E_O)
#         Hev = torch.matmul(R, EO1) #+ torch.matmul(Rs, EO1)  # edge to node (Originally Rs is not used, shape=n_o*De_r)
#         Hv = torch.cat((Hev, node_data), dim=-1)  # node (a_O)
#         Hv2 = self.mlp_vv1(Hv)  # (phi_U_O)
#         # updating the edge
#         He = F.relu(F.linear(B1.permute((0, 2, 1)),
#                              torch.cat((self.w1_1, self.w1_1, self.w1_2), dim=1), self.b1)) # d_o, d_o, d_r
#         ER1 = self.mlp_ee1(He)  # (phi_E_R) TODO no edge updating?
#         He = torch.cat((ER1, Ra_data), dim=-1)  # (a_R)
#         He2 = self.mlp_ee2(He)  # (phi_U_R) => Ra_2
#
#         # three layers and two MLP
#         B2 = torch.cat((torch.matmul(Hv2.permute((0, 2, 1)), R),
#                         torch.matmul(Hv2.permute((0, 2, 1)), S),
#                         He2.permute((0, 2, 1))), dim=1)  # node to edge (m)
#         EO2 = self.mlp_ve2(B2.permute((0, 2, 1)))  # edge mlp edge mlp (phi_E_O)
#         Hev = torch.matmul(R, EO2) #+ torch.matmul(Rs, EO2)  # edge to node
#         Hv = torch.cat((Hev, node_data), dim=-1)  # node  (a_O)
#         Hv3 = self.mlp_vv2(Hv)  # (phi_U_O)
#         # updating the edge
#         He = F.relu(F.linear((B2.permute(0, 2, 1)),
#                              torch.cat((self.w2_1, self.w2_1, self.w2_2), dim=1), self.b2)) # d_o, d_o, d_r
#         ER2 = self.mlp_ee3(He)  # (phi_E_R) TODO no edge updating?
#         He = torch.cat((ER2, Ra_data), dim=-1)  # (a_R)
#         He_logits3 = self.mlp_ee4(He)  # (phi_U_R) => Ra_2
#         He3 = self.m1(He_logits3)
#
#         return Hv3, He3, He_logits3
