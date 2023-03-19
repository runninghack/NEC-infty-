import torch
import torch.nn as nn
import torch.nn.functional as F
# from layers import INECN_Layer, INECN_Layer_Context
from torch.nn import Parameter
import numpy as np
from graphclassification.datasets import mtx_to_batch, tnsr_to_batch
from torch.nn import Parameter
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import projection_norm_inf, projection_NEC_DGT, SparseDropout
from functions import ImplicitFunction, ImplicitNECFunction, NECFunction, ImplicitNECFunction_Lite
import pickle



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
        # self.B_encoder = NECDGT_encoder(Ds=1, Dr=2, nhid_node=20, nhid_edge=20, num_node=20)
        # self.bias = Parameter(torch.FloatTensor(self.m, 1))
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.p * self. q)
        # stdv = 0.1
        # stdv = 1.
        self.W_ve.data.uniform_(-stdv, stdv)
        self.W_ev.data.uniform_(-stdv, stdv)

    def forward(self, X_0, H, E0, F0, phi, A_rho=1.0, fw_mitr=50, bw_mitr=50, A_orig=None):
        """Allow one to use a different A matrix for convolution operation in equilibrium equ"""
        self.counter += 1
        M = H @ H.T
        L = torch.linalg.eig(M).eigenvalues.real
        pf = torch.max(L).item()
        kappa = self.k/pf
        norm_XY = torch.norm(self.W_ev @ self.W_ve, float('inf'))
        if norm_XY > kappa:
            self.W_ev, self.W_ve = projection_NEC_DGT(self.W_ev, self.W_ve, kappa)

        temp = E0 @ self.W_ve
        F1= H @ temp + F0.T

        Ft = ImplicitNECFunction_Lite.apply(self.W_ev, self.W_ve, X_0, H, F1.T, phi, self.counter, fw_mitr, bw_mitr)

        Et = H.T @ Ft.T @ self.W_ev
        return Ft, Et


class Data_Generator(nn.Module):
    def __init__(self, in_features_node, in_features_edge, nhid_node, nhid_edge, num_node, dropout, kappa=0.9):
        super(Data_Generator, self).__init__()
        self.adj_rho = None

        self.ig1 = INECN_Layer(in_features_node, in_features_edge, nhid_node, nhid_edge, num_node, kappa)
        self.dropout = dropout
        self.X_0 = None
        # self.V_0 = nn.Linear(nhid_node, nhid_node)
        # self.V_1 = nn.Linear(nhid_node, in_features_node)

    def forward(self, R, S, H, node_data, Ra_data):
        '''
        if adj is not self.adj:
            self.adj = adj
            self.adj_rho = get_spectral_rad(adj)
        '''
        self.adj_rho = 1

        #three layers and two MLP
        _f = lambda _X: _X
        # dv, de = self.ig1(self.X_0, H, Ra_data, node_data.T, F.relu, self.adj_rho) # todo relu?
        dv, de = self.ig1(self.X_0, H, Ra_data, node_data.T, _f, self.adj_rho)  # todo relu?
        # x = dv.T
        # x = F.relu(self.V_0(x))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.V_1(x)
        return dv, de

def read_er_data(No, Nr):
    a = np.ones((No, No)) - np.eye(No)
    x = np.load('ER/erdos{}_x.npy'.format(No))
    y = np.load('ER/erdos{}_y.npy'.format(No))

    data = np.zeros((x.shape[0], x.shape[1], x.shape[1], 2))
    for i in range(x.shape[0]):
        data[i, :, :, 0] = x[i, :, :] * a  # diagnal is removed
        data[i, :, :, 1] = y[i, :, :] * a

    H = np.zeros((500, No, Nr), dtype=float)
    R = np.zeros((500, No, Nr), dtype=float)
    S = np.zeros((500, No, Nr), dtype=float)

    A = data[:, :, :, 0]
    E0 = np.zeros((500, 2, Nr), dtype=float)
    Et = np.zeros((500, 2, Nr), dtype=float)

    cnt = 0
    for i in range(No):
        for j in range(No):
            if (i != j):
                for k in range(data.shape[0]):
                    if data[k, i, j, 0] > 0:
                        H[k, i, cnt] = 1.0
                        H[k, j, cnt] = 1.0
                        R[k, i, cnt] = 1.0
                        S[k, j, cnt] = 1.0
                        E0[k, int(data[k, i, j, 0]), cnt] = 1
                        Et[k, int(data[k, i, j, 1]), cnt] = 1
                cnt += 1

    split = 500
    R = torch.from_numpy(R).to(torch.float32)[:split] # .to_sparse()
    S = torch.from_numpy(S).to(torch.float32)[:split] # .to_sparse()
    H = torch.from_numpy(H).to(torch.float32)[:split]  # .to_sparse()
    A = torch.from_numpy(A).to(torch.float32)[:split] # .to_sparse()
    return E0, Et, R, S, H, A


E0, Et, R, S, H, A = read_er_data(20, 380)
F0 = torch.rand(500, 20, 1).to(torch.float32)
# E0 = torch.rand(500, 380, 2).to(torch.float32)
ids = np.arange(F0.shape[0])
np.random.shuffle(ids)

E0 = torch.from_numpy(E0).permute(0, 2, 1).to(torch.float32)
# Et = torch.from_numpy(Et).permute(0, 2, 1).to(torch.float32)

print("initialize")

model = Data_Generator(in_features_node=1, in_features_edge=2, nhid_node=1, nhid_edge=2, num_node=20, dropout=0.00,
                       kappa=0.99)

print("initialize finished")

# Ft, Et = model(mtx_to_batch(R[:50]).T, mtx_to_batch(S[:50]), mtx_to_batch(H[:50]),
#                               tnsr_to_batch(F0[:50]), tnsr_to_batch(E0[:50]))

Ft, Et = model(mtx_to_batch(R).T, mtx_to_batch(S), mtx_to_batch(H),
                              tnsr_to_batch(F0), tnsr_to_batch(E0))

path = 'dynamical_system2.pt'
torch.save({
            'F0': F0,
            'Ft': Ft.reshape(500, 20, 1),
            'E0': E0,
            'Et': Et.reshape(500, 380, 2),
            'R': R,
            'S': S,
            'H': H,
            'A': A,
            }, path)

print("finished")
