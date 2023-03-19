import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class NECDGT(nn.Module):
    def __init__(self, in_features_node, in_features_edge, nhid_node, nhid_edge, num_node, dropout, inter=5, kappa=0.9):
        super(NECDGT, self).__init__()
        self.in_features_node = in_features_node
        self.in_features_edge = in_features_edge
        self.p = nhid_node
        self.q = nhid_edge
        self.n = num_node
        self.inter = inter

        self.mlp_ve1 = nn.Sequential(
            nn.Linear(in_features_node*2+in_features_edge, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, self.q)
        )

        self.mlp_vv1 = nn.Sequential(
            nn.Linear(self.q+self.in_features_node, 50),
            nn.ReLU(),
            nn.Linear(50, self.inter),
            nn.ReLU()
        )

        # self.c_mlp_1 = nn.Sequential(
        #     nn.Linear(Ds * 2 + Dr, self.De_r),
        #     nn.ReLU(),
        #     nn.Linear(self.De_r, self.De_r)
        # )

        self.mlp_ee1 = nn.Sequential(
            nn.Linear(self.q + self.in_features_edge, 100),
            nn.ReLU(),
            nn.Linear(100, self.inter)
        )

        self.m1 = nn.Softmax(dim=1)

        self.mlp_ve2 = nn.Sequential(
            nn.Linear(inter * 3, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.Linear(50, self.q)
        )

        self.mlp_vv2 = nn.Sequential(
            nn.Linear(self.q + self.in_features_node, self.p),
            nn.ReLU(),
            nn.Linear(self.p, self.in_features_node),
            nn.ReLU()
        )

        self.mlp_ee2 = nn.Sequential(
            nn.Linear(self.q + self.in_features_edge, 100),
            nn.ReLU(),
            nn.Linear(100, self.in_features_edge)
        )
        self.m2 = nn.Softmax(dim=1)

        self.w1_1 = Parameter(torch.empty((self.q, in_features_node)))
        self.w1_2 = Parameter(torch.empty((self.q, in_features_edge)))
        nn.init.trunc_normal_(self.w1_1, std=0.1)
        nn.init.trunc_normal_(self.w1_2, std=0.1)
        self.b1 = Parameter(torch.zeros(self.q))

        self.w2_1 = Parameter(torch.empty((self.q, self.inter)))
        self.w2_2 = Parameter(torch.empty((self.q, self.inter)))
        nn.init.trunc_normal_(self.w2_1, std=0.1)
        nn.init.trunc_normal_(self.w2_2, std=0.1)
        self.b2 = Parameter(torch.zeros(self.q))

    def forward(self, R, S, adj, node_data, Ra_data):
        '''
        if adj is not self.adj:
            self.adj = adj
            self.adj_rho = get_spectral_rad(adj)
        '''
        self.adj_rho = 1

        # three layers and two MLP
        B1 = torch.cat((node_data.T @ R, node_data.T @ S, Ra_data.T))  # node to edge (m)
        He = self.mlp_ve1(B1.T)  # edge mlp (phi_E_O) influence-on-node TODO not permutation invariant?
        Hv = R @ He + S @ He  # edge to node
        Hv = torch.cat((Hv, node_data), dim=1)  # node (a_O)
        Hv2 = self.mlp_vv1(Hv)  # (phi_U_O) # TODO inter=5?
        # updating the edge
        # He = self.c_mlp_1(B1.T)  # (phi_E_R) influence-on-edge TODO
        He = F.linear(B1.T, torch.cat((self.w1_1, self.w1_2, self.w1_1), dim=1), self.b1)
        He = torch.cat((He, Ra_data), dim=1)  # (a_R)
        He2 = self.mlp_ee1(He)  # (phi_U_R) TODO no edge updating?

        # H_map = Rr @ self.map(He2)

        # three layers and two MLP
        B2 = torch.cat((Hv2.T @ R, Hv2.T @ S, He2.T))  # node to edge
        He = self.mlp_ve2(B2.T)  # edge mlp edge mlp (phi_E_O)
        Hv = R @ He + S @ He  # edge to node
        Hv = torch.cat((Hv, node_data), dim=1)  # node  (a_O)
        Hv3 = self.mlp_vv2(Hv)  # (phi_U_O)
        # updating the edge
        # He = self.c_mlp_2(B1.T)  # (phi_E_R)
        He = F.linear(B2.T, torch.cat((self.w2_1, self.w2_2, self.w2_1), dim=1), self.b2)
        He = torch.cat((He, Ra_data), dim=1)  # (a_R)
        He_logits3 = self.mlp_ee2(He)  # (phi_U_R)
        He3 = self.m1(He_logits3)

        return Hv2, Hv3, He3, He_logits3 # , H_map


class NECDGT_batching(nn.Module):
    def __init__(self, in_features_node, in_features_edge, nhid_node, nhid_edge, num_node, inter=5):
        super(NECDGT_batching, self).__init__()
        self.in_features_node = in_features_node
        self.in_features_edge = in_features_edge
        self.p = nhid_node
        self.q = nhid_edge
        self.n = num_node
        self.inter = inter
        self.hidden_size_1 = 50
        self.hidden_size_2 = 50

        self.mlp_ve1 = nn.Sequential(
            nn.Linear(in_features_node*2+in_features_edge, self.hidden_size_1),
            nn.ReLU(),
            nn.Linear(self.hidden_size_1, self.hidden_size_1),
            nn.ReLU(),
            nn.Linear(self.hidden_size_1, self.q)
        )

        self.mlp_vv1 = nn.Sequential(
            nn.Linear(self.q+self.in_features_node, self.hidden_size_1),
            nn.ReLU(),
            nn.Linear(self.hidden_size_1, self.inter), # TODO why Ds_inter=5 used only for b here?
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
            nn.Linear(inter*3, self.hidden_size_1),
            nn.ReLU(),
            nn.Linear(self.hidden_size_1, self.hidden_size_1),
            nn.ReLU(),
            nn.Linear(self.hidden_size_1, self.q)
        )

        self.mlp_vv2 = nn.Sequential(
            nn.Linear(self.q+self.in_features_node, self.hidden_size_1),
            nn.ReLU(),
            nn.Linear(self.hidden_size_1, self.in_features_node),
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
        B1 = torch.cat((torch.matmul(node_data.permute((0, 2, 1)), R),
                        torch.matmul(node_data.permute((0, 2, 1)), S),
                        Ra_data.permute((0, 2, 1))), dim=1)  # node to edge (m)
        EO1 = self.mlp_ve1(B1.permute((0, 2, 1)))  # edge mlp (phi_E_O)
        Hev = torch.matmul(R, EO1) #+ torch.matmul(Rs, EO1)  # edge to node (Originally Rs is not used, shape=n_o*De_r)
        Hv = torch.cat((Hev, node_data), dim=-1)  # node (a_O)
        Hv2 = self.mlp_vv1(Hv)  # (phi_U_O)
        # updating the edge
        He = F.relu(F.linear(B1.permute((0, 2, 1)),
                             torch.cat((self.w1_1, self.w1_1, self.w1_2), dim=1), self.b1)) # d_o, d_o, d_r
        ER1 = self.mlp_ee1(He)  # (phi_E_R) TODO no edge updating?
        He = torch.cat((ER1, Ra_data), dim=-1)  # (a_R)
        He2 = self.mlp_ee2(He)  # (phi_U_R) => Ra_2

        # three layers and two MLP
        B2 = torch.cat((torch.matmul(Hv2.permute((0, 2, 1)), R),
                        torch.matmul(Hv2.permute((0, 2, 1)), S),
                        He2.permute((0, 2, 1))), dim=1)  # node to edge (m)
        EO2 = self.mlp_ve2(B2.permute((0, 2, 1)))  # edge mlp edge mlp (phi_E_O)
        Hev = torch.matmul(R, EO2) #+ torch.matmul(Rs, EO2)  # edge to node
        Hv = torch.cat((Hev, node_data), dim=-1)  # node  (a_O)
        Hv3 = self.mlp_vv2(Hv)  # (phi_U_O)
        # updating the edge
        He = F.relu(F.linear((B2.permute(0, 2, 1)),
                             torch.cat((self.w2_1, self.w2_1, self.w2_2), dim=1), self.b2)) # d_o, d_o, d_r
        ER2 = self.mlp_ee3(He)  # (phi_E_R) TODO no edge updating?
        He = torch.cat((ER2, Ra_data), dim=-1)  # (a_R)
        He_logits3 = self.mlp_ee4(He)  # (phi_U_R) => Ra_2
        He3 = self.m1(He_logits3)

        return Hv3, He3, He_logits3