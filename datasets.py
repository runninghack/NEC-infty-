import torch
import torch.utils.data as utils
from graphclassification.utils import read_er_data
import numpy as np

def tnsr_to_batch(tnsr):
    return tnsr.reshape(-1, tnsr.shape[-1])

def mtx_to_batch(mtx):
    """ batch*node*edge -> batch_node*batch_edge"""
    return torch.block_diag(*[mtx[i] for i in range(mtx.shape[0])])

# class graphDataset(utils.Dataset):
#     def __init__(self, F0, Ft, E0, Et, X0, R, S, H):
#         self.F0, self.Ft, self.E0, self.Et, self.X0, self.R, self.S, self.H = \
#             F0, Ft, E0, Et, X0, R, S, H
#
#     def __len__(self):
#         return self.F0.shape[0]
#
#     def __getitem__(self, idx):
#
#         return self.F0[idx], self.Ft[idx], \
#                self.E0[idx], self.Et[idx], self.X0[idx],\
#                self.R[idx], self.S[idx], \
#                self.H[idx]
#         # self.A[idx]


class graphDataset(utils.Dataset):
    def __init__(self, F0, Ft, E0, Et,  R, S, H):
        self.F0, self.Ft, self.E0, self.Et,  self.R, self.S, self.H = \
            F0, Ft, E0, Et, R, S, H

    def __len__(self):
        return self.F0.shape[0]

    def __getitem__(self, idx):

        return self.F0[idx], self.Ft[idx], \
               self.E0[idx], self.Et[idx], \
               self.R[idx], self.S[idx], \
               self.H[idx]
        # self.A[idx]


def PrepareDataset(train_num, val_num, tst_num, BATCH_SIZE=50):

    F0, Ft, E0, Et, R, S, H, A = read_er_data(20, 380)
    F0 = torch.from_numpy(F0).permute(0, 2, 1).to(torch.float32)
    Ft = torch.from_numpy(Ft).permute(0, 2, 1).to(torch.float32)
    E0 = torch.from_numpy(E0).permute(0, 2, 1).to(torch.float32)
    Et = torch.from_numpy(Et).permute(0, 2, 1).to(torch.float32)
    ids = np.arange(F0.shape[0])
    np.random.shuffle(ids)

    ids_train, ids_val, ids_test = ids[:train_num], ids[train_num:train_num+val_num], ids[train_num+val_num:train_num+val_num+tst_num]

    train_dataset = graphDataset(F0[ids_train], Ft[ids_train], E0[ids_train], Et[ids_train], R[ids_train], S[ids_train], H[ids_train])
    valid_dataset = graphDataset(F0[ids_val], Ft[ids_val], E0[ids_val], Et[ids_val], R[ids_val], S[ids_val], H[ids_val], )
    test_dataset = graphDataset(F0[ids_test], Ft[ids_test], E0[ids_test], Et[ids_test], R[ids_test], S[ids_test], H[ids_test] )

    train_dataloader = utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    return train_dataloader, valid_dataloader, test_dataloader