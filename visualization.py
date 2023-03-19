import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import os
import sys

from pytorch_lightning.callbacks import ModelCheckpoint

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__),
                  os.pardir)
)

print(sys.path)
sys.path.append(PROJECT_ROOT)
import numpy as np
import pytorch_lightning as pl
# from aim.pytorch_lightning import AimLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from graphclassification.models import INECN
from graphclassification.utils import read_ba_data, r2_loss
from graphclassification.datasets import graphDataset, tnsr_to_batch, mtx_to_batch

###

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
###


train_num, val_num, tst_num = 250, 200, 50
np.random.seed(2)

F0, Ft, E0, Et, R, S, H, A = read_ba_data(20, 380)
F0 = torch.from_numpy(F0).permute(0, 2, 1).to(torch.float32)
Ft = torch.from_numpy(Ft).permute(0, 2, 1).to(torch.float32)
E0 = torch.from_numpy(E0).permute(0, 2, 1).to(torch.float32)
Et = torch.from_numpy(Et).permute(0, 2, 1).to(torch.float32)
ids = np.arange(F0.shape[0])
np.random.shuffle(ids)
pl.seed_everything(11)
ids_train, ids_val, ids_test = ids[:train_num], ids[train_num:train_num + val_num], \
                               ids[train_num + val_num:train_num + val_num + tst_num]

train_ds = graphDataset(F0[ids_train], Ft[ids_train], E0[ids_train], Et[ids_train], R[ids_train], S[ids_train], H[ids_train])
train_loader = DataLoader(train_ds, batch_size=50, num_workers=0, drop_last=True)

path = 'DS_BA/epoch=9-step=100.ckpt'
# path = ''
model = INECN(in_features_node=1, in_features_edge=2, nhid_node=64, nhid_edge=128, num_node=20, dropout=0.01, kappa=0.99)
model.load_state_dict(torch.load(path), strict=False)


for step, (F0, Ft, E0, Et, R, S, H) in enumerate(train_loader):
    node_data, Ra_data = tnsr_to_batch(F0), tnsr_to_batch(E0)
    F_hat, H_e, He_logits = model(mtx_to_batch(R).T, mtx_to_batch(S), mtx_to_batch(H), node_data, Ra_data)
    node_mse = F.mse_loss(F_hat, tnsr_to_batch(Ft))
    x = F_hat.cpu().detach().numpy()
    y = tnsr_to_batch(Ft).cpu().detach().numpy()
    min_y = min(y)
    max_y = max(y)
    sns.set()
    f = plt.figure(figsize=(5, 3))
    # plt.plot([min_y-5, max_y-5], [min_y, max_y])
    plt.scatter(x, y)
    plt.xlabel('Node attributes')
    plt.ylabel('Node degrees')
    f.tight_layout()
    plt.show()
    break
