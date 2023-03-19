import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import os
import sys
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__),
                  os.pardir)
)

print(sys.path)
sys.path.append(PROJECT_ROOT)
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from graphclassification.models import INECN
from graphclassification.utils import read_er_data, r2_loss
from graphclassification.datasets import graphDataset, tnsr_to_batch, mtx_to_batch
from scipy.stats.stats import pearsonr, spearmanr


class IGNN_NECDGTModel(pl.LightningModule):

    def __init__(self, seed, lr, optimizer, in_features_node, in_features_edge, nhid_node, nhid_edge, num_node, ne_ratio):
        super(IGNN_NECDGTModel, self).__init__()
        pl.seed_everything(seed)
        self.lr = lr
        self.optimizer = optimizer
        self.train_loss = 0
        self.val_loss = 0
        self.ne_ratio = ne_ratio
        self.model = INECN(in_features_node=in_features_node, in_features_edge=in_features_edge, nhid_node=nhid_node,
                           nhid_edge=nhid_edge, num_node=num_node, dropout=0.01, kappa=0.99)
        self.model.to(device)  # moving the model to cuda

    def training_step(self, batch, batch_nb):
        F0, Ft, E0, Et, R, S, H = batch
        node_data, Ra_data = tnsr_to_batch(F0), tnsr_to_batch(E0)
        F_hat, H_e, He_logits = self.model(mtx_to_batch(R).T, mtx_to_batch(S), mtx_to_batch(H), node_data, Ra_data)
        loss_node_mse = F.mse_loss(F_hat, tnsr_to_batch(Ft))
        loss_edge_mse = F.binary_cross_entropy_with_logits(He_logits, tnsr_to_batch(Et))
        loss = loss_node_mse + self.ne_ratio * loss_edge_mse
        self.train_loss += loss.item()
        self.log("train performance", {"node mse": loss_node_mse, "total mse": loss}, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('loss', loss_node_mse)
        # return loss
        return loss

    def validation_step(self, batch, batch_nb):
        F0, Ft, E0, Et, R, S, H = batch
        node_data, Ra_data = tnsr_to_batch(F0), tnsr_to_batch(E0)
        F_hat, H_e, He_logits = self.model(mtx_to_batch(R).T, mtx_to_batch(S), mtx_to_batch(H), node_data, Ra_data)
        loss_node_mse = F.mse_loss(F_hat, tnsr_to_batch(Ft))
        loss_edge_mse = F.binary_cross_entropy_with_logits(He_logits, tnsr_to_batch(Et))
        loss = 1 * loss_node_mse + 10 * loss_edge_mse
        self.val_loss += loss.item()
        # self.val_loss += loss.item()
        # self.log("validation performance", {"val node mse": loss_node_mse}, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss_node_mse)
        # self.log("val_loss", on_epoch=True)
        return loss

    def test_step(self, batch, batch_nb):
        F0, Ft, E0, Et, R, S, H = batch
        node_data, Ra_data = tnsr_to_batch(F0), tnsr_to_batch(E0)
        F_hat, H_e, He_logits = self.model(mtx_to_batch(R).T, mtx_to_batch(S), mtx_to_batch(H), node_data, Ra_data)
        node_mse = F.mse_loss(F_hat, tnsr_to_batch(Ft))
        edge_mse = F.binary_cross_entropy_with_logits(He_logits, tnsr_to_batch(Et))
        node_r2 = r2_loss(F_hat, tnsr_to_batch(Ft))
        node_p = pearsonr(F_hat.detach().cpu().numpy().ravel(), tnsr_to_batch(Ft).detach().cpu().numpy().ravel())[0]
        node_sp = spearmanr(F_hat.detach().cpu().numpy().ravel(), tnsr_to_batch(Ft).detach().cpu().numpy().ravel())[0]
        _, edge_predictions = torch.max(H_e, 1)

        a = edge_predictions.cpu().numpy().ravel()
        b = tnsr_to_batch(Et)[:, 1].int().cpu().numpy().ravel()
        correct = (a == b)
        edge_accuracy = correct.sum() / correct.size

        metrics = {"test node mse": node_mse, "test node_r2": node_r2, "test node p": node_p,
                   "test node sp": node_sp, "test edge mse": edge_mse, "test edge acc": edge_accuracy}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return metrics

    def configure_optimizers(self):
        # Choose an optimizer and set up a learning rate according to hyperparameters
        if self.optimizer == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.0)
        elif self.optimizer == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=11, help='Random seed.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--bs', type=int, default=25, help='Batch size')
    parser.add_argument('--in_features_node', type=int, default=1, help='Node dimension')
    parser.add_argument('--in_features_edge', type=int, default=2, help='Edge dimension')
    parser.add_argument('--nhid_node', type=int, default=64, help='Node hidden dimension')
    parser.add_argument('--nhid_edge', type=int, default=128, help='Edge hidden dimension')
    parser.add_argument('--num_node', type=int, default=20, help='Number of nodes')
    parser.add_argument('--num_edges', type=int, default=380, help='Number of edges')
    parser.add_argument('--ne_ratio', type=float, default=1.0, help='node loss and edge loss ratio')
    parser.add_argument('--log_path', type=str, default="logs/ER20/", help='directory to save the logs')

    args = parser.parse_args()
    device = 'cuda'
    train_num, val_num, tst_num = 250, 200, 50

    F0, Ft, E0, Et, R, S, H, A = read_er_data(args.num_node, args.num_edges)

    F0 = torch.from_numpy(F0).permute(0, 2, 1).to(torch.float32)
    Ft = torch.from_numpy(Ft).permute(0, 2, 1).to(torch.float32)
    E0 = torch.from_numpy(E0).permute(0, 2, 1).to(torch.float32)
    Et = torch.from_numpy(Et).permute(0, 2, 1).to(torch.float32)
    ids = np.arange(F0.shape[0])
    np.random.shuffle(ids)

    ids_train, ids_val, ids_test = ids[:train_num], ids[train_num:train_num + val_num], \
        ids[train_num + val_num:train_num + val_num + tst_num]

    optimizer = 'Adam'
    train_ds = graphDataset(F0[ids_train], Ft[ids_train], E0[ids_train], Et[ids_train], R[ids_train], S[ids_train],
                            H[ids_train])
    train_loader = DataLoader(train_ds, batch_size=args.bs, num_workers=0, drop_last=True)
    val_ds = graphDataset(F0[ids_val], Ft[ids_val], E0[ids_val], Et[ids_val], R[ids_val], S[ids_val], H[ids_val])
    val_loader = DataLoader(val_ds, batch_size=args.bs, num_workers=0, drop_last=True)
    ts_ds = graphDataset(F0[ids_test], Ft[ids_test], E0[ids_test], Et[ids_test], R[ids_test], S[ids_test], H[ids_test])
    ts_loader = DataLoader(ts_ds, batch_size=args.bs, num_workers=0, drop_last=True)

    model = IGNN_NECDGTModel(args.seed, args.lr, optimizer, args.in_features_node, args.in_features_edge,
                             args.nhid_node, args.nhid_edge, args.num_node, args.ne_ratio)

    tb_logger = pl_loggers.CSVLogger(save_dir=args.log_path)
    checkpoint_callback = ModelCheckpoint(dirpath=args.log_path, save_top_k=5, monitor="val_loss")
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="max")
    trainer = pl.Trainer(
        logger=tb_logger,
        gpus=1,
        max_epochs=args.epochs,
        progress_bar_refresh_rate=1,
        log_every_n_steps=5,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback]
        # callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # training the model
    trainer.test(ckpt_path="best", dataloaders=val_loader)
    print("finished")
