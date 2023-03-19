import torch
import numpy as np
import scipy.sparse as sp
from torch.autograd import Function
from utils import sparse_mx_to_torch_sparse_tensor
import torch.nn as nn
import torch.nn.functional as F
import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class ImplicitFunction(Function):
    @staticmethod
    def forward(ctx, W, X_0, A, B, phi, fd_mitr=300, bw_mitr=300):
        X_0 = B if X_0 is None else X_0
        X, err, status, D = ImplicitFunction.inn_pred(W, X_0, A, B, phi, mitr=fd_mitr, compute_dphi=True)
        ctx.save_for_backward(W, X, A, B, D, X_0, torch.tensor(bw_mitr))
        if status not in "converged":
            print("Iterations not converging!", err, status)
        return X

    @staticmethod
    def backward(ctx, *grad_outputs):
        W, X, A, B, D, X_0, bw_mitr = ctx.saved_tensors
        bw_mitr = bw_mitr.cpu().numpy()
        # print("len of grad_outputs is {}".format(len(grad_outputs)))
        grad_x = grad_outputs[0]

        dphi = lambda X: torch.mul(X, D)
        # print("D.shape is {}".format(D.shape))
        grad_z, err, status, _ = ImplicitFunction.inn_pred(W.T, X_0, A, grad_x, dphi, mitr=bw_mitr, trasposed_A=True)

        grad_W = grad_z @ torch.spmm(A, X.T)
        grad_B = grad_z

        # Might return gradient for A if needed
        return grad_W, None, torch.zeros_like(A), grad_B, None, None, None

    @staticmethod
    def inn_pred(W, X, A, B, phi, mitr=300, tol=3e-6, trasposed_A=False, compute_dphi=False):
        At = A if trasposed_A else torch.transpose(A, 0, 1)
        # B is [64, 2259]
        err = 0
        status = 'max itrs reached'
        for i in range(mitr):
            # WXA
            X_ = W @ X
            support = torch.spmm(At, X_.T).T
            # if trasposed_A:
            #     print("support + B shape is {}".format((support + B).shape))
            X_new = phi(support + B)
            err = torch.norm(X_new - X, np.inf)
            if err < tol:
                status = 'converged'
                break
            X = X_new

        dphi = None
        if compute_dphi:
            with torch.enable_grad():
                support = torch.spmm(At, (W @ X).T).T
                Z = support + B
                Z.requires_grad_(True)
                X_new = phi(Z)
                dphi = torch.autograd.grad(torch.sum(X_new), Z, only_inputs=True)[0]

        return X_new, err, status, dphi


class ImplicitNECFunction(Function):
    @staticmethod
    def forward(ctx, Wv, We, X_0, R, S, B, phi, fd_mitr=300, bw_mitr=300):
        # todo X_0 and B
        X_0 = B if X_0 is None else X_0
        X, err, status, D1, D2 = ImplicitNECFunction.inn_pred(Wv, We, X_0, R, S, B, phi, phi, mitr=fd_mitr, compute_dphi=True)
        ctx.save_for_backward(Wv, We, X, R, S, B, D1, D2, X_0, torch.tensor(bw_mitr))
        #if status not in "converged":
        #    print("Iterations not converging!", err, status)
        return X

    @staticmethod
    def backward(ctx, *grad_outputs):
        Wv, We, X, R, S, B, D1, D2, X_0, bw_mitr = ctx.saved_tensors
        bw_mitr = bw_mitr.cpu().numpy()
        grad_x = grad_outputs[0]

        dphi1 = lambda X: torch.mul(X, D1)
        dphi2 = lambda X: torch.mul(X, D2)
        grad_z, err, status, _, _ = ImplicitNECFunction.inn_pred(Wv, We, X, R, S, grad_x, dphi1, dphi2, mitr=bw_mitr,
                                                                 trasposed_A=True, debug=True)
        # print("Shape of grad_z: {}".format(grad_z.shape))
        # print("Shape of torch.cat((X @ Rr, X @ Rs)): {}".format(torch.cat((X @ Rr, X @ Rs)).shape))
        grad_We = grad_z @ torch.cat((X @ R, X @ S))

        grad_Wv = grad_z @ (We @ torch.cat((X @ R, X @ S)))
        grad_B = grad_z

        # Might return gradient for A if needed
        return grad_Wv, grad_We, None, torch.zeros_like(R), torch.zeros_like(S), grad_B, None, None, None


    @staticmethod
    def inn_pred(Wv, We, X, R, S, B, phi, phi2, mitr=300, tol=3e-6, trasposed_A=False, compute_dphi=False, debug=False):
        err = 0
        status = 'max itrs reached'
        for i in range(mitr):
            # WXA
            # support_He = W_ve @ torch.cat(torch.spmm(X, Rr), torch.spmm(X, Rs))
            support_He = We @ torch.cat((X @ R, X @ S))
            support_He = phi(support_He)
            support = Wv @ support_He
            support = support @ R.T # support = torch.spmm(support, Rr.T)
            X_new = phi2(support + B)

            err = torch.norm(X_new - X, np.inf)
            if err < tol:
                status = 'converged'
                break
            X = X_new

        dphi1 = None
        dphi2 = None
        if compute_dphi:
            with torch.enable_grad():
                # support_He = W_ve @ torch.cat(torch.spmm(X, Rr), torch.spmm(X, Rs))
                support_He = We @ torch.cat((X @ R, X @ S))
                Z1 = support_He

                Z1.requires_grad_(True)
                support_He = phi(Z1)
                dphi1 = torch.autograd.grad(torch.sum(support_He), Z1, only_inputs=True)[0]
                support = Wv @ support_He
                support = support @ R.T  # support = torch.spmm(support, Rr.T)
                Z2 = support + B

                Z2.requires_grad_(True)
                X_new = phi2(Z2)
                dphi2 = torch.autograd.grad(torch.sum(X_new), Z2, only_inputs=True)[0]

        return X_new, err, status, dphi1, dphi2


class ImplicitNECFunction_Lite(Function):
    @staticmethod
    def forward(ctx, W_ev, W_ve, X_0, R, B, phi, counter, fd_mitr=300, bw_mitr=300):
        # todo X_0 and B
        X_0 = B if X_0 is None else X_0
        X, err, status, D1, D2 = ImplicitNECFunction_Lite.inn_pred(W_ev, W_ve, X_0, R, B, F.relu, F.relu, mitr=fd_mitr)
        ctx.counter = counter
        ctx.save_for_backward(W_ev, W_ve, X, R, B, D1, D2, X_0, torch.tensor(bw_mitr))
        #if status not in "converged":
        #    print("Iterations not converging!", err, status)
        return X

    @staticmethod
    def backward(ctx, *grad_outputs):
        W_ev, W_ve, X, R, B, D1, D2, X_0, bw_mitr = ctx.saved_tensors
        bw_mitr = bw_mitr.cpu().numpy()
        grad_x = grad_outputs[0]  # 64, 20

        dphi1 = lambda _X: torch.mul(_X, D1)
        dphi2 = lambda _X: torch.mul(_X, D2)
        # print('D2.shape is {}'.format(D2.shape))
        grad_z, err, status = ImplicitNECFunction_Lite.dz_fixedpoint(W_ev, W_ve, X, R, grad_x, dphi1, dphi2, mitr=bw_mitr)
        # print("Shape of grad_z: {}".format(grad_z.shape))
        # print("Shape of X: {}".format(X.shape))
        # print("Shape of Rr: {}".format(Rr.shape))

        grad_W_ev = grad_z @ R @ (F.relu(W_ve @ X @ R)).T
        # print(grad_W_ev.shape)
        a = W_ve @ X @ R
        with torch.enable_grad():
            a.requires_grad_(True)
            arelu = F.relu(a)
            dphi = torch.autograd.grad(torch.sum(arelu), a, only_inputs=True)[0]  # (De_r, E)

        wdzr = W_ev.T @ grad_z @ R  # (De_r, E)

        part1 = torch.mul(dphi, wdzr)
        part2 = X @ R

        grad_W_ve = part1 @ part2.T
        grad_B = grad_z

        # Might return gradient for A if needed
        # print(grad_W_ev)
        # print(grad_W_ve)
        # if ctx.counter % 2:
        #     grad_W_ev = None
        # else:
        #     grad_W_ve = None
        grad_W_ve = None
        return grad_W_ev, grad_W_ve, None, torch.zeros_like(R), grad_B, None, None, None, None

    @staticmethod
    def inn_pred(W_ev, W_ve, X, R, B, phi1, phi2, mitr=300, tol=3e-6):
        err = 0
        status = 'max itrs reached'
        err_list = []
        for i in range(mitr):
            # WXA
            # support_He = W_ve @ torch.cat(torch.spmm(X, Rr), torch.spmm(X, Rs))
            temp = W_ve @ X
            support_He1 = temp @ R
            # if i == 40:
            #     d = X[:, 681]
            #     e = torch.max(X)
            #     f = torch.max(W_ve)
            #     g = torch.max(W_ev)
            #     print('error')

            # if torch.isnan(temp).any() or torch.isnan(W_ve).any() or torch.isnan(W_ev).any():
            #     a = torch.nonzero(torch.isnan(temp.view(-1)))
            #     b = torch.nonzero(torch.isnan(X.view(-1)))
            #     c = torch.nonzero(torch.isnan(W_ve.view(-1)))
            #     d = X[:, 681]
            #     e = torch.max(X)
            #     f = torch.max(W_ve)
            #     print('error')
            support_He2 = phi2(support_He1)
            support1 = W_ev @ support_He2
            support2 = support1 @ R.T  # support = torch.spmm(support, Rr.T)
            X_new = phi1(support2 + B)

            err = torch.norm(X_new - X, np.inf)
            err_list.append(err)
            if err < tol:
                status = 'converged'
                # print(status)
                break

            X = X_new

        # # todo
        # sns.set()
        # f = plt.figure(figsize=(5, 3))
        # err_list = [e.cpu().detach().numpy() for e in err_list]
        # plt.plot(list(range(len(err_list))), err_list)
        # plt.show()
        # return

        with torch.enable_grad():
            support_He = W_ve @ X @ R
            Z1 = support_He
            Z1.requires_grad_(True)

            support_He = phi2(Z1)
            dphi2 = torch.autograd.grad(torch.sum(support_He), Z1, only_inputs=True)[0]
            support = W_ev @ support_He
            support = support @ R.T
            Z2 = support + B  # TODO

            Z2.requires_grad_(True)
            X_new = phi1(Z2)
            dphi1 = torch.autograd.grad(torch.sum(X_new), Z2, only_inputs=True)[0]

        return X_new, err, status, dphi1, dphi2

    @staticmethod
    def dz_fixedpoint(W_ev, W_ve, X, R, B, phi1, phi2, mitr=300, tol=3e-6):
        err = 0
        status = 'max itrs reached'
        for i in range(mitr):
            support_He = W_ve @ X @ R  # 128, 380
            # print('support_He.shape is {}'.format(support_He.shape))
            support_He = phi2(support_He)
            support = W_ev @ support_He
            support = support @ R.T
            X_new = phi1(support + B)

            err = torch.norm(X_new - X, np.inf)
            if err < tol:
                status = 'converged'
                break
            X = X_new
        # print('successful')

        return X_new, err, status


class NECFunction(Function):
    @staticmethod
    def forward(ctx, Wv, We, X_0, R, S, B, phi, fd_mitr=300, bw_mitr=300):
        # todo X_0 and B
        X_0 = B if X_0 is None else X_0
        X = NECFunction.inn_pred(Wv, We, X_0, R, S, B, phi, phi, mitr=fd_mitr, compute_dphi=True)
        # ctx.save_for_backward(W_ev, W_ve, X, Rr, Rs, B, D1, D2, X_0, torch.tensor(bw_mitr))
        #if status not in "converged":
        #    print("Iterations not converging!", err, status)
        return X

    @staticmethod
    def inn_pred(Wv, We, X, R, S, B, phi, phi2, mitr=300, tol=3e-6, trasposed_A=False, compute_dphi=False, debug=False):
        err = 0
        status = 'max itrs reached'
        with torch.enable_grad():
            for i in range(mitr):
                # WXA
                # support_He = W_ve @ torch.cat(torch.spmm(X, Rr), torch.spmm(X, Rs))
                tmp0 = nn.Parameter(torch.matmul(X, R), requires_grad=True)
                tmp1 = nn.Parameter(torch.matmul(X, S), requires_grad=True)
                # support_He = W_ve @ torch.cat((X @ Rr, X @ Rs))
                support_He = We @ torch.cat((tmp0, tmp1))
                support_He = phi(support_He)
                support = Wv @ support_He
                support = support @ R.T # support = torch.spmm(support, Rr.T)
                X_new = phi2(support + B)

                err = torch.norm(X_new - X, np.inf)
                if err < tol:
                    status = 'converged'
                    break
                X = X_new
        if status == 'max itrs reached':
            print(status)
        return X

