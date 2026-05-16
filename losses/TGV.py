import torch
import torch.nn as nn
import torch.nn.functional as F

def sym_grad(w):
    wx, wy = w[:, 0:1], w[:, 1:2]
    wx_x = F.pad(wx[:, :, :, 1:] - wx[:, :, :, :-1], (0,1,0,0))
    wx_y = F.pad(wx[:, :, 1:, :] - wx[:, :, :-1, :], (0,0,0,1))
    wy_x = F.pad(wy[:, :, :, 1:] - wy[:, :, :, :-1], (0,1,0,0))
    wy_y = F.pad(wy[:, :, 1:, :] - wy[:, :, :-1, :], (0,0,0,1))
    return wx_x, wy_y, 0.5 * (wx_y + wy_x)

class TGV2Loss(nn.Module):
    def __init__(self, alpha0=1.0, alpha1=2.0, inner_iters=5, lr=0.2, epsilon=1e-6):
        super().__init__()
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.inner_iters = inner_iters
        self.lr = lr
        self.epsilon = epsilon

    def forward(self, u):
        u_y = F.pad(u[:, :, 1:, :] - u[:, :, :-1, :], (0, 0, 0, 1))
        u_x = F.pad(u[:, :, :, 1:] - u[:, :, :, :-1], (0, 1, 0, 0))
        grad_u = torch.cat([u_x, u_y], dim=1)

        w = grad_u.detach().clone()
        w.requires_grad_(True)

        for _ in range(self.inner_iters):
            e11, e22, e12 = sym_grad(w)
            diff = grad_u - w
            loss_inner = self.alpha0 * torch.mean(torch.sqrt(diff**2 + self.epsilon)) + \
                         self.alpha1 * torch.mean(torch.sqrt(e11**2 + e22**2 + 2*e12**2 + self.epsilon))
            grads = torch.autograd.grad(loss_inner, w, create_graph=True)[0]
            w = w - self.lr * grads

        e11, e22, e12 = sym_grad(w)
        diff = grad_u - w
        loss_final = self.alpha0 * torch.mean(torch.sqrt(diff**2 + self.epsilon)) + \
                     self.alpha1 * torch.mean(torch.sqrt(e11**2 + e22**2 + 2*e12**2 + self.epsilon))
        return loss_final
