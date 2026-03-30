import torch
import torch.nn as nn

class ENDIGateV0(nn.Module):
    def __init__(self, n, tau=0.01, alpha=20.0, epsilon=0.05):
        super().__init__()
        self.g = nn.Parameter(torch.ones(n))
        self.tau = tau
        self.alpha = alpha
        self.epsilon = epsilon

    def _mask(self):
        with torch.no_grad():
            m = self.g.clamp(0, 1) > self.tau
        return m

    def forward(self, x):
        s = self.g.clamp(0, 1)
        h = s * self._mask().float()
        return x * (s + (h - s).detach())

    def sparsity_loss(self):
        g = self.g.clamp(0, 1)
        return torch.sigmoid(self.alpha * (g - self.epsilon)).sum()

    def project(self):
        with torch.no_grad():
            self.g.clamp_(0, 1)
            self.g.data[self.g.data < self.tau] = 0.0  # no topk, no mercy

    def alive(self):
        return self._mask().nonzero(as_tuple=True)[0]

    def values(self):
        return self.g.clamp(0, 1).detach()

    def alive_count(self):
        return self._mask().sum().item()

    def total(self):
        return self.g.numel()
