import torch
import torch.nn as nn

class ENDIGateV3(nn.Module):
    def __init__(self, n, tau=0.01, alpha_start=5.0, alpha_end=50.0,
                 epsilon=0.05, min_alive=2, strikes_needed=3):
        super().__init__()
        self.g = nn.Parameter(torch.ones(n))
        self.tau = tau
        self.alpha = alpha_start
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.epsilon = epsilon
        self.min_alive = min_alive
        self.strikes_needed = strikes_needed

        self.register_buffer('strikes', torch.zeros(n, dtype=torch.long))

    def set_epoch(self, ep, total_eps):
        self.alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * (ep / total_eps)

    def _mask(self):
        with torch.no_grad():
            s = self.g.clamp(0, 1)
            m = s > self.tau
            if m.sum() < self.min_alive:
                v = s.clone()
                v[m] = -1
                need = min(self.min_alive - int(m.sum()), (~m).sum().item())
                if need > 0:
                    _, top = v.topk(need)
                    m[top] = True
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
            below_tau = (self.g.data < self.tau) & ~self._mask()

            self.strikes[below_tau] += 1
            self.strikes[~below_tau] = 0

            truly_dead = below_tau & (self.strikes >= self.strikes_needed)
            self.g.data[truly_dead] = 0.0

    def alive(self):
        return self._mask().nonzero(as_tuple=True)[0]

    def values(self):
        return self.g.clamp(0, 1).detach()

    def alive_count(self):
        return self._mask().sum().item()

    def on_strike(self):
        return ((self.strikes > 0) & (self.strikes < self.strikes_needed)).sum().item()

    def total(self):
        return self.g.numel()
