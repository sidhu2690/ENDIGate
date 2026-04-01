import torch
import torch.nn as nn

class ENDIGateV5(nn.Module):
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
        self.stable = False

        self.register_buffer('strikes',     torch.zeros(n, dtype=torch.long))
        self.register_buffer('_mask_cache', torch.zeros(n, dtype=torch.bool))
        self.register_buffer('_mask_valid', torch.tensor(False))

    # ------------------------------------------------------------------
    # Epoch control
    # ------------------------------------------------------------------
    def set_epoch(self, ep, total_eps):
        progress = (ep / max(total_eps, 1)) ** 2
        self.alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * progress
        self._invalidate_mask()

    def set_stable(self, is_stable):
        self.stable = is_stable

    def reset_strikes(self):
        self.strikes.zero_()

    # ------------------------------------------------------------------
    # Mask cache
    # ------------------------------------------------------------------
    def _invalidate_mask(self):
        self._mask_valid.fill_(False)

    def _mask(self):
        if self._mask_valid.item():
            return self._mask_cache
        with torch.no_grad():
            s = self.g.clamp(0, 1)
            thresh_mask = s > self.tau
            k = min(self.min_alive, s.numel())
            _, top_idx = s.topk(k)
            topk_mask = torch.zeros_like(thresh_mask)
            topk_mask[top_idx] = True
            mask = thresh_mask | topk_mask
        self._mask_cache.copy_(mask)
        self._mask_valid.fill_(True)
        return self._mask_cache

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------
    def forward(self, x):
        s = self.g.clamp(0, 1)
        h = s * self._mask().float()
        return x * (s + (h - s).detach())

    def sparsity_loss(self):
        # Warmup phase lets g settle to meaningful values first
        # so plain smooth_l0 is fair — no initialization bias
        g = self.g.clamp(0, 1)
        return torch.sigmoid(self.alpha * (g - self.epsilon)).sum()

    def project(self):
        self._invalidate_mask()
        with torch.no_grad():
            self.g.clamp_(0, 1)
            below_tau = (self.g.data < self.tau) & ~self._mask()
            if self.stable:
                self.strikes[below_tau] += 1
                self.strikes[~below_tau] = 0
                truly_dead = below_tau & (self.strikes >= self.strikes_needed)
                self.g.data[truly_dead] = 0.0
            else:
                self.strikes[~below_tau] = 0
        self._invalidate_mask()

    # ------------------------------------------------------------------
    # Optimizer helper
    # ------------------------------------------------------------------
    @staticmethod
    def make_optimizer(gate_param, lr=0.05, max_grad_norm=1.0):
        opt = torch.optim.Adam([gate_param], lr=lr)
        def clipped_step():
            torch.nn.utils.clip_grad_norm_([gate_param], max_grad_norm)
            opt.step()
        return opt, clipped_step

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def alive(self):
        return self._mask().nonzero(as_tuple=True)[0]

    def alive_count(self):
        return self._mask().sum().item()

    def on_strike(self):
        return ((self.strikes > 0) & (self.strikes < self.strikes_needed)).sum().item()

    def values(self):
        return self.g.clamp(0, 1).detach()

    def total(self):
        return self.g.numel()
