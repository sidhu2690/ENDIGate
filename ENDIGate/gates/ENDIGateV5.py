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
        self.stable = False                          # Fix 1: default False

        self.register_buffer('strikes', torch.zeros(n, dtype=torch.long))

        # Fix 1: mask cache
        self._mask_cache = None
        self._mask_valid = False

    # ------------------------------------------------------------------
    # Epoch control
    # ------------------------------------------------------------------
    def set_epoch(self, ep, total_eps):
        # Curved schedule: slow start, fast end
        progress = (ep / max(total_eps, 1)) ** 2
        self.alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * progress
        self._invalidate_mask()

    def set_stable(self, is_stable):
        self.stable = is_stable

    def reset_strikes(self):
        self.strikes.zero_()

    # ------------------------------------------------------------------
    # Mask cache helpers
    # ------------------------------------------------------------------
    def _invalidate_mask(self):
        self._mask_valid = False
        self._mask_cache = None

    def _mask(self):
        # Fix 1: return cached mask if still valid
        if self._mask_valid and self._mask_cache is not None:
            return self._mask_cache

        with torch.no_grad():
            s = self.g.clamp(0, 1)

            # Fix 2: clean topk, no -1 trick
            # Step 1: threshold mask
            thresh_mask = s > self.tau
            # Step 2: topk mask for min_alive guarantee
            k = min(self.min_alive, s.numel())
            _, top_idx = s.topk(k)
            topk_mask = torch.zeros_like(thresh_mask)
            topk_mask[top_idx] = True
            # Step 3: OR both — threshold survivors + min_alive guarantee
            mask = thresh_mask | topk_mask

        self._mask_cache = mask
        self._mask_valid = True
        return mask

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------
    def forward(self, x):
        s = self.g.clamp(0, 1)
        h = s * self._mask().float()                # Fix 1: uses cache
        return x * (s + (h - s).detach())

    def sparsity_loss(self):
        g = self.g.clamp(0, 1)
        # Fix 3: confident features (g close to 1) get reduced penalty
        # confidence = g itself, so weight = (1 - g)
        # → g=1.0 → weight=0.0 (no penalty)
        # → g=0.5 → weight=0.5 (half penalty)
        # → g=0.0 → weight=1.0 (full penalty, but already dead)
        confidence_weight = (1.0 - g).clamp(min=0)
        smooth_l0 = torch.sigmoid(self.alpha * (g - self.epsilon))
        return (smooth_l0 * confidence_weight).sum()

    def project(self):
        # Fix 1: invalidate cache since g.data changes
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

        self._invalidate_mask()                     # invalidate after g change

    # ------------------------------------------------------------------
    # Fix 4: gate optimizer with gradient clipping
    # ------------------------------------------------------------------
    @staticmethod
    def make_optimizer(gate_param, lr=0.05, max_grad_norm=1.0):
        """
        Returns optimizer + a step function that clips gate gradients.
        Usage:
            opt_g, step_g = ENDIGateV5.make_optimizer(model.gate.g)
            # in training loop:
            step_g()   instead of   opt_g.step()
        """
        opt = torch.optim.Adam([gate_param], lr=lr)

        def clipped_step():
            torch.nn.utils.clip_grad_norm_([gate_param], max_grad_norm)
            opt.step()

        return opt, clipped_step

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def alive(self):
        return self._mask().nonzero(as_tuple=True)[0]   # Fix 1: uses cache

    def alive_count(self):
        return self._mask().sum().item()                # Fix 1: uses cache

    def on_strike(self):
        return ((self.strikes > 0) & (self.strikes < self.strikes_needed)).sum().item()

    def values(self):
        return self.g.clamp(0, 1).detach()

    def total(self):
        return self.g.numel()
