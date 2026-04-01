class ENDIGateV2(nn.Module):
    def __init__(self, n, tau=0.01, alpha_start=5.0,
                 alpha_end=50.0, epsilon=0.05, min_alive=2):
        super().__init__()
        self.g = nn.Parameter(torch.ones(n))
        self.tau = tau
        self.alpha = alpha_start        # starts soft
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.epsilon = epsilon
        self.min_alive = min_alive

    # Only addition over V1 - alpha annealing
    def set_epoch(self, ep, total):
        self.alpha = self.alpha_start + (
            self.alpha_end - self.alpha_start
        ) * (ep / max(total, 1))

    def _mask(self):
        with torch.no_grad():
            s = self.g.clamp(0, 1)
            mask = s > self.tau
            if mask.sum() < self.min_alive:
                # Clean topk - no edge cases
                _, idx = s.topk(self.min_alive)
                mask = torch.zeros_like(mask, dtype=torch.bool)
                mask[idx] = True
        return mask

    def forward(self, x):
        s = self.g.clamp(0, 1)
        h = s * self._mask().float()
        return x * (s + (h - s).detach())

    def sparsity_loss(self):
        g = self.g.clamp(0, 1)
        return torch.sigmoid(self.alpha * (g - self.epsilon)).sum()

    def project(self):
        # Call this ONCE PER EPOCH not per batch
        with torch.no_grad():
            self.g.clamp_(0, 1)
            mask = self._mask()
            self.g.data[~mask & (self.g.data < self.tau)] = 0.0

    def alive(self):
        return self._mask().nonzero(as_tuple=True)[0]

    def alive_count(self):
        return self._mask().sum().item()

    def values(self):
        return self.g.clamp(0, 1).detach()

    def total(self):
        return self.g.numel()
