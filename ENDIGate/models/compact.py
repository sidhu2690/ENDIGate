import torch
import torch.nn as nn


class CompactModel(nn.Module):
    """
    Wraps an extracted compact sequential with input feature selection.
    alive_indices: which of the 784 input pixels survived pruning.
    sequential:    the pruned and baked sequential network.
    """
    def __init__(self, alive_indices, sequential):
        super().__init__()
        self.register_buffer('alive_indices', alive_indices)
        self.net = sequential

    def forward(self, x):
        x = x.view(x.size(0), -1)        # flatten to 784
        x = x[:, self.alive_indices]       # select only alive inputs
        return self.net(x)                 # forward through compact net

    def input_size(self):
        return len(self.alive_indices)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def architecture(self):
        """Returns list of layer sizes for display."""
        sizes = [self.input_size()]
        for l in self.net:
            if isinstance(l, nn.Linear):
                sizes.append(l.out_features)
        return sizes

    def architecture_str(self):
        return "→".join(str(s) for s in self.architecture())
