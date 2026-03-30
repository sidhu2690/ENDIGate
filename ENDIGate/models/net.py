import torch.nn as nn
import torch.nn.functional as F
from ENDIGate.gates import ENDIGateV1  # default gate, can be swapped


def _base():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 64),  nn.ReLU(),
        nn.Linear(64, 10)
    )


class Net(nn.Module):
    def __init__(self, gate_cls=ENDIGateV1):
        """
        Args:
            gate_cls: which ENDIGate version to use. default ENDIGateV1.
                      swap to ENDIGateV2, V3, V4 freely.
        """
        super().__init__()
        self.gate_cls = gate_cls
        self.input_gate = gate_cls(784)
        self.gates = nn.ModuleList()
        layers = []
        for l in _base():
            layers.append(l)
            if isinstance(l, nn.ReLU):
                prev = [x for x in layers if isinstance(x, nn.Linear)][-1]
                g = gate_cls(prev.out_features)
                self.gates.append(g)
                layers.append(g)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.input_gate(x)
        for layer in list(self.net.children())[1:]:
            x = layer(x)
        return x

    def sparse_loss(self):
        return self.input_gate.sparsity_loss() + sum(
            g.sparsity_loss() for g in self.gates
        )

    def project(self):
        self.input_gate.project()
        for g in self.gates:
            g.project()

    def set_epoch(self, ep, total_eps):
        """For V2+ gates that need alpha annealing."""
        if hasattr(self.input_gate, 'set_epoch'):
            self.input_gate.set_epoch(ep, total_eps)
        for g in self.gates:
            if hasattr(g, 'set_epoch'):
                g.set_epoch(ep, total_eps)

    def set_stable(self, is_stable):
        """For V4 gates that need plateau awareness."""
        if hasattr(self.input_gate, 'set_stable'):
            self.input_gate.set_stable(is_stable)
        for g in self.gates:
            if hasattr(g, 'set_stable'):
                g.set_stable(is_stable)

    def on_strike_count(self):
        """For V3+ gates. total neurons currently in trial window."""
        total = 0
        if hasattr(self.input_gate, 'on_strike'):
            total += self.input_gate.on_strike()
        for g in self.gates:
            if hasattr(g, 'on_strike'):
                total += g.on_strike()
        return total

    def alive_summary(self):
        """Print alive counts for all gates."""
        print(f"  Input gate : {self.input_gate.alive_count()}/{self.input_gate.total()}")
        for i, g in enumerate(self.gates):
            print(f"  Hidden gate {i}: {g.alive_count()}/{g.total()}")
