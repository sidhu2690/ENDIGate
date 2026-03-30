import torch
import torch.nn as nn
from ENDIGate.models.compact import CompactModel


def extract(m):
    """
    Extract a compact model from a trained gated Net.
    Bakes gate values into weights, removes dead neurons,
    and wraps with input feature selection.

    Args:
        m: trained Net instance

    Returns:
        CompactModel: lean model with only alive neurons
    """
    m.eval()

    input_alive = m.input_gate.alive()
    input_vals  = m.input_gate.values()

    lins        = [l for l in m.net if isinstance(l, nn.Linear)]
    gates       = list(m.gates)
    hidden_alive = [g.alive() for g in gates]

    new_layers = []

    with torch.no_grad():

        # first hidden layer
        lin   = lins[0]
        h_idx = hidden_alive[0]
        w     = lin.weight.data[h_idx][:, input_alive]
        g_in  = input_vals[input_alive]
        w     = w * g_in.unsqueeze(0)
        b     = lin.bias.data[h_idx]
        l     = nn.Linear(len(input_alive), len(h_idx))
        l.weight.data.copy_(w)
        l.bias.data.copy_(b)
        new_layers += [l, nn.ReLU()]

        # middle hidden layers
        for i in range(1, len(lins) - 1):
            lin      = lins[i]
            prev_idx = hidden_alive[i - 1]
            curr_idx = hidden_alive[i]
            w        = lin.weight.data[curr_idx][:, prev_idx]
            g_prev   = gates[i - 1].values()[prev_idx]
            w        = w * g_prev.unsqueeze(0)
            b        = lin.bias.data[curr_idx]
            l        = nn.Linear(len(prev_idx), len(curr_idx))
            l.weight.data.copy_(w)
            l.bias.data.copy_(b)
            new_layers += [l, nn.ReLU()]

        # final layer
        lin      = lins[-1]
        prev_idx = hidden_alive[-1]
        g_prev   = gates[-1].values()[prev_idx]
        w        = lin.weight.data[:, prev_idx] * g_prev.unsqueeze(0)
        b        = lin.bias.data
        l        = nn.Linear(len(prev_idx), 10)
        l.weight.data.copy_(w)
        l.bias.data.copy_(b)
        new_layers.append(l)

    sequential = nn.Sequential(*new_layers)
    return CompactModel(input_alive, sequential)
