import torch


def acc(model, dataloader, device):
    """
    Compute accuracy of model on dataloader.

    Args:
        model:      any nn.Module
        dataloader: test DataLoader
        device:     torch device

    Returns:
        float: accuracy percentage 0-100
    """
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
    return 100 * correct / len(dataloader.dataset)


def count_params(model):
    """Total number of learnable parameters."""
    return sum(p.numel() for p in model.parameters())


def compression_ratio(original, compact):
    """
    How much smaller compact is versus original.

    Args:
        original: original nn.Module
        compact:  compact nn.Module

    Returns:
        float: percentage reduction e.g. 95.1
    """
    o = count_params(original)
    c = count_params(compact)
    return 100 * (1 - c / o)


def gate_summary(model, total_eps=None):
    """
    Print alive/total for all gates in a Net.

    Args:
        model:      Net instance
        total_eps:  optional, just for display context
    """
    print(f"  Input gate : {model.input_gate.alive_count()}/{model.input_gate.total()}"
          f"  ({100*(1 - model.input_gate.alive_count()/model.input_gate.total()):.1f}% pruned)")
    for i, g in enumerate(model.gates):
        print(f"  Hidden gate {i}: {g.alive_count()}/{g.total()}"
              f"  ({100*(1 - g.alive_count()/g.total()):.1f}% pruned)")
