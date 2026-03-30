import torch
import torch.nn.functional as F
from ENDIGate.utils import acc


def warmup(model, dataloader, test_loader, device,
           epochs=5, lr=1e-3, verbose=True):
    """
    Warmup phase — train weights only, no sparsity pressure.

    Args:
        model:       Net instance
        dataloader:  training DataLoader
        test_loader: test DataLoader
        device:      torch device
        epochs:      number of warmup epochs
        lr:          learning rate for weights
        verbose:     print progress

    Returns:
        model: trained model
        opt:   optimizer (reusable in pruning phase)
    """
    opt = torch.optim.Adam(
        [p for n, p in model.named_parameters() if '.g' not in n],
        lr=lr
    )

    if verbose:
        print("=" * 50)
        print("PHASE 1: Warmup (no sparsity pressure)")
        print("=" * 50)

    for ep in range(epochs):
        model.train()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if verbose:
            print(f"  Ep {ep+1}  Acc: {acc(model, test_loader, device):.2f}%")

    return model, opt
