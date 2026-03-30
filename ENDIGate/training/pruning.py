import math
import torch
import torch.nn.functional as F
from ENDIGate.utils import acc


def pruning(model, dataloader, test_loader, device,
            epochs=30, lr_w=1e-3, lr_g=5e-2,
            lam_max=0.1, project_every=3,
            verbose=True, opt_w=None):
    """
    Pruning phase — train weights + gates with cosine sparsity pressure.
    Projects gates every project_every epochs.

    Args:
        model:         Net instance
        dataloader:    training DataLoader
        test_loader:   test DataLoader
        device:        torch device
        epochs:        number of pruning epochs
        lr_w:          weight learning rate
        lr_g:          gate learning rate
        lam_max:       max sparsity lambda (cosine peak)
        project_every: how often to call model.project()
        verbose:       print progress
        opt_w:         optional reuse warmup optimizer

    Returns:
        model: pruned model
    """
    if opt_w is None:
        opt_w = torch.optim.Adam(
            [p for n, p in model.named_parameters() if '.g' not in n],
            lr=lr_w
        )
    opt_g = torch.optim.Adam(
        [p for n, p in model.named_parameters() if '.g' in n],
        lr=lr_g
    )

    best_acc = 0.0

    if verbose:
        print("=" * 50)
        print("PHASE 2: Pruning (cosine lambda, strike counter)")
        print("=" * 50)

    for ep in range(epochs):
        model.train()

        # cosine annealed lambda
        #lam = lam_max * (1 - math.cos(math.pi * ep / max(epochs - 1, 1))) / 2
        # or 
        #lam_min = 0.01 
        #lam = lam_min + (lam_max - lam_min) * (1 - math.cos(math.pi * ep / max(epochs - 1, 1))) / 2
        # or
        lam = lam_min + (lam_max - lam_min) * ep / max(epochs - 1, 1)

        # alpha annealing for V2+ gates
        model.set_epoch(ep, epochs)

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y) + lam * model.sparse_loss()
            opt_w.zero_grad()
            opt_g.zero_grad()
            loss.backward()
            opt_w.step()
            opt_g.step()

        # plateau aware project for V4 gates
        current_acc = acc(model, test_loader, device)
        best_acc    = max(best_acc, current_acc)
        is_stable   = current_acc >= best_acc - 0.3
        model.set_stable(is_stable)

        if (ep + 1) % project_every == 0:
            model.project()

        if verbose and ((ep + 1) % 5 == 0 or ep == 0):
            in_alive  = model.input_gate.alive_count()
            in_total  = model.input_gate.total()
            h_alive   = sum(g.alive_count() for g in model.gates)
            h_total   = sum(g.total() for g in model.gates)
            on_strike = model.on_strike_count()
            print(f"  Ep {ep+1:2d}  Acc: {current_acc:.2f}%  "
                  f"Input: {in_alive}/{in_total} "
                  f"({100*(1-in_alive/in_total):.1f}% pruned)  "
                  f"Hidden: {h_alive}/{h_total} "
                  f"({100*(1-h_alive/h_total):.1f}% pruned)  "
                  f"OnStrike: {on_strike}  "
                  f"Stable: {is_stable}")

    return model
