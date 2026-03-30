import torch
import torch.nn.functional as F
from ENDIGate.utils import acc


def finetune(model, dataloader, test_loader, device,
             epochs=10, lr=1e-4, verbose=True):
    """
    Finetune phase — fine tune compact model after extraction.

    Args:
        model:       CompactModel instance
        dataloader:  training DataLoader
        test_loader: test DataLoader
        device:      torch device
        epochs:      number of finetune epochs
        lr:          learning rate
        verbose:     print progress

    Returns:
        model: finetuned compact model
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    if verbose:
        print("=" * 50)
        print("PHASE 4: Fine-tuning compact model")
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

    return model
