import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import sys

# ── make sure package is findable when run directly ──────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ENDIGate.models   import Net
from ENDIGate.gates    import ENDIGateV4
from ENDIGate.extract  import extract
from ENDIGate.utils    import acc, count_params, compression_ratio, gate_summary
from ENDIGate.training import warmup, pruning, finetune

# ── paths ─────────────────────────────────────────────────────────────────────
HERE        = os.path.dirname(__file__)
FIGURES_DIR = os.path.join(HERE, 'figures')
RESULTS_FILE = os.path.join(HERE, 'results.txt')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── device ────────────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using: {DEVICE}")

# ── data ──────────────────────────────────────────────────────────────────────
tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
tr = DataLoader(datasets.MNIST('data', train=True,  download=True, transform=tfm), 128, shuffle=True)
te = DataLoader(datasets.MNIST('data', train=False, transform=tfm), 128)

# ── train ─────────────────────────────────────────────────────────────────────
model        = Net(gate_cls=ENDIGateV4).to(DEVICE)
model, opt_w = warmup(model,   tr, te, DEVICE, epochs=5)
model        = pruning(model,  tr, te, DEVICE, epochs=30, opt_w=opt_w)

print(f"\nFinal gate status:")
gate_summary(model)

compact = extract(model).to(DEVICE)

print(f"\n{'=' * 50}")
print("PHASE 3: Extracted compact model")
print("=" * 50)
print(f"  Architecture : {compact.architecture_str()}")
print(f"  Input features: {compact.input_size()}/784")
print(f"  Params: {count_params(compact):,} ({compression_ratio(model, compact):.1f}% smaller)")
print(f"  Acc (before finetune): {acc(compact, te, DEVICE):.2f}%")

compact = finetune(compact, tr, te, DEVICE, epochs=10)

# ── results ───────────────────────────────────────────────────────────────────
orig_acc    = acc(model,   te, DEVICE)
compact_acc = acc(compact, te, DEVICE)
reduction   = compression_ratio(model, compact)

print(f"\n{'=' * 50}")
print("FINAL RESULTS")
print("=" * 50)
print(f"  Original : {count_params(model):>8,} params  Acc: {orig_acc:.2f}%")
print(f"  Compact  : {count_params(compact):>8,} params  Acc: {compact_acc:.2f}%")
print(f"  Reduction: {reduction:.1f}%")

# ── save results.txt ──────────────────────────────────────────────────────────
with open(RESULTS_FILE, 'w') as f:
    f.write("MNIST BASELINE RESULTS\n")
    f.write("=" * 50 + "\n")
    f.write(f"Device        : {DEVICE}\n")
    f.write(f"Gate version  : ENDIGateV4\n")
    f.write(f"Architecture  : {compact.architecture_str()}\n")
    f.write(f"Input features: {compact.input_size()}/784\n\n")
    f.write(f"Original  params : {count_params(model):,}\n")
    f.write(f"Compact   params : {count_params(compact):,}\n")
    f.write(f"Reduction        : {reduction:.1f}%\n\n")
    f.write(f"Original  acc    : {orig_acc:.2f}%\n")
    f.write(f"Compact   acc    : {compact_acc:.2f}%\n")

print(f"\nSaved: {RESULTS_FILE}")

# ── visualizations ────────────────────────────────────────────────────────────

gate_vals  = model.input_gate.values().cpu().numpy().reshape(28, 28)
alive_mask = torch.zeros(784)
alive_mask[model.input_gate.alive().cpu()] = 1.0
alive_mask  = alive_mask.numpy().reshape(28, 28)
alive_count = int(alive_mask.sum())

# ── 1. input gate heatmap ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

im = axes[0].imshow(gate_vals, cmap='hot', vmin=0, vmax=1)
axes[0].set_title('Input Gate Values\n(brighter = more important)', fontsize=12)
plt.colorbar(im, ax=axes[0], fraction=0.046)

axes[1].imshow(alive_mask, cmap='Greens', vmin=0, vmax=1)
axes[1].set_title(f'Surviving Pixels\n({alive_count}/784 = {100*alive_count/784:.1f}% kept)', fontsize=12)

avg_digit = np.zeros((28, 28))
cnt = 0
for x, y in te:
    avg_digit += x.squeeze(1).sum(0).numpy()
    cnt += x.shape[0]
avg_digit /= cnt
axes[2].imshow(avg_digit, cmap='gray', alpha=0.5)
axes[2].imshow(gate_vals, cmap='hot',  alpha=0.5)
axes[2].set_title('Gate Overlay on Average Digit', fontsize=12)

for ax in axes:
    ax.axis('off')
plt.suptitle('ENDIGate: Automatic Input Feature Discovery', fontsize=14, fontweight='bold')
plt.tight_layout()
path = os.path.join(FIGURES_DIR, 'input_gate_heatmap.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {path}")

# ── 2. sample digits original vs gated ───────────────────────────────────────
fig, axes = plt.subplots(2, 10, figsize=(20, 5))

samples = {}
for x, y in te:
    for i in range(x.shape[0]):
        label = y[i].item()
        if label not in samples:
            samples[label] = x[i]
        if len(samples) == 10:
            break
    if len(samples) == 10:
        break

for digit in range(10):
    img = samples[digit].squeeze().numpy()

    axes[0, digit].imshow(img, cmap='gray')
    axes[0, digit].set_title(f'{digit}', fontsize=11, fontweight='bold')
    axes[0, digit].axis('off')

    axes[1, digit].imshow(img * alive_mask, cmap='gray')
    axes[1, digit].axis('off')

axes[0, 0].set_ylabel('Original', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Gated',    fontsize=12, fontweight='bold')
plt.suptitle('ENDIGate: Original vs Gated Input Per Digit', fontsize=14, fontweight='bold')
plt.tight_layout()
path = os.path.join(FIGURES_DIR, 'sample_digits_gated.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {path}")

# ── 3. per digit gate overlay ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for digit in range(10):
    ax = axes[digit // 5, digit % 5]
    digit_imgs = []
    for x, y in te:
        mask = (y == digit)
        if mask.any():
            digit_imgs.append(x[mask])
        if len(digit_imgs) > 10:
            break
    digit_avg = torch.cat(digit_imgs).mean(0).squeeze().numpy()
    ax.imshow(digit_avg, cmap='gray', alpha=0.4)
    ax.imshow(gate_vals, cmap='hot',  alpha=0.6)
    ax.set_title(f'Digit {digit}', fontsize=11)
    ax.axis('off')

plt.suptitle('ENDIGate: Gate Importance Overlaid on Each Digit Class',
             fontsize=14, fontweight='bold')
plt.tight_layout()
path = os.path.join(FIGURES_DIR, 'per_digit_gate.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {path}")

# ── 4. architecture comparison ────────────────────────────────────────────────
import torch.nn as nn

fig, ax = plt.subplots(1, 1, figsize=(12, 4))

layers_original = [784, 256, 128, 64, 10]
layers_compact  = compact.architecture()

x_pos = range(len(layers_original))
ax.bar([x - 0.2 for x in x_pos], layers_original, 0.4,
       label='Original',        color='#ff6b6b', alpha=0.8)
ax.bar([x + 0.2 for x in x_pos], layers_compact,  0.4,
       label='ENDIGate Compact', color='#4ecdc4', alpha=0.8)

for i, v in enumerate(layers_original):
    ax.text(i - 0.2, v + 5, str(v), ha='center', fontweight='bold', fontsize=10)
for i, v in enumerate(layers_compact):
    ax.text(i + 0.2, v + 5, str(v), ha='center', fontweight='bold', fontsize=10)

ax.set_xticks(range(len(layers_original)))
ax.set_xticklabels(['Input', 'Hidden 1', 'Hidden 2', 'Hidden 3', 'Output'])
ax.set_ylabel('Neurons', fontsize=12)
ax.set_title(f'Architecture Comparison\n'
             f'Original: {count_params(model):,} params → '
             f'Compact: {count_params(compact):,} params '
             f'({reduction:.1f}% reduction)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
path = os.path.join(FIGURES_DIR, 'architecture_comparison.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {path}")

# ── 5. pixel importance map ───────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(gate_vals, cmap='inferno', vmin=0, vmax=1)
for i in range(29):
    ax.axhline(i - 0.5, color='white', linewidth=0.3, alpha=0.3)
    ax.axvline(i - 0.5, color='white', linewidth=0.3, alpha=0.3)

ax.set_title(f'Pixel-Level Gate Values\n'
             f'{alive_count} / 784 pixels alive ({100*alive_count/784:.1f}%)',
             fontsize=14, fontweight='bold')
ax.axis('off')
plt.colorbar(plt.cm.ScalarMappable(cmap='inferno'), ax=ax,
             fraction=0.046, label='Gate Value')
plt.tight_layout()
path = os.path.join(FIGURES_DIR, 'pixel_importance.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {path}")

print("\n" + "=" * 50)
print("ALL DONE")
print("=" * 50)
