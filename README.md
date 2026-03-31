# ENDIGate

**ENDIGate: End-to-End Neural Differentiable Indicator Gating for Feature Selection and Neural Architecture Discovery**

![ENDIGate Demo](https://github.com/sidhu2690/ENDIGate/releases/download/v1.0.0/endigate_mnist_main.gif)

---

## What is ENDIGate?

ENDIGate is a structured pruning method that learns **which neurons and input features to keep** during training — not after. It wraps any feedforward network with differentiable gates that open or close based on a combination of task loss and sparsity pressure.

The key idea: every neuron gets a soft gate value `g ∈ [0, 1]`. Neurons that consistently fail to contribute accumulate **strikes**. After enough strikes during stable training, they are permanently removed. The surviving network is then extracted as a smaller, standalone model.

---

## How it works
Phase 1 — Warmup: Train the base network normally until convergence
Phase 2 — Gating: Gates learn alongside weights under sparsity pressure, Weak neurons accumulate strikes → get pruned
Phase 3 — Extraction: Dead neurons removed, gates absorbed into weights
Phase 4 — Fine-tuning: Compact network fine-tuned to recover accuracy

---

### The Strike System

Unlike standard magnitude pruning or L0 regularisation, ENDIGate uses a **conservative kill rule**:

- A neuron must stay below threshold `τ` for `K` consecutive **stable** epochs
- Stability is measured from the recent validation loss variance
- Neurons can recover — if a gate rebounds before `K` strikes, the counter resets

This prevents premature pruning during noisy or unstable training phases.

---

## Results on MNIST

<table>
  <tr>
    <th align="left">Model</th>
    <th align="right">Params</th>
    <th align="right">Accuracy</th>
    <th align="right">Features</th>
  </tr>
  <tr>
    <td>Base</td>
    <td align="right">243,994</td>
    <td align="right">96.36%</td>
    <td align="right">784</td>
  </tr>
  <tr>
    <td><b>ENDIGate</b></td>
    <td align="right"><b>12,031</b></td>
    <td align="right"><b>96.89%</b></td>
    <td align="right"><b>66</b></td>
  </tr>
</table>

**95.1% parameter reduction. Accuracy preserved.**

The method also identifies which input pixels matter — only 66 of 784 positions are retained, forming a sparse yet sufficient feature set.

---

## Key Properties

- **Differentiable** — gates are learned end-to-end via gradient descent  
- **Structured** — removes whole neurons, not individual weights → real speedup  
- **Conservative** — strike system prevents premature or irreversible pruning  
- **Stability-aware** — pruning only committed during stable training windows  
- **Three-phase pipeline** — clean separation of learning, discovery, compression  
