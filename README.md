# ENDIGate

**ENDIGate: End-to-End Neural Differentiable Indicator Gating for Feature Selection and Neural Architecture Discovery**

![ENDIGate Demo](https://github.com/sidhu2690/ENDIGate/releases/download/v1.0.0/endigate_mnist_main.gif)

---

## Overview

ENDIGate is a structured pruning method that learns which neurons and input features to retain during training, rather than after.

It augments a feedforward network with differentiable gates that adapt based on task loss and sparsity pressure. Each neuron is assigned a gate value `g ∈ [0, 1]`. Neurons that consistently fail to contribute accumulate strikes and are eventually removed.

The resulting model is a compact architecture extracted directly from the training process.

---

## Method

ENDIGate operates in four phases:

### 1. Warmup
Train the base network until convergence.

### 2. Gating
Learn gate values jointly with network weights under sparsity pressure. Neurons with persistently low gate values accumulate strikes and are marked for removal.

### 3. Extraction
Remove inactive neurons and absorb gate values into the remaining weights, yielding a smaller standalone network.

### 4. Fine-tuning
Fine-tune the extracted network to recover any performance loss.

---

## Strike System

ENDIGate uses a conservative pruning rule:

- A neuron must remain below a threshold `τ`
- For `K` consecutive stable epochs
- Stability is determined by the variance of recent validation loss

Neurons are not immediately removed:
- If a gate value recovers before reaching `K` strikes, the counter resets
- Pruning decisions are only finalized during stable training periods

This avoids premature pruning during noisy optimization phases.

---

## Results (MNIST)

<h3>Model Comparison</h3>

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

**95.1% parameter reduction with improved accuracy.**

The method also identifies a reduced set of input features, retaining only 66 of 784 pixels.

---

## Key Properties

- Differentiable: gates are learned end-to-end via gradient descent  
- Structured: removes entire neurons rather than individual weights  
- Conservative: strike-based pruning prevents premature removal  
- Stability-aware: pruning decisions depend on training stability  
- Modular: distinct phases for learning, pruning, and refinement  
