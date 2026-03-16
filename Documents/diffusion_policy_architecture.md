# Diffusion Policy with Transformer — Architecture Reference

## Overview

Diffusion Policy learns to denoise action sequences conditioned on robot observations.
At inference it starts from pure Gaussian noise and iteratively denoises it into a
coherent motion plan. At training it learns to predict the noise that was added to
expert demonstrations.

---

## Inputs

| Input | Raw Shape | Description |
|-------|-----------|-------------|
| Camera image | (84, 84, 3) | RGB frame from fixed overhead camera |
| Robot state | (18,) | 9 joint angles + 9 joint velocities |
| Timestep t | scalar | Noise level index, 0 (clean) to 100 (pure noise) |
| Noisy action chunk | (16, 9) | 16 future timesteps × 9 joints, starts as Gaussian noise |

---

## Step 1 — Encoding inputs into tokens

Every input is projected to the same 256-dim space so the transformer
can treat them uniformly as tokens.

### Image → (1, 256)
```
RGB image (84,84,3)
    → ResNet-18 (conv layers extract spatial features)
    → global average pool
    → (512,) feature vector
    → Linear layer 512 → 256
    → 1 image token (1, 256)
```

### Robot state → (1, 256)
```
Joint angles + velocities (18,)
    → Linear layer 18 → 256
    → 1 state token (1, 256)
```
Simple linear projection — no convolutions needed for structured numerical data.

### Timestep t → (1, 256)
```
Integer t (0 to 100)
    → Sinusoidal embedding: sin(t / 10000^(2i/256)), cos(...)
      (fixed math, NOT learned — unique fingerprint per noise level)
    → Small MLP → 256
    → 1 timestep token (1, 256)
```
Sinusoidal encoding ensures smooth interpolation between noise levels.

### Noisy action chunk → (16, 256)
```
Noisy actions (16, 9)
    → Shared Linear layer 9 → 256  (applied to each of 16 timesteps)
    → Positional encoding added    (so model knows step 1 vs step 16)
    → 16 action tokens (16, 256)
```

---

## Step 2 — Two token groups entering the transformer

```
Context tokens:  [img_token, state_token, time_token]  →  shape (3, 256)
Action tokens:   [a₁, a₂, a₃ ... a₁₆]                 →  shape (16, 256)
```

**Critical asymmetry:**
- Context tokens → Key (K) and Value (V) ONLY — they provide information but never update
- Action tokens → Query (Q) in cross-attention, Q+K+V in self-attention — they are refined

---

## Step 3 — Inside one transformer decoder layer (repeated 4×)

### A) Self-attention: action tokens attend each other
```
Each action token aᵢ:
    Qᵢ = aᵢ · Wq    "what do I need?"
    Kⱼ = aⱼ · Wk    "what does each other token have?"
    Vⱼ = aⱼ · Wv    "actual content of each token"

score(i,j) = Qᵢ · Kⱼ / √256
weights    = softmax(scores)
aᵢ'        = Σⱼ weights[j] × Vⱼ
```
Result: each action token `aᵢ'` now contains a blend of all 16 tokens.
Token a₁ (reach) now knows what a₁₆ (place) is planning — temporal coherence.

### B) Cross-attention: action tokens query observation
```
Q = action tokens a'     (16, 256)
K = context tokens       (3, 256)   ← image, state, time
V = context tokens       (3, 256)

Each action token asks: "which part of image/state/time is relevant to my step?"
Token a₃ (grasp phase) attends heavily to cube position in image.
Token a₄ (lift phase) attends to gripper state.
```
Result: action tokens `a''` are now observation-aware.

### C) Feed-forward network
```
Per token: Linear(256 → 1024) → ReLU → Linear(1024 → 256)
+ LayerNorm and residual connection: output = LayerNorm(x + sublayer(x))
```

After 4 layers of A→B→C, action tokens are:
- Temporally consistent (self-attention)
- Observation-aware (cross-attention)
- Nonlinearly transformed (FFN)

---

## Step 4 — Linear head and output
```
Refined action tokens (16, 256)
    → Linear layer 256 → 9 (applied per token)
    → Predicted noise ε̂  (16, 9)
```

---

## Step 5 — Denoising loop (inference only)
```
action_chunk = Gaussian noise (16, 9)
for t = T down to 0:
    ε̂ = transformer(action_chunk, obs_tokens, t)
    action_chunk = action_chunk − α_t × ε̂     ← DDPM step
    # DDIM: only 10 steps needed instead of 100
→ clean action chunk (16, 9)
→ send 16 joint targets to robot controller
→ execute, then re-run policy with new observation
```

---

## Training loop

```
for each batch:
    1. Sample clean demo action chunk a⁰ (16, 9) + observation
    2. Sample random t ~ Uniform(0, T)
    3. Sample random noise ε ~ N(0, I)  shape (16, 9)
    4. Corrupt: aᵗ = √ᾱₜ · a⁰ + √(1−ᾱₜ) · ε
    5. Forward pass: ε̂ = transformer(aᵗ, obs, t)
    6. Loss = MSE(ε̂, ε) = mean((ε̂ − ε)²)
    7. Backprop: gradients flow through ALL layers
    8. Update all weights θ ← θ − lr × ∂Loss/∂θ
```

**Why MSE on noise and not on actions?**
We always know the exact noise ε we added (we sampled it ourselves).
The loss is always well-defined. No need for discriminators or RL rewards.

### Backprop path (in order, all updated):
```
Loss
  → Linear head (256→9)
  → Transformer layer 4 (self-attn Wq,Wk,Wv + cross-attn Wq,Wk,Wv + FFN)
  → Transformer layer 3
  → Transformer layer 2
  → Transformer layer 1
  → Action token projection (9→256)
  → State encoder MLP (18→256)
  → Timestep MLP (sinusoidal→256)
  → ResNet-18 (if not frozen)
```

---

## ResNet-18 — To freeze or not to freeze?

### The problem: catastrophic forgetting
Fine-tuning a pretrained ResNet on small robot datasets risks overwriting
its general visual knowledge. With only 100 demos, gradients from robot-specific
data can corrupt ImageNet-learned features.

### Recommendation by dataset size:

| Dataset size | ResNet strategy | Reason |
|---|---|---|
| < 200 demos | **Freeze** | Too little data — will overfit |
| 200–500 demos | Fine-tune with low lr (1e-5) | Gentle adaptation |
| > 500 demos | Full fine-tune | Enough data to adapt safely |

### Config in Lerobot:
```yaml
vision_backbone: resnet18
pretrained_backbone_weights: ResNet18_Weights.IMAGENET1K_V1
freeze_backbone: true   # recommended for our ~100 demo dataset
```

### Future improvement (worth mentioning in README):
Replace ResNet-18 with a CLIP or DINOv2 pretrained ViT backbone.
These transfer much better to robot manipulation because they were trained
on broader visual concepts including spatial relationships and object properties.

---

## Key numbers for our project

| Parameter | Value |
|-----------|-------|
| Action chunk size | 16 timesteps |
| Observation horizon | 2 frames |
| Token dimension | 256 |
| Transformer layers | 4 |
| Attention heads | 8 (multi-head) |
| FFN hidden dim | 1024 |
| Denoising steps (DDPM) | 100 |
| Denoising steps (DDIM) | 10 |
| Action space | 9 DOF (7 arm + 2 fingers) |
| Image size | 84 × 84 RGB |
| ResNet backbone | ResNet-18 (frozen) |
| Training epochs | ~2000 |
| Estimated training time | 8–12 hrs on RTX 4070 |

---

## One-paragraph summary 

Diffusion Policy with Transformer takes a camera image, robot joint states, and a
noisy action chunk as input. Each input is encoded into a 256-dim token. The image
goes through a frozen ResNet-18, states through a linear MLP, the timestep through
a sinusoidal embedding, and each of the 16 action steps through a shared linear
projection. The transformer decoder runs 4 layers of self-attention (making action
tokens temporally coherent) followed by cross-attention (grounding each action token
in the current observation). A linear head projects the refined tokens to predicted
noise. At inference, starting from Gaussian noise, 10 DDIM denoising steps produce
a clean 16-step motion plan the robot executes. At training, we corrupt expert
demonstrations with known noise and minimize MSE between predicted and actual noise,
with gradients flowing through every layer end-to-end.
