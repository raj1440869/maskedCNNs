# Mask-Aware CNN for MNIST

This project investigates how CNNs handle partial observations. It compares three training strategies on MNIST digit classification and explores whether a single model can generalise across arbitrary masking patterns.

## The Core Idea

Standard CNNs assume full images at inference time. But what if you only have access to a random subset of pixels? This project tests three approaches:

| Model | Training Input | Mask Strategy |
|-------|---------------|---------------|
| **Unmasked** (baseline) | Full 28×28 image | No masking |
| **Fixed Mask** | 25% of pixels, same positions every image | One fixed random mask, seeded |
| **Random Mask** (mask-aware) | Variable 10–90% of pixels, different per sample | New random mask per image per batch |

The masked models receive a **two-channel input**: `(X ⊙ M, M)` — the masked image concatenated with the mask itself. This lets the model reason about *which* pixels it can see, not just what they contain.

## Project Structure

```
rematch/
├── Unmasked/
│   ├── create_unmasked.py          # Baseline CNN on full images
│   └── unmasked.keras              # Saved baseline model
│
├── Fixed Mask/
│   ├── fixed25masked.py            # CNN trained on fixed 25% subset
│   ├── fixedMask.keras             # Saved fixed-mask model
│   ├── fixed_mask_25percent.npy    # The fixed mask (196/784 pixels)
│   ├── fixed_mask_visualization.png
│   ├── original_vs_masked_comparison.png
│   ├── fixed_mask_predictions.png
│   └── training_history.png
│
├── Random Mask/
│   ├── random_masked.py            # Mask-aware CNN with variable visibility
│   ├── random_mask.keras           # Saved mask-aware model
│   ├── 1_accuracy_over_time.png
│   ├── 2_types_of_masks.png
│   └── 3_masked_vs_original.png
│
└── firstComparison.py              # Evaluates baseline vs fixed-mask on full images
```

## Requirements

```
tensorflow >= 2.x
numpy
matplotlib
```

Install with:

```bash
pip install tensorflow numpy matplotlib
```

## Running the Experiments

Run each script from inside its folder (paths are relative):

```bash
# 1. Train baseline
cd "Unmasked"
python create_unmasked.py

# 2. Train fixed-mask model
cd "../Fixed Mask"
python fixed25masked.py

# 3. Train random/mask-aware model
cd "../Random Mask"
python random_masked.py

# 4. Compare baseline vs fixed-mask model (run from project root)
cd ..
python firstComparison.py
```

## Model Architectures

All three models share the same CNN backbone:

- **Conv2D(32, 3×3, ReLU)** → MaxPool(2×2)
- **Conv2D(64, 3×3, ReLU)** → MaxPool(2×2)
- **Conv2D(128, 3×3, ReLU)**
- **Flatten** → Dropout(0.5) → **Dense(128, ReLU)** → Dropout(0.3) → **Dense(10, Softmax)**

The unmasked baseline takes `(28, 28, 1)` input. The masked models take `(28, 28, 2)` — one channel for the masked image, one for the mask.

## Results

The fixed-mask model approximates `P(Y | X_S)` for one specific pixel subset `S`. The random-mask model approximates this for *any* subset, making it a generalised partial-observation classifier.

Running `firstComparison.py` evaluates both the baseline and fixed-mask model on the full (unmasked) test set and produces a bar chart comparison (`stage4_comparison.png`).

## Reproducibility

All experiments use fixed seeds:
- `tf.random.set_seed(42)`
- `np.random.seed(42)`

The fixed mask is deterministically generated from seed 42, selecting exactly 196 of 784 pixels (25%).
