import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize
x_test = x_test.astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1) # Shape: (10000, 28, 28, 1)

# One-hot encode labels
y_test_cat = keras.utils.to_categorical(y_test, 10)

# 2. Prepare Inputs

# Input for Baseline Model (Standard Image)
input_baseline = x_test 

# Input for Masked Model (Image + Full Mask)
# We create a mask of ALL ONES (Full Visibility) to test "Full Image" performance
full_mask = np.ones((28, 28, 1), dtype="float32")
mask_batch = np.tile(full_mask, (len(x_test), 1, 1, 1))

# The masked model expects (X * M, M). Since M is all 1s, X * M = X.
input_masked_model = np.concatenate([x_test, mask_batch], axis=-1)

# 3. Load Models
# Update these paths to your actual .keras files
path_baseline = 'Unmasked/unmasked.keras'
path_fixed = 'Fixed Mask/fixedMask.keras'

try:
    model_baseline = keras.models.load_model(path_baseline)
    print("✓ Baseline Model Loaded")
except:
    print(f"ERROR: Could not load baseline model at {path_baseline}")
    model_baseline = None

try:
    model_masked = keras.models.load_model(path_fixed)
    print("✓ Masked Model Loaded")
except:
    print(f"ERROR: Could not load masked model at {path_fixed}")
    model_masked = None

# 4. Evaluate & Graph
if model_baseline and model_masked:
    print("\nEvaluating Baseline on Full Images...")
    _, acc_base = model_baseline.evaluate(input_baseline, y_test_cat, verbose=0)
    
    print("Evaluating Masked Model on Full Images (with Full Mask)...")
    _, acc_mask = model_masked.evaluate(input_masked_model, y_test_cat, verbose=0)

    print(f"\nResults (Accuracy on Full Unmasked MNIST):")
    print(f"Baseline: {acc_base*100:.2f}%")
    print(f"Masked:   {acc_mask*100:.2f}%")

    # Bar Graph
    models = ['Baseline (Unmasked)', 'Masked Model (Full Input)']
    accuracies = [acc_base * 100, acc_mask * 100]
    colors = ['#3498db', '#e74c3c']

    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, accuracies, color=colors)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}%',
                 ha='center', va='bottom', fontweight='bold')

    plt.ylabel('Accuracy (%)')
    plt.title('Baseline vs. Fixed-Mask Model: Accuracy on Full Unmasked Images')
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('stage4_comparison.png')
    plt.show()