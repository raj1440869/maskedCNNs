import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape to add channel dimension (28x28x1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"Training samples: {x_train.shape[0]}")
print(f"Test samples: {x_test.shape[0]}")
print(f"Image shape: {x_train.shape[1:]}")

# Convert labels to categorical (one-hot encoding)
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)


def create_fixed_mask_25percent(shape=(28, 28), seed=42):
    """
    Create a FIXED binary mask with EXACTLY 25% of pixels visible.
    This mask is created ONCE and used for ALL images.
    
    Args:
        shape: Shape of the mask (default: 28x28)
        seed: Random seed for reproducibility
    
    Returns:
        Binary mask with shape (28, 28, 1) where exactly 25% of entries are 1
    """
    np.random.seed(seed)
    
    # Total number of pixels
    total_pixels = shape[0] * shape[1]  # 28 * 28 = 784
    
    # Calculate 25% of pixels
    num_visible = int(total_pixels * 0.25)  # 784 * 0.25 = 196 pixels
    
    # Create a flat array of zeros
    mask_flat = np.zeros(total_pixels, dtype="float32")
    
    # Randomly select indices to set to 1 (but with fixed seed)
    visible_indices = np.random.choice(total_pixels, num_visible, replace=False)
    mask_flat[visible_indices] = 1.0
    
    # Reshape to 28x28 and add channel dimension
    mask = mask_flat.reshape(shape)
    mask = np.expand_dims(mask, -1)
    
    # Verify exactly 25%
    print(f"Fixed mask created: {np.sum(mask)} out of {total_pixels} pixels visible")
    print(f"Percentage visible: {np.sum(mask) / total_pixels * 100:.2f}%")
    
    return mask


def apply_fixed_mask_to_dataset(images, mask):
    """
    Apply the SAME fixed mask to ALL images in the dataset.
    
    Args:
        images: Array of images with shape (N, 28, 28, 1)
        mask: Fixed mask with shape (28, 28, 1)
    
    Returns:
        Two-channel input (X ⊙ M, M) with shape (N, 28, 28, 2)
    """
    # Broadcast mask to match batch size
    # mask shape: (28, 28, 1) -> broadcasts to (N, 28, 28, 1)
    masked_images = images * mask
    
    # Create mask array for all images
    # Same mask repeated N times
    mask_batch = np.tile(mask, (images.shape[0], 1, 1, 1))
    
    # Concatenate along channel dimension
    two_channel_input = np.concatenate([masked_images, mask_batch], axis=-1)
    
    return two_channel_input


def create_fixed_mask_model():
    """
    Create CNN model for fixed mask training.
    Same architecture as other models but trained on ONE fixed 25% subset.
    """
    model = keras.Sequential([
        # Input: 28x28x2 (masked image + mask)
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", 
                           input_shape=(28, 28, 2)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation="softmax")
    ])
    
    return model


# ============================================================================
# STAGE 3: Create and apply FIXED 25% mask
# ============================================================================

print("\n" + "="*70)
print("STAGE 3: Training model with FIXED 25% mask")
print("="*70)
print("ALL images use the SAME 25% mask across ALL epochs")
print("="*70)

# Create the fixed mask (this mask will be used for ALL images, ALL epochs)
print("\nCreating fixed mask with exactly 25% visible pixels...")
fixed_mask = create_fixed_mask_25percent(shape=(28, 28), seed=42)

# VISUALIZATION 1: Show the fixed mask
print("\nVisualizing the fixed mask...")
plt.figure(figsize=(8, 8))
plt.imshow(fixed_mask[:, :, 0], cmap='gray')
plt.title('Fixed Mask (White = Visible, Black = Hidden)\n196 out of 784 pixels visible (25%)', 
          fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig('fixed_mask_visualization.png', dpi=150, bbox_inches='tight')
print("Fixed mask saved to: fixed_mask_visualization.png")
plt.show()

# VISUALIZATION 2: Show examples of original vs masked images
print("\nVisualizing original vs masked images...")
fig, axes = plt.subplots(3, 6, figsize=(15, 8))

for i in range(9):
    # Get a random test image
    idx = np.random.randint(len(x_test))
    original_img = x_test[idx]
    label = y_test[idx]
    
    # Apply mask
    masked_img = original_img * fixed_mask
    
    # Show original
    ax_orig = axes[i // 3, (i % 3) * 2]
    ax_orig.imshow(original_img[:, :, 0], cmap='gray')
    ax_orig.set_title(f'Original: {label}', fontsize=10)
    ax_orig.axis('off')
    
    # Show masked
    ax_mask = axes[i // 3, (i % 3) * 2 + 1]
    ax_mask.imshow(masked_img[:, :, 0], cmap='gray')
    ax_mask.set_title(f'Masked (25%)', fontsize=10)
    ax_mask.axis('off')

plt.suptitle('Original Images vs Fixed 25% Masked Images', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('original_vs_masked_comparison.png', dpi=150, bbox_inches='tight')
print("Original vs masked comparison saved to: original_vs_masked_comparison.png")
plt.show()

# Apply fixed mask to ALL training images
print("\nApplying fixed mask to all training images...")
x_train_fixed = apply_fixed_mask_to_dataset(x_train, fixed_mask)
print(f"Training data shape: {x_train_fixed.shape}")

# Split validation set
val_size = int(0.1 * len(x_train_fixed))
x_val_fixed = x_train_fixed[-val_size:]
y_val = y_train_cat[-val_size:]
x_train_fixed = x_train_fixed[:-val_size]
y_train_fixed = y_train_cat[:-val_size]

print(f"Training samples after split: {x_train_fixed.shape[0]}")
print(f"Validation samples: {x_val_fixed.shape[0]}")

# Apply fixed mask to test images
print("\nApplying fixed mask to test images...")
x_test_fixed = apply_fixed_mask_to_dataset(x_test, fixed_mask)

# Create and compile model
print("\nBuilding fixed-mask model...")
model_fixed = create_fixed_mask_model()

model_fixed.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Display model architecture
model_fixed.summary()

# Train the model
print("\nTraining on fixed 25% mask...")
print("Note: Model sees the SAME 25% of pixels for every image, every epoch")
history = model_fixed.fit(
    x_train_fixed, y_train_fixed,
    batch_size=128,
    epochs=15,
    validation_data=(x_val_fixed, y_val),
    verbose=1
)

# VISUALIZATION 3: Training history
print("\nVisualizing training history...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot accuracy
ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot loss
ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print("Training history saved to: training_history.png")
plt.show()

# Evaluate on test set
print("\nEvaluating on test set with fixed 25% mask...")
test_loss, test_accuracy = model_fixed.evaluate(x_test_fixed, y_test_cat, verbose=0)
print(f"Test accuracy (fixed 25% mask): {test_accuracy*100:.2f}%")
print(f"Test loss: {test_loss:.4f}")

# VISUALIZATION 4: Predictions on test set
print("\nVisualizing predictions on test images...")
fig, axes = plt.subplots(3, 5, figsize=(15, 9))

for i in range(15):
    ax = axes[i // 5, i % 5]
    
    # Get a test sample
    idx = np.random.randint(len(x_test))
    img = x_test[idx:idx+1]
    true_label = y_test[idx]
    
    # Apply fixed mask and create two-channel input
    masked_input = apply_fixed_mask_to_dataset(img, fixed_mask)
    
    # Predict
    pred = model_fixed.predict(masked_input, verbose=0)
    pred_label = np.argmax(pred)
    confidence = np.max(pred) * 100
    
    # Display masked image (first channel only)
    ax.imshow(masked_input[0, :, :, 0], cmap='gray')
    
    # Color-code title based on correctness
    color = 'green' if pred_label == true_label else 'red'
    ax.set_title(f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.1f}%', 
                 fontsize=10, color=color, fontweight='bold')
    ax.axis('off')

plt.suptitle('Model Predictions on Fixed 25% Masked Images\n(Green = Correct, Red = Wrong)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('fixed_mask_predictions.png', dpi=150, bbox_inches='tight')
print("Predictions visualization saved to: fixed_mask_predictions.png")
plt.show()

# Save the model and mask
model_path = 'fixedMask.keras'
model_fixed.save(model_path)
print(f"\nFixed-mask model saved to: {model_path}")

# Save the fixed mask for later use in comparisons
mask_path = 'fixed_mask_25percent.npy'
np.save(mask_path, fixed_mask)
print(f"Fixed mask saved to: {mask_path}")

print("\n" + "="*70)
print("ANALYSIS: Model Performance on Fixed 25% Subset")
print("="*70)
print(f"Total pixels per image: 784")
print(f"Visible pixels (25%): {int(np.sum(fixed_mask))}")
print(f"Hidden pixels (75%): {784 - int(np.sum(fixed_mask))}")
print(f"Test accuracy with only 25% of pixels: {test_accuracy*100:.2f}%")
print("\nThis model approximates P(Y | X_S_fixed) for a single fixed subset S_fixed")
print("It serves as a benchmark for the mask-aware model's performance on this subset.")
