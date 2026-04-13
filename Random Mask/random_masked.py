import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load and preprocess MNIST dataset
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape to add channel dimension (28x28x1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"Training samples: {x_train.shape[0]}")
print(f"Test samples: {x_test.shape[0]}")

# Convert labels to categorical
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)


def generate_random_mask(shape=(28, 28), min_visibility=0.10, max_visibility=0.90):
    """Generate random mask with random visibility between min and max."""
    total_pixels = shape[0] * shape[1]
    visibility = np.random.uniform(min_visibility, max_visibility)
    num_visible = int(total_pixels * visibility)
    
    mask_flat = np.zeros(total_pixels, dtype="float32")
    visible_indices = np.random.choice(total_pixels, num_visible, replace=False)
    mask_flat[visible_indices] = 1.0
    
    mask = mask_flat.reshape(shape)
    mask = np.expand_dims(mask, -1)
    return mask


def generate_mask_at_visibility(shape=(28, 28), visibility=0.50):
    """Generate mask at specific visibility level."""
    total_pixels = shape[0] * shape[1]
    num_visible = int(total_pixels * visibility)
    
    mask_flat = np.zeros(total_pixels, dtype="float32")
    visible_indices = np.random.choice(total_pixels, num_visible, replace=False)
    mask_flat[visible_indices] = 1.0
    
    mask = mask_flat.reshape(shape)
    mask = np.expand_dims(mask, -1)
    return mask


def create_masked_batch(images, masks):
    """Create two-channel input: (X ⊙ M, M)"""
    masked_images = images * masks
    two_channel_input = np.concatenate([masked_images, masks], axis=-1)
    return two_channel_input


class VariableVisibilityMaskGenerator(keras.utils.Sequence):
    """Data generator with variable visibility masks (10-90%)."""
    def __init__(self, x, y, batch_size=128, min_vis=0.10, max_vis=0.90):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.min_vis = min_vis
        self.max_vis = max_vis
        self.indices = np.arange(len(x))
        
    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.x))
        batch_indices = self.indices[start_idx:end_idx]
        
        batch_x = self.x[batch_indices]
        batch_y = self.y[batch_indices]
        
        batch_masks = np.array([
            generate_random_mask(min_visibility=self.min_vis, 
                               max_visibility=self.max_vis)
            for _ in range(len(batch_x))
        ])
        
        batch_input = create_masked_batch(batch_x, batch_masks)
        return batch_input, batch_y
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)


def create_mask_aware_model():
    """Create CNN model for mask-aware training."""
    model = keras.Sequential([
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
# TRAIN MODEL
# ============================================================================

print("\n" + "="*80)
print("TRAINING: Mask-Aware Model with Variable Visibility (10% - 90%)")
print("="*80)

train_generator = VariableVisibilityMaskGenerator(
    x_train, y_train_cat, batch_size=128, min_vis=0.10, max_vis=0.90
)

# Create validation set
val_size = int(0.1 * len(x_train))
val_indices = np.random.choice(len(x_train), val_size, replace=False)
x_val = x_train[val_indices]
y_val = y_train_cat[val_indices]
val_masks = np.array([generate_random_mask() for _ in range(len(x_val))])
x_val_masked = create_masked_batch(x_val, val_masks)

# Build and train model
model_mask = create_mask_aware_model()
model_mask.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining model...")
history = model_mask.fit(
    train_generator,
    epochs=15,
    validation_data=(x_val_masked, y_val),
    verbose=1
)

print("\nTraining complete!")


# ============================================================================
# IMAGE 1: ACCURACY OVER TIME
# ============================================================================

print("\n" + "="*80)
print("Creating Image 1: Accuracy Over Time")
print("="*80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot accuracy
epochs = range(1, len(history.history['accuracy']) + 1)
ax1.plot(epochs, history.history['accuracy'], 'b-o', linewidth=2, 
         markersize=6, label='Training Accuracy')
ax1.plot(epochs, history.history['val_accuracy'], 'r-s', linewidth=2, 
         markersize=6, label='Validation Accuracy')
ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
ax1.set_title('Model Accuracy Over Training', fontsize=15, fontweight='bold')
ax1.legend(fontsize=11, loc='lower right')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim([0, 1])

# Plot loss
ax2.plot(epochs, history.history['loss'], 'b-o', linewidth=2, 
         markersize=6, label='Training Loss')
ax2.plot(epochs, history.history['val_loss'], 'r-s', linewidth=2, 
         markersize=6, label='Validation Loss')
ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=13, fontweight='bold')
ax2.set_title('Model Loss Over Training', fontsize=15, fontweight='bold')
ax2.legend(fontsize=11, loc='upper right')
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('1_accuracy_over_time.png',
            dpi=200, bbox_inches='tight')
print("✓ Saved: 1_accuracy_over_time.png")
plt.close()


# ============================================================================
# IMAGE 2: TYPES OF MASKS CREATED
# ============================================================================

print("\n" + "="*80)
print("Creating Image 2: Types of Masks Created")
print("="*80)

visibility_levels = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
num_examples = 5

fig = plt.figure(figsize=(18, 10))
gs = GridSpec(num_examples, len(visibility_levels), figure=fig, hspace=0.3, wspace=0.1)

for row in range(num_examples):
    for col, vis in enumerate(visibility_levels):
        mask = generate_mask_at_visibility(visibility=vis)
        
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(mask[:, :, 0], cmap='gray', vmin=0, vmax=1)
        
        # Only add title to top row
        if row == 0:
            ax.set_title(f'{int(vis*100)}%', fontsize=12, fontweight='bold')
        
        ax.axis('off')

plt.suptitle('Random Binary Masks at Different Visibility Levels\n(Black = Hidden, White = Visible)', 
             fontsize=16, fontweight='bold', y=0.98)
plt.savefig('2_types_of_masks.png',
            dpi=200, bbox_inches='tight')
print("✓ Saved: 2_types_of_masks.png")
plt.close()


# ============================================================================
# IMAGE 3: MASKED IMAGES VS ORIGINAL
# ============================================================================

print("\n" + "="*80)
print("Creating Image 3: Masked Images vs Original")
print("="*80)

# Get diverse sample images
sample_indices = [0, 10, 100, 500, 1000, 2000, 3000, 5000]
sample_images = x_test[sample_indices]
sample_labels = y_test[sample_indices]

visibility_levels = [0.10, 0.30, 0.50, 0.70, 0.90]

fig = plt.figure(figsize=(18, 14))
gs = GridSpec(len(sample_indices), len(visibility_levels) + 1, 
              figure=fig, hspace=0.3, wspace=0.15)

for i, (img, label) in enumerate(zip(sample_images, sample_labels)):
    # Show original image
    ax = fig.add_subplot(gs[i, 0])
    ax.imshow(img[:, :, 0], cmap='gray')
    ax.set_title(f'Original\nDigit: {label}', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Show masked versions
    for j, vis in enumerate(visibility_levels):
        ax = fig.add_subplot(gs[i, j + 1])
        mask = generate_mask_at_visibility(visibility=vis)
        masked_img = img * mask
        ax.imshow(masked_img[:, :, 0], cmap='gray')
        
        # Only add title to top row
        if i == 0:
            ax.set_title(f'{int(vis*100)}% Visible', fontsize=11, fontweight='bold')
        
        ax.axis('off')

plt.suptitle('Original MNIST Digits vs. Masked Versions at Different Visibility Levels', 
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig('3_masked_vs_original.png',
            dpi=200, bbox_inches='tight')
print("✓ Saved: 3_masked_vs_original.png")
plt.close()


# ============================================================================
# SAVE MODEL
# ============================================================================

model_path = 'random_mask.keras'
model_mask.save(model_path)
print(f"\n✓ Model saved to: {model_path}")
