import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

# Step 1: Load, subset, normalize and flatten MNIST
(x_train_full, _), (x_test_full, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train_full[:20000].astype("float32").reshape(-1, 784) / 255.0
x_test = x_test_full[:5000].astype("float32").reshape(-1, 784) / 255.0

# Step 2: Add Gaussian noise
noise_factor = 0.5
rng = np.random.default_rng(42)

def add_noise(images, factor):
    noisy = images + factor * rng.normal(size=images.shape)
    return np.clip(noisy, 0.0, 1.0).astype("float32")

x_train_noisy = add_noise(x_train, noise_factor)
x_test_noisy = add_noise(x_test, noise_factor)

print("Clean training shape:", x_train.shape)
print("Noisy training shape:", x_train_noisy.shape)

# Step 3: Build the denoising autoencoder
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(784, activation="sigmoid")
])

# Step 4: Compile the model
model.compile(optimizer="adam", loss="mse")
model.summary()

# Step 5: Train the model
start_time = time.perf_counter()

history = model.fit(
    x_train_noisy,
    x_train,
    validation_split=0.1,
    epochs=20,
    batch_size=256,
    shuffle=True,
    verbose=1
)

training_time = time.perf_counter() - start_time

# Step 6: Evaluate the model
test_mse = model.evaluate(x_test_noisy, x_test, verbose=0)
noisy_mse = np.mean((x_test_noisy - x_test) ** 2)

print("\nDenoising Autoencoder Performance")
print("MSE of noisy input vs. clean image:", round(noisy_mse, 5))
print("MSE of denoised output vs. clean image:", round(float(test_mse), 5))
print("Training time:", round(training_time, 2), "seconds")

# Step 7: Plot training and validation loss
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.title("Denoising Autoencoder: Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.legend()
plt.tight_layout()
plt.show()

# Step 8: Display clean, noisy and denoised images
denoised = model.predict(x_test_noisy[:10], verbose=0)

rows = [
    ("Clean", x_test),
    ("Noisy", x_test_noisy),
    ("Denoised", denoised)
]

fig, axes = plt.subplots(3, 10, figsize=(14, 4.5))

for r, (title, images) in enumerate(rows):
    for i in range(10):
        axes[r, i].imshow(images[i].reshape(28, 28), cmap="gray")
        axes[r, i].axis("off")

    axes[r, 0].set_title(title, loc="left", fontsize=9)

plt.tight_layout()
plt.show()