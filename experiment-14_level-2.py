import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Step 1: Load and preprocess MNIST
tf.keras.utils.set_random_seed(42)

(x_train_full, _), (x_test_full, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train_full[:20000].astype("float32").reshape(-1, 784) / 255.0
x_test = x_test_full[:5000].astype("float32").reshape(-1, 784) / 255.0

# Step 2: Add noise and define metrics
rng = np.random.default_rng(42)

def add_noise(images, factor):
    noisy = images + factor * rng.normal(size=images.shape)
    return np.clip(noisy, 0, 1).astype("float32")

def mse(a, b):
    return np.mean((a - b) ** 2)

def psnr(a, b):
    return 10 * np.log10(1.0 / mse(a, b))

def ssim(a, b):
    a = a.reshape(-1, 28, 28, 1)
    b = b.reshape(-1, 28, 28, 1)
    return float(tf.reduce_mean(tf.image.ssim(a, b, max_val=1.0)))

# Step 3: Build autoencoder
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(784, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="mse")
    return model

# Step 4: Train both models
x_train_noisy = add_noise(x_train, 0.5)
training_times = {}

# Standard Autoencoder
tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(42)

standard_ae = build_model()
start = time.time()

standard_ae.fit(x_train, x_train, epochs=20, batch_size=256, verbose=0)

training_times["Standard AE"] = time.time() - start

# Denoising Autoencoder
tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(42)

denoising_ae = build_model()
start = time.time()

denoising_ae.fit(x_train_noisy, x_train, epochs=20, batch_size=256, verbose=0)

training_times["Denoising AE"] = time.time() - start

# Step 5: Evaluate both models
noise_levels = [0.2, 0.5, 0.8]
results = []
samples = {}

for level in noise_levels:

    x_noisy = add_noise(x_test, level)

    outputs = {
        "Noisy Input": x_noisy,
        "Standard AE": standard_ae.predict(x_noisy, verbose=0),
        "Denoising AE": denoising_ae.predict(x_noisy, verbose=0)
    }

    for name, recon in outputs.items():
        results.append({
            "Noise": level,
            "Model": name,
            "MSE": mse(x_test, recon),
            "PSNR": psnr(x_test, recon),
            "SSIM": ssim(x_test, recon)
        })

    samples[level] = outputs

# Display results
df = pd.DataFrame(results)

print("\nReconstruction Performance")
print(df.round(4).to_string(index=False))

print("\nTraining Time:")
print(training_times)

# Step 6: Plot MSE and PSNR
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for metric, ax in zip(["MSE", "PSNR"], axes):
    for name, group in df.groupby("Model"):
        ax.plot(group["Noise"], group[metric], marker="o", label=name)

    ax.set_xlabel("Noise Level")
    ax.set_ylabel(metric)
    ax.set_title(metric + " vs Noise")
    ax.legend()

plt.tight_layout()
plt.show()

# Step 7: Display original and reconstructed images
fig, axes = plt.subplots(4, 10, figsize=(14, 6))

rows = [("Original", x_test)] + list(samples[0.5].items())

for r, (title, images) in enumerate(rows):
    for i in range(10):
        axes[r, i].imshow(images[i].reshape(28, 28), cmap="gray")
        axes[r, i].axis("off")

    axes[r, 0].set_title(title, loc="left")

plt.tight_layout()
plt.show()
