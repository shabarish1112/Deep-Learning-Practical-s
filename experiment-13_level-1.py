import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Step 1: Load MNIST dataset
tf.keras.utils.set_random_seed(42)

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Step 2: Normalize and flatten data
x_train = x_train[:20000].astype("float32").reshape(-1, 784) / 255.0
x_test = x_test[:5000].astype("float32").reshape(-1, 784) / 255.0

print("Training shape:", x_train.shape)
print("Testing shape:", x_test.shape)

# Step 3: Build Autoencoder
inputs = tf.keras.layers.Input(shape=(784,))

x = tf.keras.layers.Dense(128, activation="relu")(inputs)
code = tf.keras.layers.Dense(32, activation="relu")(x)
x = tf.keras.layers.Dense(128, activation="relu")(code)

outputs = tf.keras.layers.Dense(784, activation="sigmoid")(x)

autoencoder = tf.keras.Model(inputs, outputs)
encoder = tf.keras.Model(inputs, code)

# Step 4: Compile model
autoencoder.compile(optimizer="adam", loss="mse")

autoencoder.summary()

# Step 5: Train model
start = time.time()

history = autoencoder.fit(
    x_train, x_train,
    validation_split=0.1,
    epochs=20,
    batch_size=256,
    shuffle=True
)

training_time = time.time() - start

# Step 6: Evaluate model
test_mse = autoencoder.evaluate(x_test, x_test, verbose=0)

print("\nAutoencoder Performance")
print("Test MSE:", round(test_mse, 5))
print("Compression Ratio:", 784 / 32, ":1")
print("Training Time:", round(training_time, 2), "seconds")

# Step 7: Plot loss
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Autoencoder Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.show()

# Step 8: Display original and reconstructed images
reconstructed = autoencoder.predict(x_test[:10])

fig, axes = plt.subplots(2, 10, figsize=(14, 3))

for i in range(10):
    axes[0, i].imshow(x_test[i].reshape(28, 28), cmap="gray")
    axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].axis("off")

axes[0, 0].set_title("Original")
axes[1, 0].set_title("Reconstructed")

plt.show()
