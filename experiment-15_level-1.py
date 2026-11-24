import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Step 1: Load and preprocess MNIST
tf.keras.utils.set_random_seed(42)

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train[:20000].astype("float32").reshape(-1, 784) / 255.0
x_test = x_test[:5000].astype("float32").reshape(-1, 784) / 255.0

print("Training shape:", x_train.shape)
print("Testing shape:", x_test.shape)

# Step 2: Build Sparse Autoencoder
inputs = tf.keras.layers.Input(shape=(784,))

hidden = tf.keras.layers.Dense(
    128,
    activation="relu",
    activity_regularizer=tf.keras.regularizers.l1(1e-4)
)(inputs)

outputs = tf.keras.layers.Dense(784, activation="sigmoid")(hidden)

sparse_ae = tf.keras.Model(inputs, outputs)
encoder = tf.keras.Model(inputs, hidden)

# Step 3: Compile model
sparse_ae.compile(optimizer="adam", loss="mse")
sparse_ae.summary()

# Step 4: Train model
start = time.time()

history = sparse_ae.fit(
    x_train, x_train,
    validation_split=0.1,
    epochs=20,
    batch_size=256,
    shuffle=True
)

training_time = time.time() - start

# Step 5: Evaluate performance
reconstructed_test = sparse_ae.predict(x_test, verbose=0)
test_mse = np.mean((x_test - reconstructed_test) ** 2)

activations = encoder.predict(x_test, verbose=0)

inactive = np.mean(activations < 0.01)
active = np.mean(np.sum(activations >= 0.01, axis=1))

print("\nSparse Autoencoder Performance")
print("Test MSE:", round(float(test_mse), 5))
print("Mean activation:", round(float(activations.mean()), 4))
print("Inactive fraction:", round(float(inactive), 4))
print("Average active units:", round(float(active), 1), "of 128")
print("Training time:", round(training_time, 2), "seconds")

# Step 6: Plot training loss
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Sparse Autoencoder Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Step 7: Plot hidden activations
plt.hist(activations.ravel(), bins=60, log=True)
plt.title("Hidden Activation Distribution")
plt.xlabel("Activation Value")
plt.ylabel("Count")
plt.show()

# Step 8: Display original and reconstructed images
reconstructed = sparse_ae.predict(x_test[:10], verbose=0)

fig, axes = plt.subplots(2, 10, figsize=(14, 3))

for i in range(10):
    axes[0, i].imshow(x_test[i].reshape(28, 28), cmap="gray")
    axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap="gray")

    axes[0, i].axis("off")
    axes[1, i].axis("off")

axes[0, 0].set_title("Original")
axes[1, 0].set_title("Reconstructed")

plt.show()
