import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Load and preprocess MNIST
tf.keras.utils.set_random_seed(42)

(x_train_full, y_train_full), (x_test_full, y_test_full) = tf.keras.datasets.mnist.load_data()

x_train = x_train_full[:20000].astype("float32").reshape(-1, 784) / 255.0
x_test = x_test_full[:5000].astype("float32").reshape(-1, 784) / 255.0

y_train = y_train_full[:20000]
y_test = y_test_full[:5000]

# Step 2: Build Autoencoder
def build_autoencoder(latent_dim):
    inputs = tf.keras.layers.Input(shape=(784,))
    x = tf.keras.layers.Dense(128, activation="relu")(inputs)
    code = tf.keras.layers.Dense(latent_dim)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(code)
    outputs = tf.keras.layers.Dense(784, activation="sigmoid")(x)

    autoencoder = tf.keras.Model(inputs, outputs)
    encoder = tf.keras.Model(inputs, code)

    autoencoder.compile(optimizer="adam", loss="mse")

    return autoencoder, encoder

# Step 3: Compare Autoencoder and PCA
latent_dims = [2, 8, 32]
results = []
codes_2d = {}

for d in latent_dims:

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    # Autoencoder
    autoencoder, encoder = build_autoencoder(d)

    start = time.time()

    autoencoder.fit(
        x_train, x_train,
        epochs=15,
        batch_size=256,
        verbose=0
    )

    ae_time = time.time() - start

    ae_reconstructed = autoencoder.predict(x_test, verbose=0)
    ae_mse = np.mean((x_test - ae_reconstructed) ** 2)

    ae_train_code = encoder.predict(x_train, verbose=0)
    ae_test_code = encoder.predict(x_test, verbose=0)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(ae_train_code, y_train)
    ae_acc = knn.score(ae_test_code, y_test)

    # PCA
    pca = PCA(n_components=d, random_state=42)
    pca.fit(x_train)

    pca_reconstructed = pca.inverse_transform(pca.transform(x_test))
    pca_mse = np.mean((x_test - pca_reconstructed) ** 2)

    pca_train = pca.transform(x_train)
    pca_test = pca.transform(x_test)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(pca_train, y_train)
    pca_acc = knn.score(pca_test, y_test)

    results.append({
        "Dimensions": d,
        "AE MSE": ae_mse,
        "PCA MSE": pca_mse,
        "AE Accuracy": ae_acc,
        "PCA Accuracy": pca_acc,
        "AE Time": ae_time
    })

    if d == 2:
        codes_2d["Autoencoder (2D)"] = ae_test_code
        codes_2d["PCA (2D)"] = pca_test

# Step 4: Display comparison table
df = pd.DataFrame(results)

print("\nAutoencoder vs PCA")
print(df.round(4).to_string(index=False))

# Step 5: Plot reconstruction error
df.set_index("Dimensions")[["AE MSE", "PCA MSE"]].plot(
    kind="bar", figsize=(7, 4)
)

plt.title("Reconstruction Error vs Dimensions")
plt.ylabel("Test MSE")
plt.show()

# Step 6: Visualize 2D representations
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, (name, z) in zip(axes, codes_2d.items()):
    sc = ax.scatter(
        z[:, 0], z[:, 1],
        c=y_test, cmap="tab10", s=3
    )
    ax.set_title(name)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

fig.colorbar(sc, ax=axes, label="Digit Class")
plt.show()
