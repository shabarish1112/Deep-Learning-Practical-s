import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.linear_model import LogisticRegression

# Step 1: Load and preprocess MNIST
tf.keras.utils.set_random_seed(42)

(x_train_full, y_train_full), (x_test_full, y_test_full) = tf.keras.datasets.mnist.load_data()

x_train = x_train_full[:20000].astype("float32").reshape(-1, 784) / 255.0
x_test = x_test_full[:5000].astype("float32").reshape(-1, 784) / 255.0

y_train = y_train_full[:20000]
y_test = y_test_full[:5000]

# Step 2: Build autoencoder
def build_model(weight):
    regularizer = tf.keras.regularizers.l1(weight) if weight > 0 else None

    inputs = tf.keras.layers.Input(shape=(784,))
    hidden = tf.keras.layers.Dense(
        128, activation="relu",
        activity_regularizer=regularizer,
        name="hidden"
    )(inputs)

    outputs = tf.keras.layers.Dense(784, activation="sigmoid")(hidden)

    autoencoder = tf.keras.Model(inputs, outputs)
    encoder = tf.keras.Model(inputs, hidden)

    autoencoder.compile(optimizer="adam", loss="mse")

    return autoencoder, encoder

# Step 3: Train different models
weights = {
    "PlainAE": 0.0,
    "Sparse(1e-5)": 1e-5,
    "Sparse(1e-4)": 1e-4,
    "Sparse(1e-3)": 1e-3
}

models = {}
features = {}

for name, w in weights.items():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    autoencoder, encoder = build_model(w)

    autoencoder.fit(
        x_train, x_train,
        epochs=20,
        batch_size=256,
        verbose=0
    )

    models[name] = (autoencoder, encoder)

    features[name] = (
        encoder.predict(x_train, verbose=0),
        encoder.predict(x_test, verbose=0)
    )

# Step 4: Classify extracted features
def classify(train_x, test_x):
    clf = LogisticRegression(max_iter=300)
    clf.fit(train_x[:10000], y_train[:10000])
    return clf.score(test_x, y_test)

results = [{
    "Features": "Raw Pixels",
    "MSE": np.nan,
    "Inactive Units": np.nan,
    "Accuracy": classify(x_train, x_test)
}]

for name, (ae, encoder) in models.items():
    h_train, h_test = features[name]

    reconstructed = ae.predict(x_test, verbose=0)
    mse = np.mean((x_test - reconstructed) ** 2)

    results.append({
        "Features": name,
        "MSE": mse,
        "Inactive Units": np.mean(h_test < 0.01),
        "Accuracy": classify(h_train, h_test)
    })

df = pd.DataFrame(results)

print("\nFeature Extraction Comparison")
print(df.round(4).to_string(index=False))

# Step 5: Visualize learned features
def plot_filters(name):
    weights_matrix = models[name][1].get_layer("hidden").get_weights()[0]

    fig, axes = plt.subplots(4, 8, figsize=(10, 5))

    for i, ax in enumerate(axes.ravel()):
        ax.imshow(weights_matrix[:, i].reshape(28, 28), cmap="gray")
        ax.axis("off")

    plt.suptitle("Learned Features: " + name)
    plt.show()

plot_filters("PlainAE")
plot_filters("Sparse(1e-4)")

# Step 6: Compare hidden activations
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for ax, name in zip(axes, ["PlainAE", "Sparse(1e-4)"]):
    ax.bar(range(128), features[name][1][0])
    ax.set_title(name)
    ax.set_xlabel("Hidden Unit")

axes[0].set_ylabel("Activation")

plt.tight_layout()
plt.show()
