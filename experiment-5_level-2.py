
# Experiment: CNN Pooling and Stride Variation
# Dataset: MNIST

# Step 1: Import libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models

# Make experiment reproducible
tf.keras.utils.set_random_seed(42)


# Step 2: Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training images:", x_train.shape)
print("Testing images:", x_test.shape)


# Step 3: Preprocessing
# Normalize pixel values between 0 and 1

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print("New training shape:", x_train.shape)


# Step 4: Display sample images
plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i].squeeze(), cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()


# Step 5: Build baseline CNN with Max Pooling
def build_max_pool_model():

    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# Step 6: Train baseline CNN
print("\nTraining Max Pooling CNN...")

max_pool_model = build_max_pool_model()

max_history = max_pool_model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

max_loss, max_accuracy = max_pool_model.evaluate(
    x_test, y_test, verbose=0
)

print("\nMax Pooling Accuracy:", max_accuracy)
print("Max Pooling Loss:", max_loss)


# Step 7: Experiment A - Average Pooling
avg_pool_model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.AveragePooling2D(pool_size=(2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.AveragePooling2D(pool_size=(2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

avg_pool_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining Average Pooling CNN...")

avg_history = avg_pool_model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

avg_loss, avg_accuracy = avg_pool_model.evaluate(
    x_test, y_test, verbose=0
)

print("\nAverage Pooling Accuracy:", avg_accuracy)
print("Average Pooling Loss:", avg_loss)


# Step 8: Experiment B - Convolution Stride = 2
stride2_model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(
        32, (3, 3),
        strides=(2, 2),
        activation="relu"
    ),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(
        64, (3, 3),
        strides=(2, 2),
        activation="relu"
    ),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

stride2_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining Stride 2 CNN...")

stride2_history = stride2_model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

stride2_loss, stride2_accuracy = stride2_model.evaluate(
    x_test, y_test, verbose=0
)

print("\nStride 2 Accuracy:", stride2_accuracy)
print("Stride 2 Loss:", stride2_loss)


# Step 9: Compare all three models
results = pd.DataFrame({
    "Model": [
        "Max Pooling",
        "Average Pooling",
        "Stride 2"
    ],
    "Test Accuracy": [
        max_accuracy,
        avg_accuracy,
        stride2_accuracy
    ],
    "Test Loss": [
        max_loss,
        avg_loss,
        stride2_loss
    ]
})

print("\nCNN Model Comparison")
print(results.round(4).to_string(index=False))


# Step 10: Plot accuracy comparison
plt.figure(figsize=(8, 5))

plt.bar(
    results["Model"],
    results["Test Accuracy"],
    color=["blue", "orange", "green"]
)

plt.xlabel("CNN Model")
plt.ylabel("Test Accuracy")
plt.title("Comparison of CNN Models")
plt.ylim(0, 1)
plt.show()


# Step 11: Compare validation accuracy during training
plt.figure(figsize=(9, 5))

plt.plot(
    max_history.history["val_accuracy"],
    marker="o",
    label="Max Pooling"
)

plt.plot(
    avg_history.history["val_accuracy"],
    marker="s",
    label="Average Pooling"
)

plt.plot(
    stride2_history.history["val_accuracy"],
    marker="^",
    label="Stride 2"
)

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy Comparison")
plt.legend()
plt.grid()
plt.show()


# Step 12: Compare training loss
plt.figure(figsize=(9, 5))

plt.plot(max_history.history["loss"], label="Max Pooling")
plt.plot(avg_history.history["loss"], label="Average Pooling")
plt.plot(stride2_history.history["loss"], label="Stride 2")

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Comparison")
plt.legend()
plt.grid()
plt.show()


# Step 13: Display predictions using baseline CNN
predictions = max_pool_model.predict(x_test[:10], verbose=0)

predicted_labels = np.argmax(predictions, axis=1)

plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)

    plt.imshow(x_test[i].squeeze(), cmap="gray")

    plt.title(
        f"Actual: {y_test[i]}\nPredicted: {predicted_labels[i]}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()


# Step 14: Identify the best performing model
best_index = results["Test Accuracy"].idxmax()

print("\nBest Performing Model:")
print(results.loc[best_index, "Model"])

print("Best Test Accuracy:",
      round(results.loc[best_index, "Test Accuracy"], 4))
