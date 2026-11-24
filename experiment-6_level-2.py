
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# Step 1: Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Step 2: Normalize data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Step 3: Build Deep CNN
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Step 4: Train Deep CNN
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1
)

test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Deep CNN Accuracy:", test_accuracy)

# Step 5: Build Shallow CNN
shallow_model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(2),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

shallow_model.summary()

# Step 6: Compile Shallow CNN
shallow_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Step 7: Train Shallow CNN
shallow_history = shallow_model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1
)

shallow_loss, shallow_accuracy = shallow_model.evaluate(x_test, y_test)

print("Shallow CNN Accuracy:", shallow_accuracy)
print("Deep CNN Accuracy:", test_accuracy)

# Step 8: Compare both models
plt.figure(figsize=(8, 5))

plt.plot(shallow_history.history["val_accuracy"], label="Shallow CNN")
plt.plot(history.history["val_accuracy"], label="Deep CNN")

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Shallow CNN vs Deep CNN")
plt.legend()
plt.show()

# Step 9: Predict Fashion items using Deep CNN
predictions = model.predict(x_test)
predicted_labels = predictions.argmax(axis=1)

# Step 10: Display predictions
plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap="gray")

    plt.title(
        f"Actual: {class_names[y_test[i]]}\n"
        f"Pred: {class_names[predicted_labels[i]]}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()
