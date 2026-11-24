import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# Step 1: Load IMDB dataset
vocab_size = 10000

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
    num_words=vocab_size
)

print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))

# Step 2: Understand the data
print("First review:", x_train[0])
print("Label:", y_train[0])
print("Review length:", len(x_train[0]))

# Step 3: Pad sequences
max_length = 200

x_train = tf.keras.utils.pad_sequences(x_train, maxlen=max_length)
x_test = tf.keras.utils.pad_sequences(x_test, maxlen=max_length)

print("Training shape:", x_train.shape)
print("Testing shape:", x_test.shape)

# Step 4: Build 1D CNN model
model = models.Sequential([
    layers.Input(shape=(max_length,)),
    layers.Embedding(vocab_size, 64),
    layers.Conv1D(64, 5, activation="relu"),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(128, 5, activation="relu"),
    layers.GlobalMaxPooling1D(),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.summary()

# Step 5: Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Step 6: Train model
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# Step 7: Evaluate model
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Step 8: Plot accuracy
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("CNN Text Classification - Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Step 9: Plot loss
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("CNN Text Classification - Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
