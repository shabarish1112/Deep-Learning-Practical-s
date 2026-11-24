import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import layers, models

# Step 1: Load IMDB dataset
tf.keras.utils.set_random_seed(42)

vocab_size = 10000
max_length = 200

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
    num_words=vocab_size
)

print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))

# Step 2: Pad sequences
x_train = tf.keras.utils.pad_sequences(x_train, maxlen=max_length)
x_test = tf.keras.utils.pad_sequences(x_test, maxlen=max_length)

print("Training shape:", x_train.shape)
print("Testing shape:", x_test.shape)

# Step 3: Build CNN model with embedding dimension 64
model = models.Sequential([
    layers.Input(shape=(max_length,)),
    layers.Embedding(vocab_size, 64),
    layers.Conv1D(64, 5, activation="relu"),
    layers.MaxPooling1D(2),
    layers.Conv1D(128, 5, activation="relu"),
    layers.GlobalMaxPooling1D(),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# Step 4: Compile and train Model 1
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history64 = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

loss64, accuracy64 = model.evaluate(x_test, y_test)
print("Embedding 64 Test Accuracy:", accuracy64)

# Step 5: Build CNN model with embedding dimension 128
model_embedding128 = models.Sequential([
    layers.Input(shape=(max_length,)),
    layers.Embedding(vocab_size, 128),
    layers.Conv1D(64, 5, activation="relu"),
    layers.MaxPooling1D(2),
    layers.Conv1D(128, 5, activation="relu"),
    layers.GlobalMaxPooling1D(),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# Step 6: Compile and train Model 2
model_embedding128.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history128 = model_embedding128.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

loss128, accuracy128 = model_embedding128.evaluate(x_test, y_test)

print("Embedding 128 Test Accuracy:", accuracy128)

# Step 7: Compare both models
results = pd.DataFrame({
    "Parameter": ["Vocabulary Size", "Maximum Length",
                  "Embedding Dimension", "Conv1D Filters",
                  "Epochs", "Test Accuracy", "Test Loss"],
    "Model 1": [10000, 200, 64, "64, 128", 5,
                round(accuracy64, 4), round(loss64, 4)],
    "Model 2": [10000, 200, 128, "64, 128", 5,
                round(accuracy128, 4), round(loss128, 4)]
})

print("\nModel Comparison")
print(results.to_string(index=False))

# Step 8: Plot accuracy comparison
plt.plot(history64.history["val_accuracy"], label="Embedding 64")
plt.plot(history128.history["val_accuracy"], label="Embedding 128")

plt.title("Validation Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Step 9: Plot loss comparison
plt.plot(history64.history["val_loss"], label="Embedding 64")
plt.plot(history128.history["val_loss"], label="Embedding 128")

plt.title("Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Step 10: Test a new review
sample_review = x_test[0]

prediction = model.predict(
    sample_review.reshape(1, -1),
    verbose=0
)

print("\nPrediction:", prediction[0][0])

if prediction[0][0] >= 0.5:
    print("Predicted Sentiment: Positive")
else:
    print("Predicted Sentiment: Negative")
