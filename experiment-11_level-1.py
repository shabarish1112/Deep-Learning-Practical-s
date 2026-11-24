import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Step 1: Load MNIST dataset
tf.keras.utils.set_random_seed(42)

(x_train_full, y_train_full), (x_test_full, y_test_full) = tf.keras.datasets.mnist.load_data()

# Step 2: Select subset
x_train = x_train_full[:20000].astype("float32")
y_train = y_train_full[:20000]

x_test = x_test_full[:5000].astype("float32")
y_test = y_test_full[:5000]

# Step 3: Normalize data
x_train /= 255.0
x_test /= 255.0

print("Training shape:", x_train.shape)
print("Testing shape:", x_test.shape)

# Step 4: Build Bidirectional RNN
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(64)),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Step 5: Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Step 6: Train model
start = time.time()

history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=5,
    batch_size=128
)

training_time = time.time() - start

# Step 7: Evaluate model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print("\nBidirectional RNN Performance")
print("Test Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy, 4))
print("Training Time:", round(training_time, 2), "seconds")

# Step 8: Classification report
probabilities = model.predict(x_test, verbose=0)
y_pred = np.argmax(probabilities, axis=1)

print("\nClassification Report")
print(classification_report(y_test, y_pred, digits=4))

# Step 9: Plot accuracy
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Bidirectional RNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Step 10: Confusion matrix
cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot(cmap="Blues")

plt.title("MNIST Confusion Matrix")
plt.show()