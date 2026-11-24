
# Experiment: Iris Flower Classification using ANN

# Step 1: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

import tensorflow as tf

# Make experiment reproducible
tf.keras.utils.set_random_seed(42)


# Step 2: Load the Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

print("Input shape:", X.shape)
print("Target shape:", y.shape)


# Step 3: Display the classes
print("\nClasses:", iris.target_names)


# Step 4: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# Step 5: Normalize the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Step 6: Build the Neural Network
model = Sequential([
    Input(shape=(4,)),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(3, activation="softmax")
])


# Step 7: Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# Step 8: Display the model
model.summary()


# Step 9: Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)


# Step 10: Evaluate the model
test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=0
)

print("\nModel Evaluation")
print("Test Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy, 4))


# Step 11: Make predictions
y_prob = model.predict(X_test, verbose=0)

y_pred = np.argmax(y_prob, axis=1)

print("\nPredicted Classes:", y_pred)
print("Actual Classes:", y_test)


# Step 12: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nClassification Accuracy:", round(accuracy, 4))


# Step 13: Display classification report
print("\nClassification Report")

print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))


# Step 14: Display confusion matrix
cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

display.plot(cmap="Blues", values_format="d")

plt.title("Iris Classification Confusion Matrix")
plt.tight_layout()
plt.show()


# Step 15: Plot Training and Validation Accuracy
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid()
plt.show()


# Step 16: Plot Training and Validation Loss
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()
plt.show()


# Step 17: Predict a new Iris flower
# Features:
# Sepal length, Sepal width, Petal length, Petal width

new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])

# Normalize using the same scaler
new_flower_scaled = scaler.transform(new_flower)

# Predict
prediction = model.predict(new_flower_scaled, verbose=0)

predicted_class = np.argmax(prediction, axis=1)[0]

print("\nNew Flower Prediction")
print("Predicted Class:", iris.target_names[predicted_class])
print("Prediction Probability:", round(float(np.max(prediction)), 4))


# Step 18: Display predicted probabilities
plt.figure(figsize=(7, 4))

plt.bar(
    iris.target_names,
    prediction[0],
    color=["blue", "orange", "green"]
)

plt.xlabel("Iris Flower Class")
plt.ylabel("Prediction Probability")
plt.title("Predicted Class Probabilities")
plt.ylim(0, 1)
plt.show()
