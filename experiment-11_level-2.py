import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 1: Set random seed
tf.keras.utils.set_random_seed(42)

# Step 2: Set parameters
vocab_size = 10000
max_length = 200
embedding_dim = 64
rnn_units = 32
epochs = 3
batch_size = 128

# Step 3: Load IMDB dataset
(x_train_full, y_train_full), (x_test_full, y_test_full) = tf.keras.datasets.imdb.load_data(
    num_words=vocab_size
)

# Step 4: Select and pad data
x_train = pad_sequences(x_train_full[:15000], maxlen=max_length, padding="post")
x_test = pad_sequences(x_test_full[:5000], maxlen=max_length, padding="post")

y_train = np.array(y_train_full[:15000])
y_test = np.array(y_test_full[:5000])

print("Training shape:", x_train.shape)
print("Testing shape:", x_test.shape)

# Step 5: Build RNN model
def build_model(bidirectional=False):

    if bidirectional:
        rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.SimpleRNN(rnn_units)
        )
    else:
        rnn = tf.keras.layers.SimpleRNN(rnn_units)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_length,)),
        tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True),
        rnn,
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# Step 6: Train and evaluate both models
results = []

for name, bidirectional in [
    ("Simple RNN", False),
    ("Bidirectional RNN", True)
]:

    print("\nTraining:", name)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    model = build_model(bidirectional)

    start = time.time()

    history = model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    training_time = time.time() - start

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    pred = (model.predict(x_test, verbose=0).ravel() >= 0.5).astype(int)

    results.append({
        "Model": name,
        "Test Loss": loss,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1 Score": f1_score(y_test, pred, zero_division=0),
        "Training Time": training_time,
        "Validation Accuracy": max(history.history["val_accuracy"])
    })

# Step 7: Display comparison table
df = pd.DataFrame(results)

print("\nSimple RNN vs Bidirectional RNN")
print(df.round(4).to_string(index=False))

# Step 8: Plot comparison
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(9, 5),
    ylim=(0, 1),
    rot=0
)

plt.title("IMDB Model Comparison")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.show()

# Step 9: Find best model
best_model = df.loc[df["F1 Score"].idxmax(), "Model"]

print("\nBest Model:", best_model)