import os
import zipfile
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, TextVectorization
from tensorflow.keras.models import Model

# Step 1: Configuration
tf.keras.utils.set_random_seed(42)

MAX_PAIRS = 3000
MAX_VOCAB = 4000
MAX_LEN = 10
EMBED_DIM = 32
LATENT_DIM = 64
BATCH_SIZE = 64
EPOCHS = 5

# Step 2: Download dataset
url = "https://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"

zip_path = tf.keras.utils.get_file(
    "spa-eng.zip", url, cache_subdir="datasets"
)

cache_dir = os.path.dirname(zip_path)

with zipfile.ZipFile(zip_path, "r") as archive:
    member = next(name for name in archive.namelist() if name.endswith("spa.txt"))
    archive.extract(member, cache_dir)

text_path = os.path.join(cache_dir, *member.split("/"))

# Step 3: Load English-Spanish sentences
pairs = []

with open(text_path, encoding="utf-8") as file:
    for line in file:
        fields = line.strip().split("\t")

        if len(fields) >= 2:
            en, es = fields[0], fields[1]

            if len(en.split()) <= MAX_LEN and len(es.split()) < MAX_LEN:
                pairs.append((en, es))

np.random.default_rng(42).shuffle(pairs)
pairs = pairs[:MAX_PAIRS]

english = [p[0] for p in pairs]
spanish = [p[1] for p in pairs]

decoder_input = ["startseq " + s for s in spanish]
decoder_target = [s + " endseq" for s in spanish]

# Step 4: Split dataset
train_en, test_en, train_es, test_es, train_dec, test_dec, train_tar, test_tar = train_test_split(
    english, spanish, decoder_input, decoder_target,
    test_size=0.1, random_state=42
)

# Step 5: Convert text into sequences
en_vectorizer = TextVectorization(
    max_tokens=MAX_VOCAB,
    output_mode="int",
    output_sequence_length=MAX_LEN
)

es_vectorizer = TextVectorization(
    max_tokens=MAX_VOCAB,
    output_mode="int",
    output_sequence_length=MAX_LEN
)

en_vectorizer.adapt(np.array(train_en))
es_vectorizer.adapt(np.array(train_dec + train_tar))

en_vocab = len(en_vectorizer.get_vocabulary())
es_vocab = es_vectorizer.get_vocabulary()

word_to_id = {w: i for i, w in enumerate(es_vocab)}
START_ID = word_to_id["startseq"]
END_ID = word_to_id["endseq"]

x_train = en_vectorizer(np.array(train_en)).numpy()
d_train = es_vectorizer(np.array(train_dec)).numpy()
y_train = es_vectorizer(np.array(train_tar)).numpy()

x_test = en_vectorizer(np.array(test_en)).numpy()
d_test = es_vectorizer(np.array(test_dec)).numpy()
y_test = es_vectorizer(np.array(test_tar)).numpy()

train_weights = (y_train != 0).astype("float32")
test_weights = (y_test != 0).astype("float32")

# Step 6: Build Encoder
encoder_input = Input(shape=(MAX_LEN,))
encoder_embedding = Embedding(en_vocab, EMBED_DIM, mask_zero=True)
encoder_gru = GRU(LATENT_DIM, return_state=True)

embedded = encoder_embedding(encoder_input)
_, encoder_state = encoder_gru(embedded)

# Step 7: Build Decoder
decoder_input = Input(shape=(MAX_LEN,))

decoder_embedding = Embedding(es_vocab.__len__(), EMBED_DIM, mask_zero=True)
decoder_gru = GRU(LATENT_DIM, return_sequences=True, return_state=True)
decoder_dense = Dense(len(es_vocab), activation="softmax")

decoder_embedded = decoder_embedding(decoder_input)

decoder_output, _ = decoder_gru(
    decoder_embedded, initial_state=encoder_state
)

output = decoder_dense(decoder_output)

model = Model([encoder_input, decoder_input], output)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    weighted_metrics=["sparse_categorical_accuracy"]
)

# Step 8: Train model
import time

start = time.time()

model.fit(
    [x_train, d_train], y_train,
    sample_weight=train_weights,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

print("Training time:", round(time.time() - start, 2), "seconds")

# Step 9: Evaluate model
evaluation = model.evaluate(
    [x_test, d_test], y_test,
    sample_weight=test_weights,
    return_dict=True,
    verbose=0
)

print("Test Loss:", round(evaluation["loss"], 4))
print("Test Accuracy:", round(evaluation["sparse_categorical_accuracy"], 4))

# Step 10: Create inference models
encoder_model = Model(encoder_input, encoder_state)

current_token = Input(shape=(1,))
state_input = Input(shape=(LATENT_DIM,))

current_embedded = decoder_embedding(current_token)

step_output, next_state = decoder_gru(
    current_embedded, initial_state=state_input
)

step_output = decoder_dense(step_output)

decoder_model = Model(
    [current_token, state_input],
    [step_output, next_state]
)

# Step 11: Translate English sentences
def translate(sentence):
    ids = en_vectorizer(np.array([sentence])).numpy()
    state = encoder_model.predict(ids, verbose=0)

    token = START_ID
    result = []

    for _ in range(MAX_LEN):
        prediction, state = decoder_model.predict(
            [np.array([[token]]), state], verbose=0
        )

        token = int(np.argmax(prediction[0, 0]))

        if token == 0 or token == END_ID:
            break

        if token != START_ID:
            result.append(es_vocab[token])

    return " ".join(result)

# Display sample translations
print("\nEnglish to Spanish Translation")

for i in range(min(5, len(test_en))):
    print("\nEnglish:", test_en[i])
    print("Actual:", test_es[i])
    print("Predicted:", translate(test_en[i]))