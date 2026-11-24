import os
import time
import zipfile
import requests
import numpy as np
import tensorflow as tf
import string

from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, TextVectorization
from tensorflow.keras.models import Model

# Step 1: Configuration
tf.keras.utils.set_random_seed(42)

MAX_PAIRS = 5000
MAX_VOCAB = 6000
MAX_LEN = 14
EMBED_DIM = 64
LATENT_DIM = 128
BATCH_SIZE = 64
EPOCHS = 8

# Step 2: Download French dataset
url = "https://mail.manythings.org/anki/fra-eng.zip"

folder = os.path.join(os.path.expanduser("~"), ".keras", "datasets")
os.makedirs(folder, exist_ok=True)

zip_path = os.path.join(folder, "fra-eng.zip")

if not os.path.isfile(zip_path) or not zipfile.is_zipfile(zip_path):
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=90
    )
    response.raise_for_status()

    with open(zip_path, "wb") as f:
        f.write(response.content)

with zipfile.ZipFile(zip_path, "r") as z:
    member = next(name for name in z.namelist() if name.endswith("fra.txt"))
    z.extract(member, folder)

text_path = os.path.join(folder, *member.split("/"))

# Step 3: Load sentence pairs
pairs = []

with open(text_path, encoding="utf-8") as f:
    for line in f:
        fields = line.strip().split("\t")

        if len(fields) >= 2:
            en, fr = fields[0], fields[1]

            if en and fr and len(en.split()) <= MAX_LEN and len(fr.split()) < MAX_LEN:
                pairs.append((en, fr))

np.random.default_rng(42).shuffle(pairs)
pairs = pairs[:MAX_PAIRS]

english = [p[0] for p in pairs]
french = [p[1] for p in pairs]

decoder_input = ["startseq " + s for s in french]
decoder_target = [s + " endseq" for s in french]

# Step 4: Split dataset
train_en, test_en, train_fr, test_fr, train_dec, test_dec, train_tar, test_tar = train_test_split(
    english, french, decoder_input, decoder_target,
    test_size=0.1, random_state=42
)

# Step 5: Convert text into sequences
en_vectorizer = TextVectorization(
    max_tokens=MAX_VOCAB,
    output_mode="int",
    output_sequence_length=MAX_LEN
)

fr_vectorizer = TextVectorization(
    max_tokens=MAX_VOCAB,
    output_mode="int",
    output_sequence_length=MAX_LEN
)

en_vectorizer.adapt(np.array(train_en))
fr_vectorizer.adapt(np.array(train_dec + train_tar))

fr_vocab = fr_vectorizer.get_vocabulary()
word_to_id = {w: i for i, w in enumerate(fr_vocab)}

START_ID = word_to_id["startseq"]
END_ID = word_to_id["endseq"]

x_train = en_vectorizer(np.array(train_en)).numpy()
d_train = fr_vectorizer(np.array(train_dec)).numpy()
y_train = fr_vectorizer(np.array(train_tar)).numpy()

x_test = en_vectorizer(np.array(test_en)).numpy()
d_test = fr_vectorizer(np.array(test_dec)).numpy()
y_test = fr_vectorizer(np.array(test_tar)).numpy()

train_weights = (y_train != 0).astype("float32")
test_weights = (y_test != 0).astype("float32")

# Step 6: Build Encoder
encoder_input = Input(shape=(MAX_LEN,))

encoder_embedding = Embedding(
    len(en_vectorizer.get_vocabulary()), EMBED_DIM, mask_zero=True
)

encoder_gru = GRU(LATENT_DIM, return_state=True)

_, encoder_state = encoder_gru(encoder_embedding(encoder_input))

# Step 7: Build Decoder
decoder_input = Input(shape=(MAX_LEN,))

decoder_embedding = Embedding(
    len(fr_vocab), EMBED_DIM, mask_zero=True
)

decoder_gru = GRU(
    LATENT_DIM, return_sequences=True, return_state=True
)

decoder_dense = Dense(len(fr_vocab), activation="softmax")

decoder_output, _ = decoder_gru(
    decoder_embedding(decoder_input),
    initial_state=encoder_state
)

output = decoder_dense(decoder_output)

model = Model([encoder_input, decoder_input], output)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    weighted_metrics=["sparse_categorical_accuracy"]
)

# Step 8: Train model
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

print("Test Accuracy:", round(
    evaluation["sparse_categorical_accuracy"], 4
))

# Step 10: Create inference models
encoder_model = Model(encoder_input, encoder_state)

current_token = Input(shape=(1,))
state_input = Input(shape=(LATENT_DIM,))

embedded = decoder_embedding(current_token)

step_output, next_state = decoder_gru(
    embedded, initial_state=state_input
)

decoder_model = Model(
    [current_token, state_input],
    [decoder_dense(step_output), next_state]
)

# Translate sentences
def translate_batch(sentences):
    ids = en_vectorizer(np.array(sentences)).numpy()
    states = encoder_model.predict(ids, verbose=0)

    tokens = np.full((len(sentences), 1), START_ID)
    finished = np.zeros(len(sentences), dtype=bool)
    results = [[] for _ in sentences]

    for _ in range(MAX_LEN):
        probs, states = decoder_model.predict(
            [tokens, states], verbose=0
        )

        next_ids = np.argmax(probs[:, 0, :], axis=1)

        for i, token in enumerate(next_ids):
            if finished[i]:
                continue

            if token == 0 or token == END_ID:
                finished[i] = True
            elif token != START_ID:
                results[i].append(fr_vocab[token])

        tokens = next_ids.reshape(-1, 1)

        if np.all(finished):
            break

    return [" ".join(words) for words in results]

# Step 11: Calculate BLEU score
def bleu_tokens(text):
    text = text.lower().translate(
        str.maketrans("", "", string.punctuation)
    )
    return text.split()

count = min(100, len(test_en))

predictions = translate_batch(test_en[:count])

references = [
    [bleu_tokens(s)] for s in test_fr[:count]
]

hypotheses = [
    bleu_tokens(s) for s in predictions
]

bleu = corpus_bleu(
    references,
    hypotheses,
    smoothing_function=SmoothingFunction().method1
)

print("\nEnglish to French Translation")

for i in range(min(5, count)):
    print("\nEnglish:", test_en[i])
    print("Actual:", test_fr[i])
    print("Predicted:", predictions[i])

print("\nBLEU-4 Score:", round(bleu * 100, 2))
