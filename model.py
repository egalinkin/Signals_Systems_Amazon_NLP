import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import numpy as np
import json
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# GPU stuff
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)


# With credit to keras.io team for the tokenization parts of this code.
def load_data(datapath, glovepath, sample_size=25000):
    maxlen = 300  # Only use the first 300 words of each review
    vocab_size = 10000  # Have a 10000 word vocabulary

    print("Loading data...")
    labels = list()
    texts = list()
    with open(datapath, "r") as f:
        if sample_size > 0:  # For development, we use a reduced sample of 25000 to make sure the code works.
            for i in tqdm(range(sample_size)):
                line = json.loads(f.readline())
                labels.append(int(line["overall"]) - 1)  # Want to convert values to int, since we're categorizing.
                texts.append(line["reviewText"])
        else:
            for line in tqdm(f):
                line = json.loads(line)
                labels.append(int(line["overall"]) - 1)  # Want to convert values to int, since we're categorizing.
                texts.append(line["reviewText"])

    embeddings_index = dict()
    with open(glovepath, "r") as f:
        for line in tqdm(f.readlines()):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print("Data loaded! Training embeddings, this may take a while...")

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    data = pad_sequences(sequences, maxlen)
    labels = np.asarray(labels)

    indices = np.arange(data.shape[0])  # Shuffling dataset using the number of samples
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]  # Mapping is preserved by using the same indices.

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=.25, stratify=labels)  # Use 25% of data for testing.
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.2, stratify=y_train)  # 20% of remaining to validate

    # Prepare embedding matrix
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((vocab_size, 300))  # Our embedding vectors are 300-dimensional.
    for word, i in tqdm(word_index.items()):
        if i < vocab_size:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return x_train, x_test, x_val, y_train, y_test, y_val, embedding_matrix


def build_model(embeddings):
    model = Sequential()
    # 10000 word vocabulary, 300-dimensional embedding, 200 words.
    model.add(Embedding(10000, 300, weights=[embeddings], input_length=300))
    # Long Short Term Memory layer - explained in README.md
    model.add(LSTM(300, return_sequences=True))
    model.add(LSTM(300, return_sequences=True))
    model.add(Dropout(0.2))  # Dropout can help improve training on a model and reduce overfitting
    model.add(LSTM(300, return_sequences=False))
    model.add(Dense(5, input_dim=300, activation='softmax'))  # Output our predicted score.
    print(model.summary())
    return model


def plot_history(history):
    accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    validation_loss = history.history['val_loss']
    epochs = range(1, len(accuracy) + 1)
    plt.plot(epochs, accuracy, 'bo', label="Training accuracy")
    plt.plot(epochs, validation_accuracy, 'b', label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label="Training loss")
    plt.plot(epochs, validation_loss, 'b', label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()

    plt.savefig("accuracy_loss.png")


if __name__ == "__main__":
    sample_size = 25000
    if len(sys.argv) == 2:
        sample_size = int(sys.argv[1])
    X_train, X_test, X_val, y_train, y_test, y_val, embeddings = load_data("aggressive_dedup.json", "glove_300d.txt", sample_size)
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=1)
    print("Building model...")
    model = build_model(embeddings)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'],
                  callbacks=[earlystopping]
                  )
    print("Running!")
    history = model.fit(X_train, y_train, batch_size=256, epochs=15, validation_data=(X_val, y_val))
    preds = model.predict(X_test)
    preds = [np.argmax(x) for x in preds]
    accuracy = np.average([1 if p == l else 0 for p, l in zip(preds, y_test)])
    cm = confusion_matrix(preds, y_test)
    precision = np.diag(cm) / np.sum(cm, axis = 0)
    recall = np.diag(cm) / np.sum(cm, axis = 1)
    print("Test Accuracy: {}".format(accuracy))
    print("Precision: {0:.4f}\tRecall: {0:.4f}".format(np.mean(precision), np.mean(recall)))
    df_cm = pd.DataFrame(cm, range(5), range(5))
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, annot=True, annot_kws={"size":16})
    plt.savefig("confusion_matrix.png")
    plot_history(history)
