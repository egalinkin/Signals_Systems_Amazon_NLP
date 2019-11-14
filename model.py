import sys
import numpy as np
import json
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, Dense, LSTM, Dropout
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm


# With credit to keras.io team for the tokenization parts of this code.
def load_data(datapath, glovepath, sample_size=25000):
    maxlen = 300  # Only use the first 300 words of each review
    vocab_size = 10000  # Have a 10000 word vocabulary

    print("Loading data...")
    labels = list()
    texts = list()
    with open(datapath, "r") as f:
        if sample_size > 0:  # For development, we use a reduced sample of 25000 to make sure the code works.
            for i in range(sample_size):
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
        for line in f.readlines():
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print("Data loaded!")

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    data = pad_sequences(sequences, maxlen)
    labels = np.asarray(labels)

    indices = np.arange(data.shape[0])  # Shuffling dataset using the number of samples
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]  # Mapping is preserved by using the same indices.

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=.25)  # Use 25% of data for testing.
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.2)  # 20% of remaining to validate

    # Prepare embedding matrix
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((vocab_size, 300))  # Our embedding vectors are 300-dimensional.
    for word, i in word_index.items():
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
    model.add(LSTM(300, return_sequences=True))
    model.add(Dropout(0.1))  # Dropout can help improve training on a model and reduce overfitting
    model.add(LSTM(300, return_sequences=False))
    model.add(Dense(5, input_dim=300, activation='softmax'))  # Output our predicted score.
    print(model.summary())
    return model


def plot_history(history):
    accuracy = history.history['acc']
    validation_accuracy = history.history['val_acc']
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

    plt.show()


if __name__ == "__main__":
    sample_size = 25000
    if len(sys.argv) == 2:
        sample_size = int(sys.argv[1])
    X_train, X_test, X_val, y_train, y_test, y_val, embeddings = load_data("aggressive_dedup.json", "glove_300d.txt", sample_size)
    print()
    print("Building model...")
    model = build_model(embeddings)
    print("Running!")
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(lr=0.005),
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=256, epochs=15, validation_data=(X_val, y_val))
    plot_history(history)
    scores = model.evaluate(X_test, y_test)
    print(scores[0])
