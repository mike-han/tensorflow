import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
import numpy as np

imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
train_data, test_data = imdb['train'], imdb['test']

traning_sentences = []
training_labels = []

testing_sequences = []
testing_labels = []

for s,l in train_data:
  traning_sentences.append(str(s.numpy()))
  training_labels.append(l.numpy())

for s,l in test_data:
  testing_sequences.append(str(s.numpy()))
  testing_labels.append(l.numpy())

training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_token = "<OOV>"


tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(traning_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(traning_sentences)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sequences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length)

model = tf.keras.Sequential()
model.layers.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length))
# model.layer.add(tf.keras.layers.Flatten())
model.layers.add(tf.layers.GlobalAveragePooling1D())
model.layers.add(tf.keras.layers.Dense(6, activation='relu'))
model.layers.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(padded, training_labels_final, epochs=10, validation_data=(testing_padded, testing_labels_final))


