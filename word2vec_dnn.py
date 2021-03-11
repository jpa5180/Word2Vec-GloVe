########################################################
#
# CMPSC 597: Homework 5 Question 2
#      Word2Vec Dense Neural Network Model
#
########################################################



student_name1 = 'Jean P. Astudillo Guerra'
student_name2 = 'John Gilbertson'
student_email1 = 'jpa5180@psu.edu'
student_email2 = 'jag5962@psu.edu'



########################################################
# Import
########################################################
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer



########################################################
# Functions
########################################################



########################################################
# Main function
########################################################
data_pd = pd.read_csv("Movie Reviews.csv")

data_pd.isnull().values.any()
print(data_pd.shape)
print(data_pd.head())

#for data processing
def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Removing capital letters
    sentence = sentence.lower()

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    words = sentence.split()

    stop_words = set(stopwords.words('english'))

    words = [w for w in words if not w in stop_words]

    return words

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

X = []
sentences = list(data_pd['review'].values.astype('U'))
for sen in sentences:
    X.append(preprocess_text(sen))

print(X[3])

y = data_pd['sentiment']
y = np.array(list(map(lambda x: 1 if x == "positive" else 0, y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_val = tokenizer.texts_to_sequences(X_val)

# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)

import gensim

model = gensim.models.Word2Vec(sentences=X, size=100,window=5,workers=4,min_count=1)

#vocab size
words = list(model.wv.vocab)
print('Vocabulary size: %d' % len(words))

#save model
filename = 'embedding_word2vec.txt'
model.wv.save_word2vec_format(filename, binary=False)

from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()
word2vec_file = open('embedding_word2vec.txt', encoding="utf8")

for line in word2vec_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
word2vec_file.close()

embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    if index > vocab_size:
        continue
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

from keras.layers import *

model = Sequential()

embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)

model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer="adam",metrics=["accuracy"])

model.fit(X_train,y_train,validation_data=(X_val, y_val), batch_size=128, epochs=10)

score = model.evaluate(X_test, y_test, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])

