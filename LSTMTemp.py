import nltk
import numpy
import collections
import numpy as np
import pandas as pandas
from keras.engine import Merge
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
from sklearn.model_selection import train_test_split

from GetData import snippet_file_schema
from Utils import csvSmartReader, write_result_file


def tokenize(sens):
    '''
    Should I keep the punctuation tokens or not?
    :param sens:
    :return:
    '''
    # Tokenize a given sentence into a sequence of tokens.
    #return nltk.word_tokenize(sens)
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(sens)



def map_token_seq_to_word_id_seq(token_seq, word_to_id):
    return [map_word_to_id(word_to_id,word) for word in token_seq]


def map_word_to_id(word_to_id, word):
    # map each word to its id.
    if word in word_to_id:
        return word_to_id[word]
    else:
        return word_to_id['$UNK$']


def build_vocab(train_file_name):
    data = []
    train_file = csvSmartReader(train_file_name, snippet_file_schema)
    for row in train_file:
        content_a = row['ATitle'] + " " + row['ASnippet']
        tokens = tokenize(content_a)
        data.extend(tokens)
    count = [['$UNK$', 0]]
    sorted_counts = collections.Counter(data).most_common()
    count.extend(sorted_counts)
    word_to_id = dict()
    for word, _ in count:
        word_to_id[word] = len(word_to_id)
    print('size of vocabulary is %s. ' % len(word_to_id))
    return word_to_id


def read_X(sens_file_name, word_to_id)->[]:
    train_file = csvSmartReader(sens_file_name, snippet_file_schema)
    data = []
    for row in train_file:
        content_a = row['ATitle'] + " " + row['ASnippet']
        content_b = row['ATitle'] + " " + row['ASnippet']
        word_id_seq_a = map_token_seq_to_word_id_seq(tokenize(content_a), word_to_id)
        word_id_seq_b = map_token_seq_to_word_id_seq(tokenize(content_b), word_to_id)
        data.append((word_id_seq_a, word_id_seq_b) )
    print("read %d pairs from %s ." % (len(data), sens_file_name))
    return data



numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest

word_to_id = build_vocab("alta16_kbcoref_train_search_results.csv")
top_words = len(word_to_id)  #keep all words


X = read_X("alta16_kbcoref_train_search_results.csv", word_to_id)



y = pandas.read_csv("alta16_kbcoref_train_labels.csv")["Outcome"]

X_train, X_dev, y_train, y_dev = train_test_split( X, y, test_size=0.1, random_state=42)



X_train_A, X_train_B = zip(*X_train)
X_dev_A, X_dev_B = zip(*X_dev)

X_train_A = np.array(X_train_A)
X_dev_A = np.array(X_dev_A)
X_train_B = np.array(X_train_B)
X_dev_B = np.array(X_dev_B)


print(X_dev_A.shape)

# truncate and pad input sequences
max_word_length = 30
X_train_A = sequence.pad_sequences(X_train_A, maxlen=max_word_length)
X_dev_A = sequence.pad_sequences(X_dev_A, maxlen=max_word_length)
X_train_B = sequence.pad_sequences(X_train_B, maxlen=max_word_length)
X_dev_B = sequence.pad_sequences(X_dev_B, maxlen=max_word_length)
X_train = np.concatenate((X_train_A, X_train_B), axis = 1)


double_X_train_A = np.concatenate((X_train_A, X_train_B), axis = 0)
double_X_train_B = np.concatenate((X_train_B, X_train_A), axis = 0)
double_y_train =  np.append(y_train, y_train)



# create the model
embedding_vecor_length = 20
model_pre = Sequential()
model_pre.add(Embedding(top_words, embedding_vecor_length, input_length=max_word_length, dropout=0.2))
model_pre.add(Dropout(0.1))
model_pre.add(LSTM(25, return_sequences=True))
model_pre.add(Dropout(0.1))
model_pre.add(LSTM(25))
model_pre.add(Dropout(0.1))


model_pre2 = Sequential()
model_pre2.add(Embedding(top_words, embedding_vecor_length, input_length=max_word_length, dropout=0.2))
model_pre2.add(Dropout(0.1))
model_pre2.add(LSTM(25, return_sequences=True))
model_pre2.add(Dropout(0.1))
model_pre2.add(LSTM(25))
model_pre2.add(Dropout(0.1))



model = Sequential()
model.add(Merge([model_pre, model_pre2], mode='concat'))
model.add(Dense(50, activation="relu"))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


model.fit([double_X_train_A, double_X_train_B], double_y_train, nb_epoch=20, batch_size=8)
# Final evaluation of the model
scores = model.evaluate([X_dev_A, X_dev_B], y_dev, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# Fit model and get result, output to the results folder
X_A, X_B = zip(*X)
X_A = sequence.pad_sequences(X_A, maxlen=max_word_length)
X_B = sequence.pad_sequences(X_B, maxlen=max_word_length)
double_X_A = np.concatenate((X_A, X_B), axis = 0)
double_X_B = np.concatenate((X_B, X_A), axis = 0)
double_y = np.append(y, y)
model.fit([double_X_A, double_X_B], double_y, nb_epoch=10, batch_size=8)
results = model.predict_classes([X_A, X_B], batch_size=8, verbose=1)

# Convert ndarray into scalar
results_final = []
for value in results:
    results_final.append(np.asscalar(value))

write_result_file(results_final, 'Results/lstm_embedding.csv')
