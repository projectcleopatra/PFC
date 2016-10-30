from random import shuffle

import gensim
import nltk
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import re

from GetData import snippet_file_schema
from Utils import csvSmartReader, csvSmartWriter
'''
Reference: http://cs.stanford.edu/people/danqi/papers/nips2013.pdf

'''



embedding_dim = 300 # Google and Stanford embedding vectors are both (300,) vectors
num_classes =2
num_epochs = 10
discount_rate = 0.99


def create_y_vec(label: str):
    '''
    Creates target vectors from csv label '0' or '1'
    :param label: String
    :return:
    '''
   # Generate a label vector for a given classification label.
    label_vec = np.zeros(num_classes)
    label_vec[int(label)] = 1
    return label_vec

def write_result_file(results, filename):
    writer = csvSmartWriter(filename, ["Id", "Outcome"])
    id = 200
    for result in results:
        row_dict = {}
        row_dict["Outcome"] = result
        row_dict["Id"] = id
        writer.writerow(row_dict)
        id += 1
    return




class NeuralTensorNetwork:

    def __init__(self):

        self.google_embed = gensim.models.Word2Vec.load_word2vec_format(
                './pretrained_embeddings/GoogleNews-vectors-negative300.bin', binary=True)

        print("Model Pretrained on Google News has been loaded ")

        X = self.read_X("alta16_kbcoref_train_search_results.csv", "Google")
        y = pandas.read_csv("alta16_kbcoref_train_labels.csv")["Outcome"]

        self.X_train, self.X_dev, self.y_train, self.y_dev = train_test_split(X, y, test_size=0.0, random_state=42)

        self.train_dataset = list(zip(self.X_train, self.y_train))
        self.dev_dataset = list(zip(self.X_dev, self.y_dev))

        self.X_test = self.read_X("alta16_kbcoref_test_search_results.csv", "Google")
        self.y_test = np.zeros(len(self.X_test))
        self.test_dataset = list(zip(self.X_test, self.y_test))


    def eval(self, l_rate):

        # Initialize the placeholders and Variables. E.g.

        input_page_A = tf.placeholder(tf.float32, shape=[embedding_dim])
        input_page_B = tf.placeholder(tf.float32, shape=[embedding_dim])
        learning_rate = tf.placeholder(tf.float32, shape=[])
        correct_label = tf.placeholder(tf.float32, shape=[num_classes])

        # Change the initialised variable smaller from -1.0 to 1 to -0.1 to 0.1 --> reduce chance of stuck at terrible local minimum

        W1 = tf.Variable(tf.random_uniform([embedding_dim, embedding_dim], -0.5, 0.5))
        W2 = tf.Variable(tf.random_uniform([embedding_dim, embedding_dim], -0.5, 0.5))
        V = tf.Variable(tf.random_uniform([num_classes, 2*embedding_dim], -0.5, 0.5))
        b = tf.Variable(tf.random_uniform([num_classes], -0.5, 0.5))



        # Hint: use [None] when you are not certain about the value of shape

        with tf.Session() as sess:
            # Write code for constructing computation graph here.
            # Hint:
            #    1. Find the math operations at https://www.tensorflow.org/versions/r0.10/api_docs/python/math_ops.html
            #    2. Try to reuse/modify the code from tensorflow tutorial.
            #    3. Use tf.reshape if the shape information of a tensor gets lost during the contruction of computation graph.

            # general formula for the Neural Network


            input_page_A2 = tf.expand_dims(tf.squeeze(input_page_A), dim = 0) # shape [1, embedding_dim]
            input_page_B2 = tf.expand_dims(tf.squeeze(input_page_B), dim = 1) # shape [embedding_dim, 1]


            tmp1 = tf.squeeze(tf.matmul(tf.matmul(input_page_A2, W1), input_page_B2))
            tmp2 = tf.squeeze(tf.matmul(tf.matmul(input_page_A2, W2), input_page_B2))
            # print(tmp1)
            # print(tmp2)
            concatenated_sum = tf.squeeze(tf.pack([tmp1, tmp2]))
            # print(concatenated_sum)

            big_e = tf.concat(0, [tf.transpose(input_page_A2), input_page_B2])

            sum2 = tf.squeeze(tf.matmul(V, big_e))

            sum = tf.add_n([concatenated_sum, sum2, b])


            # Formulate word embedding learning as a word prediction task. Note that, no negative sampling is applied here.
            # [batch size, num classes] is the dimension that sends into softmax function - batch size is 1 in this case
            y = tf.nn.softmax(sum)
            print(y)
            print(correct_label)

            cross_entropy = tf.reduce_mean(-tf.reduce_sum(correct_label * tf.log(y), reduction_indices=[0]))

            prediction = tf.cast(tf.argmax(y, 0), tf.int32)
            actual = tf.cast(tf.argmax(correct_label, 0), tf.int32)
            correct_prediction = tf.equal(prediction, actual)
            accuracy = tf.cast(correct_prediction, tf.float32)

            # In this assignment it is sufficient to use GradientDescentOptimizer, you are not required to implement a regularizer.

            # Build SGD optimizer

            # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
            train_step = tf.train.AdagradOptimizer(learning_rate=l_rate).minimize(cross_entropy)
            # train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(cross_entropy)

            sess.run(tf.initialize_all_variables())
            for epoch in range(num_epochs):
                shuffle(self.train_dataset)
                # Writing the code for training. It is not required to use a batch with size larger than one.
                for i, ((vec_A, vec_B), label) in enumerate(zip(self.X_train, self.y_train)):
                    # Run one step of SGD to update word embeddings.

                    train_step.run(feed_dict={input_page_A: vec_A,
                                              input_page_B: vec_B,
                                              correct_label: create_y_vec(label),
                                              learning_rate: l_rate})

                    train_step.run(feed_dict={input_page_A: vec_B,
                                              input_page_B: vec_A,
                                              correct_label: create_y_vec(label),
                                              learning_rate: l_rate})


                    l_rate = l_rate * discount_rate
                # The following line computes the accuracy on the development dataset in each epoch.
                print('Epoch %d : %s .' % (epoch,
                                           compute_accuracy(accuracy, input_page_A, input_page_B, correct_label,
                                                                    self.dev_dataset)))

            # uncomment the following line in the grading lab for evaluation
            # print('Accuracy on the test set : %s.' % compute_accuracy(accuracy, input_page_A, input_page_B , correct_label, self.test_dataset))
            # input_page_A, input_page_B are the placeholders for two input webpages.


            test_results = predict(prediction, input_page_A, input_page_B, self.test_dataset)
            write_result_file(test_results, "./Results/NeuralTensorNetworkResult.csv")

            print('=' * 20)
            print('Finished training the Pre-trained Embed-word classification model')
            print('=' * 20)
        return

    def read_X(self, sens_file_name, embedding_source: str) -> []:

        if embedding_source == "Google":
            # import pretrained Google Embed Model
            embedding = self.google_embed


        elif embedding_source == "Stanford":
            embedding = gensim.models.Word2Vec.load_word2vec_format( \
                './pretrained_embeddings/glove.840B.300d.bin', binary=False)
            print("Stanford GloVe has been loaded ")

        train_file = csvSmartReader(sens_file_name, snippet_file_schema)
        data = []
        for row in train_file:
            content_a = row['ATitle'] + " " + row['ASnippet']
            content_b = row['BTitle'] + " " + row['BSnippet']

            stopwords = nltk.corpus.stopwords.words('english')
            a_tokens = [w for w in nltk.word_tokenize(content_a) if w not in stopwords]
            b_tokens = [w for w in nltk.word_tokenize(content_b) if w not in stopwords]

            vec_a = np.zeros((0,embedding_dim), float)  # newscorp has dimensionality 300
            vec_b = np.zeros((0,embedding_dim), float)

            for token in a_tokens:
                if token in embedding.vocab:
                    vec_a = np.row_stack((vec_a, embedding[token]))
                elif token.lower() in embedding.vocab:
                    vec_a = np.row_stack((vec_a, embedding[token.lower()]))
                else:
                    for t in re.split('/|\|',token):
                        if t in embedding.vocab:
                            vec_a = np.row_stack((vec_a, embedding[t]))

            assert vec_a.shape[0]!= 0
            avg_vec_a = np.mean(vec_a, axis=0) #not including the first row of zeros


            for token in b_tokens:
                if token in embedding.vocab:
                    vec_b = np.row_stack((vec_b, embedding[token]))
                elif token.lower() in embedding.vocab:
                    vec_b = np.row_stack((vec_b, embedding[token.lower()]))
                else:
                    for t in re.split('/|\|',token):
                        if t in embedding.vocab:
                            vec_b = np.row_stack((vec_b, embedding[t]))

            assert vec_b.shape[0] != 0
            avg_vec_b = np.mean(vec_b, axis=0) #not including the first row of zeross


            data.append((avg_vec_a, avg_vec_b))

            if len(data) % 20 == 0:
                print("read %d pairs from %s ." % (len(data), sens_file_name))

        print("read %d pairs from %s ." % (len(data), sens_file_name))
        return data

def compute_accuracy(accuracy, input_page_A, input_page_B, correct_label, eval_dataset):
    num_correct = 0

    for ((page_A, page_B), label) in eval_dataset:
        num_correct += accuracy.eval(feed_dict={input_page_A: page_A,
                                                input_page_B: page_B,
                                                correct_label: create_y_vec(label)})
    print('#correct submodel classification is %s ' % num_correct)

    if len(eval_dataset)==0:
        return 0
    else:
        return num_correct / len(eval_dataset)

def predict(prediction, input_page_A, input_page_B, test_dataset):
    test_results = []
    for ((page_A, page_B), dummy_label) in test_dataset:
        test_results.append(prediction.eval(feed_dict={input_page_A: page_A,  input_page_B: page_B}))
    return test_results


if __name__ == "__main__":
    ntn = NeuralTensorNetwork()
    ntn.eval(l_rate = 0.8)
