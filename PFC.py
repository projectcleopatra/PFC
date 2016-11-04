'''
Assume that there are two pickled files, one for training, and one for testing
'''
from sklearn import svm, metrics

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from os import environ
import os
from random import shuffle
import nltk
import numpy as np
import tensorflow as tf


from BiCharEmbeddingPrep import bichar_to_id, read_unlabeled_dataset_3, embedding_dim_3
from BiCharEmbeddingPrep import read_labeled_dataset_3
from CharEmbeddingPrep import read_labeled_dataset_2, read_unlabeled_dataset_2, monitored_chars, embedding_dim_2, \
    char_to_id
from GetData import train_snippet, test_snippet, train_labels, label_file_schema, snippet_file_schema
from Utils import write_result_file

from WordEmbeddingPrep import build_vocab, read_unlabeled_dataset, embedding_dim, num_classes
from WordEmbeddingPrep import read_labeled_dataset

num_submodels = 1 #3 is just the initial value, the real number of submodels will be reset later
l_rate = 1
discount_rate = 0.999

num_epochs = 80
#Each datapoint: (vecA, vecB, label)








class ALTATrainAndTest:



    def __init__(self):
        NER_only = False #bchu: Name entity recognistion only, use only use a name entity in a webpage.
                        #bchu: Name entity: personal name / organization / place
                        #bchu: used the name entity tagger in the assignment trained by stanford
        #Load NER Tagger
        #http://stackoverflow.com/questions/32819573/nltk-why-does-nltk-not-recognize-the-classpath-variable-for-stanford-ner
        environ['CLASSPATH'] = os.path.dirname(os.path.abspath(__file__)) + "/stanford-ner-2015-12-09/" #bchu: because it's in java

        # Submodel2 : character embedding lookup table
        self.char_to_id = char_to_id
        #bchu: do it in each character. (inspried by word-embedding, you embed each char)

        #Submodel1: word embedding lookup table
        self.word_to_id = build_vocab(train_snippet, snippet_file_schema, NER_only = NER_only ,ner = self.st_ner)

        # Submodel3:
        self.bichar_to_id = bichar_to_id
        #bchu: similar to char_to_id, but this is do it in pairs. this is like a 2-gram on char


        #bchu: formatted data prepared for word-embedding, output is a []
        self.data, self.labels = read_labeled_dataset(train_snippet, snippet_file_schema,
                                         train_labels, label_file_schema, self.word_to_id, NER_only = NER_only, ner = self.st_ner)  # store dataset
        #bchu: formatted data prepared for uni-char embedding
        self.data_2 = read_labeled_dataset_2(train_snippet, snippet_file_schema,
                                         train_labels, label_file_schema, self.char_to_id)  # store dataset
        #bchu: formatted data prepared for bichar embeding
        self.data_3 = read_labeled_dataset_3(train_snippet, snippet_file_schema,
                                         train_labels, label_file_schema, self.bichar_to_id)  # store dataset


        #bchu: due to the fact that we are using a multi model, we want shuffle them correspondingly
        joint_list = list(zip(self.data, self.data_2, self.data_3, self.labels))
        #bchu: cast a zip object into a list

        shuffle(joint_list) #bchu: try to test our alg, 20% - 80%
        self.data, self.data_2, self.data_3,  self.labels = zip(*joint_list) #derefrencing
        self.data = list(self.data) #bchu: an iterable tuple, need to cast into a list
        self.data_2 = list(self.data_2)
        self.data_3 = list(self.data_3)
        self.labels = list(self.labels)

        #Word embedding
        ratio = 0.8
        #Dataset for word-embedding
        self.train_dataset = self.data[:]#int(ratio*len(self.data))]
        self.dev_dataset = self.data[int(0.8*len(self.data)):]
        self.test_dataset = read_unlabeled_dataset(test_snippet, snippet_file_schema, self.word_to_id, NER_only = NER_only, ner = self.st_ner)


        # labels for all sub-models
        self.train_labels = self.labels[:]#int(ratio*len(self.data))]
        self.dev_labels = self.labels[int(0.8*len(self.labels)):]

        self.train_dataset_2 = self.data_2[:]#int(ratio * len(self.data_2))]
        self.dev_dataset_2 = self.data_2[int(0.8 * len(self.data_2)):]
        self.test_dataset_2 = read_unlabeled_dataset_2(test_snippet, snippet_file_schema, self.char_to_id)

        self.train_dataset_3 = self.data_3[:]#int(ratio * len(self.data_3))]
        self.dev_dataset_3 = self.data_3[int(0.8 * len(self.data_3)):]
        self.test_dataset_3 = read_unlabeled_dataset_3(test_snippet, snippet_file_schema, self.bichar_to_id)




    def word_embedding(self, l_rate=0.1, embedding_dim=15):
        test_results = []

        train_dataset = self.train_dataset

        num_words = len(self.word_to_id)
        # Initialize the placeholders and Variables. E.g.
        input_page_A = tf.placeholder(tf.int32, shape=[None])
        input_page_B = tf.placeholder(tf.int32, shape=[None])
        learning_rate = tf.placeholder(tf.float32, shape=[])
        correct_label = tf.placeholder(tf.float32, shape=[])

        # Change the initialised variable smaller from -1.0 to 1 to -0.1 to 0.1 --> reduce chance of stuck at terrible local minimum
        embeddings = tf.Variable(tf.random_uniform([num_words, embedding_dim], -0.1, 0.1))
        weights = tf.Variable(tf.random_uniform([embedding_dim], -0.5, 0.5))
        # Hint: use [None] when you are not certain about the value of shape

        with tf.Session() as sess:
            # Write code for constructing computation graph here.
            # Hint:
            #    1. Find the math operations at https://www.tensorflow.org/versions/r0.10/api_docs/python/math_ops.html
            #    2. Try to reuse/modify the code from tensorflow tutorial.
            #    3. Use tf.reshape if the shape information of a tensor gets lost during the contruction of computation graph.

            # general formula for the Neural Network
            embed1 = tf.nn.embedding_lookup(embeddings, input_page_A)
            embed2 = tf.nn.embedding_lookup(embeddings, input_page_B)

            tmp_m1 = tf.reduce_mean(embed1, 0)  # sum of the matrix in the VERTICAL manner, then calculate the average based on t
            tmp_m2 = tf.reduce_mean(embed2, 0)

            sum_rep1 = tf.reshape(tmp_m1, [embedding_dim])
            sum_rep2 = tf.reshape(tmp_m2, [embedding_dim])

            score_temp = tf.mul(sum_rep1, sum_rep2)
            score = tf.mul(weights, score_temp)
            score = tf.reduce_sum(score)

            # Formulate word embedding learning as a word prediction task. Note that, no negative sampling is applied here.
            # [batch size, num classes] is the dimension that sends into softmax function - batch size is 1 in this case
            y = tf.sigmoid(tf.reshape(score, []))
            prediction = tf.cast(tf.round(y), tf.int32)

            #cross_entropy = tf.reduce_mean(-tf.reduce_sum(correct_label * tf.log(y), reduction_indices=[1]))

            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(y, correct_label)

            actual = tf.cast(correct_label, tf.int32)
            correct_prediction = tf.equal(prediction, actual)
            accuracy = tf.cast(correct_prediction, tf.float32)

            # In this assignment it is sufficient to use GradientDescentOptimizer, you are not required to implement a regularizer.

            # Build SGD optimizer

            # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
            train_step = tf.train.AdagradOptimizer(learning_rate = l_rate).minimize(cross_entropy)
            # train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(cross_entropy)

            sess.run(tf.initialize_all_variables())
            for epoch in range(num_epochs):
                shuffle(train_dataset)
                # Writing the code for training. It is not required to use a batch with size larger than one.
                for i, (page_A, page_B, label) in enumerate(train_dataset):
                    # Run one step of SGD to update word embeddings.
                    #print("Label: ",label)
                    train_step.run(feed_dict={input_page_A: page_A,  input_page_B: page_B, correct_label: float(label), learning_rate: l_rate})
                    l_rate = l_rate * discount_rate
                # The following line computes the accuracy on the development dataset in each epoch.
                print('Epoch %d : %s .' % (epoch, compute_accuracy(accuracy, input_page_A, input_page_B , correct_label, self.dev_dataset)))

            # uncomment the following line in the grading lab for evaluation
            #print('Accuracy on the test set : %s.' % compute_accuracy(accuracy, input_page_A, input_page_B , correct_label, self.test_dataset))
            # iinput_page_A, input_page_B are the placeholders for two input webpages.

            test_results = predict(prediction, input_page_A, input_page_B, self.test_dataset)


            write_result_file(test_results, 'Results/PFC_word_embedding_results.csv')


            dev_results = calculate_y(y, input_page_A, input_page_B, self.dev_dataset)
            test_results = calculate_y(y, input_page_A, input_page_B, self.test_dataset)
            train_results = calculate_y(y, input_page_A, input_page_B, self.train_dataset)

            print('=' * 20)
            print('Finished training the Ghetto Embed-word classification model')
            print('=' * 20)



        return np.asarray(train_results),  \
               np.asarray(dev_results),  \
               np.asarray(test_results)



    def character_embedding(self, l_rate=0.1):

        train_dataset_2 = self.train_dataset_2
        test_results = []
        num_char = len(self.char_to_id)
        # Initialize the placeholders and Variables. E.g.
        input_page_A = tf.placeholder(tf.int32, shape=[None])
        input_page_B = tf.placeholder(tf.int32, shape=[None])
        correct_label = tf.placeholder(tf.float32, shape=[])
        learning_rate = tf.placeholder(tf.float32, shape=[])

        # Change the initialised variable smaller from -1.0 to 1 to -0.1 to 0.1 --> reduce chance of stuck at terrible local minimum
        embeddings = tf.Variable(tf.random_uniform([num_char, embedding_dim_2], -0.01, 0.01))
        weights = tf.Variable(tf.random_uniform([embedding_dim_2, embedding_dim_2], -0.5, 0.5))
        # Hint: use [None] when you are not certain about the value of shape

        with tf.Session() as sess:
            # Write code for constructing computation graph here.
            # Hint:
            #    1. Find the math operations at https://www.tensorflow.org/versions/r0.10/api_docs/python/math_ops.html
            #    2. Try to reuse/modify the code from tensorflow tutorial.
            #    3. Use tf.reshape if the shape information of a tensor gets lost during the contruction of computation graph.

            # general formula for the Neural Network
            embed1 = tf.nn.embedding_lookup(embeddings, input_page_A)
            embed2 = tf.nn.embedding_lookup(embeddings, input_page_B)

            tmp_m1 = tf.reduce_mean(embed1,
                                    0)  # sum of the matrix in the VERTICAL manner, then calculate the average based on t
            tmp_m2 = tf.reduce_mean(embed2, 0)

            sum_rep1 = tf.reshape(tmp_m1, [1, embedding_dim_2])
            sum_rep2 = tf.reshape(tmp_m2, [1, embedding_dim_2])

            similarity_score_temp = tf.matmul(sum_rep1, weights)
            similarity_score = tf.matmul(similarity_score_temp, sum_rep2, transpose_b=True)

            # Formulate word embedding learning as a word prediction task. Note that, no negative sampling is applied here.
            # [batch size, num classes] is the dimension that sends into softmax function - batch size is 1 in this case
            y = tf.sigmoid(tf.reshape(similarity_score, []))
            prediction = tf.cast(tf.round(y), tf.int32)

            # cross_entropy = tf.reduce_mean(-tf.reduce_sum(correct_label * tf.log(y), reduction_indices=[1]))

            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(y, correct_label)

            actual = tf.cast(correct_label, tf.int32)
            correct_prediction = tf.equal(prediction, actual)
            accuracy = tf.cast(correct_prediction, tf.float32)


            # Build SGD optimizer

            # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
            train_step = tf.train.AdagradOptimizer(learning_rate = l_rate).minimize(cross_entropy)
            # train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(cross_entropy)

            sess.run(tf.initialize_all_variables())
            for epoch in range(num_epochs):
                shuffle(train_dataset_2)
                # Writing the code for training. It is not required to use a batch with size larger than one.

                for i, (page_A, page_B, label) in enumerate(train_dataset_2):
                    # Run one step of SGD to update word embeddings.
                    train_step.run(feed_dict={input_page_A: page_A,  input_page_B: page_B, correct_label: label, learning_rate: l_rate})
                    l_rate = l_rate * discount_rate
                # The following line computes the accuracy on the development dataset in each epoch.
                print('Epoch %d : %s .' % (epoch, compute_accuracy(accuracy, input_page_A, input_page_B , correct_label, self.dev_dataset_2)))

            # uncomment the following line in the grading lab for evaluation
            #print('Accuracy on the test set : %s.' % compute_accuracy(accuracy, input_page_A, input_page_B , correct_label, self.test_dataset))
            # iinput_page_A, input_page_B are the placeholders for two input webpages.

            #dev_results = predict(prediction, input_page_A, input_page_B, self.dev_dataset_2)
            #test_results = predict(prediction, input_page_A, input_page_B, self.test_dataset_2)
            #train_results = predict(prediction, input_page_A, input_page_B, self.train_dataset_2)

            dev_results = calculate_y(y, input_page_A, input_page_B, self.dev_dataset_2)
            test_results = calculate_y(y, input_page_A, input_page_B, self.test_dataset_2)
            train_results = calculate_y(y, input_page_A, input_page_B, self.train_dataset_2)

            print('=' * 20)
            print('Finished training the Ghetto character-embedding classification model')
            print('=' * 20)

        return np.asarray(train_results), \
               np.asarray(dev_results), \
               np.asarray(test_results)

    def bi_character_embedding(self, l_rate=0.1):

        train_dataset_3 = self.train_dataset_3
        test_results = []
        num_char = len(self.bichar_to_id)
        # Initialize the placeholders and Variables. E.g.
        input_page_A = tf.placeholder(tf.int32, shape=[None])
        input_page_B = tf.placeholder(tf.int32, shape=[None])
        correct_label = tf.placeholder(tf.float32, shape=[])
        learning_rate = tf.placeholder(tf.float32, shape=[])
        # Change the initialised variable smaller from -1.0 to 1 to -0.1 to 0.1 --> reduce chance of stuck at terrible local minimum
        embeddings = tf.Variable(tf.random_uniform([num_char, embedding_dim_3], -0.1, 0.1))
        weights = tf.Variable(tf.random_uniform([embedding_dim_3, embedding_dim_3], -0.5, 0.5))

        # Hint: use [None] when you are not certain about the value of shape

        with tf.Session() as sess:
            # Write code for constructing computation graph here.
            # Hint:
            #    1. Find the math operations at https://www.tensorflow.org/versions/r0.10/api_docs/python/math_ops.html
            #    2. Try to reuse/modify the code from tensorflow tutorial.
            #    3. Use tf.reshape if the shape information of a tensor gets lost during the contruction of computation graph.

            # general formula for the Neural Network
            embed1 = tf.nn.embedding_lookup(embeddings, input_page_A)
            embed2 = tf.nn.embedding_lookup(embeddings, input_page_B)

            tmp_m1 = tf.reduce_mean(embed1,
                                    0)  # sum of the matrix in the VERTICAL manner, then calculate the average based on t
            tmp_m2 = tf.reduce_mean(embed2, 0)

            sum_rep1 = tf.reshape(tmp_m1, [1, embedding_dim_3])
            sum_rep2 = tf.reshape(tmp_m2, [1, embedding_dim_3])

            # general formula for the Neural Network
            similarity_score_temp = tf.matmul(sum_rep1, weights)
            similarity_score = tf.matmul(similarity_score_temp, sum_rep2, transpose_b=True)

            # Formulate word embedding learning as a word prediction task. Note that, no negative sampling is applied here.
            # [batch size, num classes] is the dimension that sends into softmax function - batch size is 1 in this case
            y = tf.sigmoid(tf.reshape(similarity_score, []))
            prediction = tf.cast(tf.round(y), tf.int32)


            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(y, correct_label)

            actual = tf.cast(correct_label, tf.int32)
            correct_prediction = tf.equal(prediction, actual)
            accuracy = tf.cast(correct_prediction, tf.float32)


            # Build SGD optimizer

            # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
            train_step = tf.train.AdagradOptimizer(learning_rate=l_rate).minimize(cross_entropy)
            # train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(cross_entropy)

            sess.run(tf.initialize_all_variables())
            for epoch in range(num_epochs):
                shuffle(train_dataset_3)
                # Writing the code for training. It is not required to use a batch with size larger than one.

                for i, (page_A, page_B, label) in enumerate(train_dataset_3):
                    # Run one step of SGD to update word embeddings.
                    train_step.run(feed_dict={input_page_A: page_A,  input_page_B: page_B, correct_label: label, learning_rate: l_rate})
                    l_rate = l_rate * discount_rate
                # The following line computes the accuracy on the development dataset in each epoch.
                print('Epoch %d : %s .' % (epoch, compute_accuracy(accuracy, input_page_A, input_page_B , correct_label, self.dev_dataset_3)))

            # uncomment the following line in the grading lab for evaluation
            #print('Accuracy on the test set : %s.' % compute_accuracy(accuracy, input_page_A, input_page_B , correct_label, self.test_dataset))
            # iinput_page_A, input_page_B are the placeholders for two input webpages.

            #dev_results = predict(prediction, input_page_A, input_page_B, self.dev_dataset_3)
            #test_results = predict(prediction, input_page_A, input_page_B, self.test_dataset_3)
            #train_results = predict(prediction, input_page_A, input_page_B, self.train_dataset_3)

            dev_results = calculate_y(y, input_page_A, input_page_B, self.dev_dataset_3)
            test_results = calculate_y(y, input_page_A, input_page_B, self.test_dataset_3)
            train_results = calculate_y(y, input_page_A, input_page_B, self.train_dataset_3)

            print('=' * 20)
            print('Finished training the bi-character-embedding classification model')
            print('=' * 20)

        return np.asarray(train_results), \
               np.asarray(dev_results), \
               np.asarray(test_results)



    def main_model(self):



        #result from the submodels
        t2, d2, te2 = self.word_embedding(1, embedding_dim=20)
        t1, d1, te1 = self.character_embedding(1)

        t3, d3, te3 = self.bi_character_embedding(1)


        t= np.column_stack((t1, t2, t3, np.ones(t2.shape[0])))
        print(t.shape)
        d = np.column_stack((d1, d2, d3, np.ones(d2.shape[0])))
        te = np.column_stack((te1, te2, te3, np.ones(te2.shape[0])))



        # http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_hastie_10_2.html#sphx-glr-auto-examples-ensemble-plot-adaboost-hastie-10-2-py
        clf = svm.SVC(kernel='rbf', C=1)
        # Cross validation score result

        clf.fit(t, self.train_labels)
        test_results = clf.predict(te)
        print(metrics.accuracy_score(self.dev_labels, clf.predict(d)))

        write_result_file(test_results, 'Results/ALTA2016 PFC Mixture.csv')

        return test_results





#####################
#Two input vector - for submodel training and evaluation
def compute_accuracy(accuracy, input_page_A, input_page_B, correct_label, eval_dataset):
    num_correct = 0
    for (page_A, page_B, label) in eval_dataset:
        num_correct += accuracy.eval(feed_dict={input_page_A: page_A,  input_page_B: page_B, correct_label: label})
    print('#correct submodel classification is %s ' % num_correct)
    return num_correct / len(eval_dataset)


def calculate_y(y, input_page_A, input_page_B, dataset):
    results=[]
    for (page_A, page_B, dummy_label) in dataset:
        output = y.eval(feed_dict={input_page_A: page_A, input_page_B: page_B})
        results.append(output)
    return results

def predict(prediction, input_page_A, input_page_B, test_dataset):
    test_results = []
    for (page_A, page_B, dummy_label) in test_dataset:
        test_results.append(prediction.eval(feed_dict={input_page_A: page_A,  input_page_B: page_B}))
    return test_results

#####################
#One input vector - for final model training and evaluation
def compute_final_accuracy(accuracy, submodel_prediction_matrix, correct_label, eval_dataset):
    num_correct = 0
    for (input, label) in eval_dataset:
        num_correct += accuracy.eval(feed_dict={submodel_prediction_matrix: input, correct_label: label})
    print('#correct final committee classification is %s ' % num_correct)
    return num_correct / len(eval_dataset)


def final_predict(prediction, submodel_prediction_matrix, test_dataset):
    test_results = []
    for (input, dummy_label) in test_dataset:
        test_results.append(prediction.eval(feed_dict={submodel_prediction_matrix: input}))
    return test_results

######################


def main():
    tnt = ALTATrainAndTest() #train And Test instance
    tnt.main_model() #inside of this, it will get the train result of all the sub models
    return

if __name__ == "__main__":
    main()
