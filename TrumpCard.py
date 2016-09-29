'''
Assume that there are two pickled files, one for training, and one for testing
'''
import pickle
import collections
from random import shuffle

import nltk
import numpy as np
import tensorflow as tf
from GetData import train_snippet, test_snippet, train_labels, label_file_schema, snippet_file_schema
from Utils import *
# http://www.deeplearningbook.org
# https://www.youtube.com/watch?v=SGZ6BttHMPw
# David Barber - Baysian Based Machine Learning
# ACL TACL ELLP Stanford NLP Group - keyword when googling papers
# Tensorflow --> SyntaxNet released one month ago
#Edword -> Graphical model based on tensorflow
# Torch - fb vs <--- Google competed
num_submodels = 1
num_classes = 2
embedding_dim = 20
learning_rate = 0.02
num_epochs = 10
#Each datapoint: (vecA, vecB, label)
#
def create_label_vec(label: str):
    '''
    Creates target vectors from csv label '0' or '1'
    :param label: String
    :return:
    '''
   # Generate a label vector for a given classification label.
    label_vec = np.zeros(num_classes)
    label_vec[int(label)] = 1
    return label_vec


def tokenize(page_content):
    '''
    Should I keep the punctuation tokens or not?
    :param page_content:
    :return:
    '''
    # Tokenize a given sentence into a sequence of tokens.
    #return nltk.word_tokenize(sens)
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(page_content)


def map_token_seq_to_word_id_seq(token_seq, word_to_id):
    return [map_word_to_id(word_to_id,word) for word in token_seq]

def map_word_to_id(word_to_id, word):
    # map each word to its id.
    if word in word_to_id:
        return word_to_id[word]
    else:
        return word_to_id['$UNK$']  #TODO

def build_vocab(input_file_name, input_file_schema):
    '''
    Important! This builds a list of vocabulary indexes from file
    :param sens_file_name:
    :return:
    '''
    data = []
    csvreader = csvSmartReader(input_file_name, input_file_schema)
    for input in csvreader:
        tokens = tokenize(input['ATitle']+input['ASnippet']+input['BTitle']+input['BSnippet'])
        data.extend(tokens)


    count = [['$UNK$', 0]]
    sorted_counts = collections.Counter(data).most_common()
    count.extend(sorted_counts)
    word_to_id = dict()
    for word, _ in count:
        word_to_id[word] = len(word_to_id)
    print('size of vocabulary is %s. ' % len(word_to_id))
    return word_to_id


def read_labeled_dataset(input_file_name, input_file_schema, label_file_name, label_file_schema, word_to_id):
    input_file_reader = csvSmartReader(input_file_name, input_file_schema)
    label_file_reader = csvSmartReader(label_file_name, label_file_schema)

    data = []
    for input,label in zip(input_file_reader, label_file_reader):
        content_A = input['ATitle'] + input['ASnippet'] #TODO
        content_B = input['BTitle'] + input['BSnippet'] #TODO
        word_id_seq_A = map_token_seq_to_word_id_seq(tokenize(content_A), word_to_id)
        word_id_seq_B = map_token_seq_to_word_id_seq(tokenize(content_B), word_to_id)

        data.append((word_id_seq_A, word_id_seq_B, create_label_vec(label['outcome'].strip('\n'))))
    print("read %d sentences from %s ." % (len(data), input_file_name))
    return data


def read_unlabeled_dataset(input_file_name, input_file_schema, word_to_id):
    input_file_reader = csvSmartReader(input_file_name, input_file_schema)
    data = []
    for input in input_file_reader:
        content_A = input['ATitle'] + input['ASnippet']  # TODO
        content_B = input['BTitle'] + input['BSnippet']  # TODO
        word_id_seq_A = map_token_seq_to_word_id_seq(tokenize(content_A), word_to_id)
        word_id_seq_B = map_token_seq_to_word_id_seq(tokenize(content_B), word_to_id)
        #it was: data.append(word_id_seq) without tuples
        data.append((word_id_seq_A, word_id_seq_B, [0,0,0]))
    print("read %d sentences from %s ." % (len(data), input_file_name))
    return data




class ALTATrainAndTest:

    def __init__(self, bootstrap_or_not=False):

        #Load scraped training data
        try:
            with open('alta16_kbcoref_train_pairs_scraped_content', 'rb') as pickle_file:
                self.train_page_data  = pickle.load(pickle_file)  # id --> {[field:value]*}
        except:
            self.train_page_data = dict()

        '''
        #Load scraped test data
        try:
            with open('alta16_kbcoref_test_pairs_scraped_content', 'rb') as pickle_file:
                self.test_page_data  = pickle.load(pickle_file)  # id --> {[field:value]*}
        except:
            self.test_page_data = dict()
        '''


        self.word_to_id = build_vocab(train_snippet, snippet_file_schema)

        self.data = read_labeled_dataset(train_snippet, snippet_file_schema,
                                         train_labels,label_file_schema, self.word_to_id)  # store dataset

        #segment data into training and development set
        self.boostrap = bootstrap_or_not
        shuffle(self.data)
        self.train_dataset = self.data[:int(0.8*len(self.data))]
        self.dev_dataset = self.data[int(0.8*len(self.data)):]

        self.test_dataset = read_unlabeled_dataset(test_snippet, snippet_file_schema, self.word_to_id)


#TODO LSTM availale on tensorflow (reusable) LM1B  input: sequence output: vector
#G??ve word2vec available for high dim embeddings
#https://github.com/tensorflow/models/tree/master/lm_1b word-embedding

    def word_embedding(self):
        test_results = []


        num_words = len(self.word_to_id)
        # Initialize the placeholders and Variables. E.g.
        input_page_A = tf.placeholder(tf.int32, shape=[None])
        input_page_B = tf.placeholder(tf.int32, shape=[None])
        correct_label = tf.placeholder(tf.float32, shape=[num_classes])

        # Change the initialised variable smaller from -1.0 to 1 to -0.1 to 0.1 --> reduce chance of stuck at terrible local minimum
        embeddings = tf.Variable(tf.random_uniform([num_words, embedding_dim], -0.1, 0.1))
        weights = tf.Variable(tf.random_uniform([2*embedding_dim, num_classes], -1.0, 1.0))
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

            sum_rep1 = tf.reshape(tmp_m1, [1, embedding_dim])
            sum_rep2 = tf.reshape(tmp_m2, [1, embedding_dim])

            concatenated_sum = tf.concat(1, [sum_rep1, sum_rep2])

            # Formulate word embedding learning as a word prediction task. Note that, no negative sampling is applied here.
            # [batch size, num classes] is the dimension that sends into softmax function - batch size is 1 in this case
            y = tf.nn.softmax(tf.matmul(concatenated_sum, weights))

            cross_entropy = tf.reduce_mean(-tf.reduce_sum(correct_label * tf.log(y), reduction_indices=[1]))

            prediction = tf.cast(tf.argmax(y, 1), tf.int32)
            actual = tf.cast(tf.argmax(correct_label, 0), tf.int32)
            correct_prediction = tf.equal(prediction, actual)
            accuracy = tf.cast(correct_prediction, tf.float32)

            # In this assignment it is sufficient to use GradientDescentOptimizer, you are not required to implement a regularizer.

            # Build SGD optimizer

            # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
            train_step = tf.train.AdagradOptimizer(learning_rate).minimize(cross_entropy)
            # train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(cross_entropy)

            sess.run(tf.initialize_all_variables())
            for epoch in range(num_epochs):
                shuffle(self.train_dataset)
                # Writing the code for training. It is not required to use a batch with size larger than one.
                for i, (page_A, page_B, label) in enumerate(self.train_dataset):
                    # Run one step of SGD to update word embeddings.
                    train_step.run(feed_dict={input_page_A: page_A,  input_page_B: page_B, correct_label: label})
                # The following line computes the accuracy on the development dataset in each epoch.
                print('Epoch %d : %s .' % (epoch, compute_accuracy(accuracy, input_page_A, input_page_B , correct_label, self.dev_dataset)))
                
                #Bootstrap algorithm starts
                if self.boostrap ==True:
                    print('Bootstrapping new data points:')
                    
                    

            # uncomment the following line in the grading lab for evaluation
            #print('Accuracy on the test set : %s.' % compute_accuracy(accuracy, input_page_A, input_page_B , correct_label, self.test_dataset))
            # iinput_page_A, input_page_B are the placeholders for two input webpages.

            dev_results = predict(prediction, input_page_A, input_page_B, self.dev_dataset)
            test_results = predict(prediction, input_page_A, input_page_B, self.test_dataset)
            train_results = predict(prediction, input_page_A, input_page_B, self.train_dataset)

            print('=' * 20)
            print('Finished training the Ghetto Embed-word classification model')
            print('=' * 20)

        return np.asarray(train_results), np.asarray(dev_results), np.asarray(test_results)

    def main_model(self):
        train_results = []
        dev_results = []
        test_results = []
        #result from the submodels
        train_results1, dev_results1, test_results1 = self.word_embedding()

        #train_results2, dev_results2, test_results2= #TODO Just a placeholder


        #combine training labels to be the new dataset
        train_results.append(train_results1)
        dev_results.append(dev_results1)
        test_results.append(test_results1)

        train_results = np.concatenate(train_results, axis=1)
        dev_results =np.concatenate(dev_results, axis=1)
        test_results = np.concatenate(test_results, axis=1)


        #print(len(train_results), len(dev_results), len(test_results))
        assert train_results.shape[1] == dev_results.shape[1] == test_results.shape[1] == num_submodels
        assert train_results1.shape[0] == train_results.shape[0]
        assert dev_results1.shape[0] == dev_results.shape[0]

        final_trainset = self.make_final_committee_dataset(train_results, self.train_dataset)
        final_devset = self.make_final_committee_dataset(dev_results, self.dev_dataset)
        final_testset = self.make_final_committee_dataset(test_results, self.test_dataset)

        #initialise the final committee voting model

        correct_label = tf.placeholder(tf.float32, shape=[num_classes]) #aka the target variable

        #initialise the input variable aka the big X
        submodel_prediction_matrix = tf.placeholder(tf.float32, shape=[1,
                                                                    num_submodels])  # TODO shape will expand [num_classes, num_models]

        weights = tf.Variable(tf.random_uniform([num_submodels, num_classes], -1.0, 1.0)) #aka the big W

        with tf.Session() as sess:
            y = tf.nn.softmax(tf.matmul(submodel_prediction_matrix, weights))

            cross_entropy = tf.reduce_mean(-tf.reduce_sum(correct_label * tf.log(y), reduction_indices=[1]))

            #change y result from matrix to string labels
            prediction = tf.cast(tf.argmax(y, 1), tf.int32)
            actual = tf.cast(tf.argmax(correct_label, 0), tf.int32)
            correct_prediction = tf.equal(prediction, actual)
            accuracy = tf.cast(correct_prediction, tf.float32)

            # Build SGD optimizer

            # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
            train_step = tf.train.AdagradOptimizer(learning_rate).minimize(cross_entropy)
            # train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(cross_entropy)

            sess.run(tf.initialize_all_variables())
            for epoch in range(num_epochs):
                shuffle(final_trainset)
                # Writing the code for training. It is not required to use a batch with size larger than one.
                for i, (input, label) in enumerate(final_trainset):
                    # Run one step of SGD to update word embeddings.
                    train_step.run(feed_dict={submodel_prediction_matrix: input, correct_label: label})
                # The following line computes the accuracy on the development dataset in each epoch.
                print('Epoch %d : %s .' % (
                epoch, compute_final_accuracy(accuracy, submodel_prediction_matrix, correct_label, final_devset)))

            # uncomment the following line in the grading lab for evaluation
            # print('Accuracy on the test set : %s.' % compute_accuracy(accuracy, input_page_A, input_page_B , correct_label, self.test_dataset))
            # iinput_page_A, input_page_B are the placeholders for two input webpages.

            dev_results = final_predict(prediction, submodel_prediction_matrix, final_devset)
            test_results = final_predict(prediction, submodel_prediction_matrix, final_testset)




        return dev_results, test_results

    def make_final_committee_dataset(self, submodel_results: list, dataset:list):
        '''
        A helper method within the class to prepare for the final result
        :return:
        '''

        final_data =[]


        for i, (page_A, page_B, label) in enumerate(dataset):
            input_vector = submodel_results[i, :].reshape(1, num_submodels)

            assert input_vector.shape == (1, num_submodels)

            final_data.append((input_vector, label))

        return final_data


#####################
#Two input vector - for submodel training and evaluation
def compute_accuracy(accuracy, input_page_A, input_page_B, correct_label, eval_dataset):
    num_correct = 0
    for (page_A, page_B, label) in eval_dataset:
        num_correct += accuracy.eval(feed_dict={input_page_A: page_A,  input_page_B: page_B, correct_label: label})
    print('#correct submodel classification is %s ' % num_correct)
    return num_correct / len(eval_dataset)


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



def write_result_file(test_results, result_file):
    with open(result_file, mode='w') as f:
         for r in test_results:
             f.write("%d\n" % r)

def main():

    tnt = ALTATrainAndTest(bootstrap_or_not=True)
    tnt.main_model()

    return

if __name__ == "__main__":
    main()
