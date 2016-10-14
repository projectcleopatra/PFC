'''
Assume that there are two pickled files, one for training, and one for testing
'''

from os import environ
import os
from random import shuffle
import nltk
import numpy as np
import tensorflow as tf
import gensim
from BiCharEmbeddingPrep import bichar_to_id, read_unlabeled_dataset_3, embedding_dim_3
from BiCharEmbeddingPrep import read_labeled_dataset_3
from CharEmbeddingPrep import read_labeled_dataset_2, read_unlabeled_dataset_2, monitored_chars, embedding_dim_2, \
    char_to_id
from GetData import train_snippet, test_snippet, train_labels, label_file_schema, snippet_file_schema
from PretrainedWord2VecPrep import read_labeled_dataset_for_pretrained_embeddings, \
    read_unlabeled_dataset_for_pretrained_embeddings
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

        self.st_ner = nltk.StanfordNERTagger('stanford-ner-2015-12-09/classifiers/english.all.3class.distsim.crf.ser.gz')
        #bchu: it can tag any document. when useing, call st_ner.tag

        # Submodel2 : character embedding lookup table
        self.char_to_id = char_to_id
        #bchu: do it in each character. (inspried by word-embedding, you embed each char)

        #Submodel1: word embedding lookup table
        self.word_to_id = build_vocab(train_snippet, snippet_file_schema, NER_only = NER_only ,ner = self.st_ner)

        # Submodel3:
        self.bichar_to_id = bichar_to_id
        #bchu: similar to char_to_id, but this is do it in pairs. this is like a 2-gram on char


        #bchu: formatted data prepared for word-embedding, output is a []
        self.data = read_labeled_dataset(train_snippet, snippet_file_schema,
                                         train_labels, label_file_schema, self.word_to_id, NER_only = NER_only, ner = self.st_ner)  # store dataset
        #bchu: formatted data prepared for uni-char embedding
        self.data_2 = read_labeled_dataset_2(train_snippet, snippet_file_schema,
                                         train_labels, label_file_schema, self.char_to_id)  # store dataset
        #bchu: formatted data prepared for bichar embeding
        self.data_3 = read_labeled_dataset_3(train_snippet, snippet_file_schema,
                                         train_labels, label_file_schema, self.bichar_to_id)  # store dataset
        #formatted data prepared for pretrained google data set embeding
        self.data_google_embeddings = read_labeled_dataset_for_pretrained_embeddings(train_snippet, snippet_file_schema,
                                         train_labels, label_file_schema)

        #bchu: due to the fact that we are using a multi model, we want shuffle them correspondingly
        joint_list = list(zip(self.data, self.data_2, self.data_3, self.data_google_embeddings))
        #bchu: cast a zip object into a list

        shuffle(joint_list) #bchu: try to test our alg, 20% - 80%
        self.data, self.data_2, self.data_3, self.data_google_embeddings = zip(*joint_list) #derefrencing
        self.data = list(self.data) #bchu: an iterable tuple, need to cast into a list
        self.data_2 = list(self.data_2)
        self.data_3 = list(self.data_3)
        self.data_google_embeddings = list(self.data_google_embeddings)

        #Word embedding

        self.train_dataset = self.data[:]#int(0.8*len(self.data))]
        self.dev_dataset = self.data[int(0.8*len(self.data)):]
        self.test_dataset = read_unlabeled_dataset(test_snippet, snippet_file_schema, self.word_to_id, NER_only = NER_only, ner = self.st_ner)


        self.train_dataset_2 = self.data_2[:]#int(0.8 * len(self.data_2))]
        self.dev_dataset_2 = self.data_2[int(0.8 * len(self.data_2)):]
        self.test_dataset_2 = read_unlabeled_dataset_2(test_snippet, snippet_file_schema, self.char_to_id)

        self.train_dataset_3 = self.data_3[:]#int(0.8 * len(self.data_3))]
        self.dev_dataset_3 = self.data_3[int(0.8 * len(self.data_3)):]
        self.test_dataset_3 = read_unlabeled_dataset_3(test_snippet, snippet_file_schema, self.bichar_to_id)

        self.train_dataset_emb = self.data_google_embeddings[:]#int(0.8 * len(self.data_google_embeddings))]
        self.dev_dataset_emb = self.data_google_embeddings[int(0.8 * len(self.data_google_embeddings)):]
        self.test_dataset_emb = read_unlabeled_dataset_for_pretrained_embeddings(test_snippet, snippet_file_schema)



    def word_embedding(self, l_rate=0.1, embedding_dim=20):
        test_results = []

        train_dataset = self.train_dataset

        num_words = len(self.word_to_id)
        # Initialize the placeholders and Variables. E.g.
        input_page_A = tf.placeholder(tf.int32, shape=[None])
        input_page_B = tf.placeholder(tf.int32, shape=[None])
        learning_rate = tf.placeholder(tf.float32, shape=[])
        correct_label = tf.placeholder(tf.float32, shape=[num_classes])

        # Change the initialised variable smaller from -1.0 to 1 to -0.1 to 0.1 --> reduce chance of stuck at terrible local minimum
        embeddings = tf.Variable(tf.random_uniform([num_words, embedding_dim], -0.1, 0.1))
        weights = tf.Variable(tf.random_uniform([2*embedding_dim, num_classes], -0.5, 0.5))
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
            train_step = tf.train.AdagradOptimizer(learning_rate = l_rate).minimize(cross_entropy)
            # train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(cross_entropy)

            sess.run(tf.initialize_all_variables())
            for epoch in range(num_epochs):
                shuffle(train_dataset)
                # Writing the code for training. It is not required to use a batch with size larger than one.
                for i, (page_A, page_B, label) in enumerate(train_dataset):
                    # Run one step of SGD to update word embeddings.
                    train_step.run(feed_dict={input_page_A: page_A,  input_page_B: page_B, correct_label: label, learning_rate: l_rate})
                    l_rate = l_rate * discount_rate
                # The following line computes the accuracy on the development dataset in each epoch.
                print('Epoch %d : %s .' % (epoch, compute_accuracy(accuracy, input_page_A, input_page_B , correct_label, self.dev_dataset)))

            # uncomment the following line in the grading lab for evaluation
            #print('Accuracy on the test set : %s.' % compute_accuracy(accuracy, input_page_A, input_page_B , correct_label, self.test_dataset))
            # iinput_page_A, input_page_B are the placeholders for two input webpages.

            dev_results = predict(prediction, input_page_A, input_page_B, self.dev_dataset)
            test_results = predict(prediction, input_page_A, input_page_B, self.test_dataset)
            train_results = predict(prediction, input_page_A, input_page_B, self.train_dataset)

            dev_results = calculate_y(y, input_page_A, input_page_B, self.dev_dataset)
            test_results = calculate_y(y, input_page_A, input_page_B, self.test_dataset)
            train_results = calculate_y(y, input_page_A, input_page_B, self.train_dataset)

            print('=' * 20)
            print('Finished training the Ghetto Embed-word classification model')
            print('=' * 20)

        return np.asarray(train_results), np.asarray(dev_results), np.asarray(test_results)

    def pretrained_word_embedding(self, l_rate=0.1, embedding_dim=20):
        test_results = []
        embedding_dim = 300 #GNews Corpus has dim 300



        train_dataset = self.train_dataset_emb

        num_words = len(self.word_to_id)
        # Initialize the placeholders and Variables. E.g.
        input_page_A = tf.placeholder(tf.float32, shape=[None])
        input_page_B = tf.placeholder(tf.float32, shape=[None])
        learning_rate = tf.placeholder(tf.float32, shape=[])
        correct_label = tf.placeholder(tf.float32, shape=[num_classes])

        # Change the initialised variable smaller from -1.0 to 1 to -0.1 to 0.1 --> reduce chance of stuck at terrible local minimum
        weights = tf.Variable(tf.random_uniform([2*embedding_dim, num_classes], -0.5, 0.5))
        # Hint: use [None] when you are not certain about the value of shape

        with tf.Session() as sess:
            # Write code for constructing computation graph here.
            # Hint:
            #    1. Find the math operations at https://www.tensorflow.org/versions/r0.10/api_docs/python/math_ops.html
            #    2. Try to reuse/modify the code from tensorflow tutorial.
            #    3. Use tf.reshape if the shape information of a tensor gets lost during the contruction of computation graph.

            # general formula for the Neural Network

            sum_rep1 = tf.reshape(input_page_A, [1, embedding_dim])
            sum_rep2 = tf.reshape(input_page_B, [1, embedding_dim])

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
            train_step = tf.train.AdagradOptimizer(learning_rate = l_rate).minimize(cross_entropy)
            # train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(cross_entropy)

            sess.run(tf.initialize_all_variables())
            for epoch in range(num_epochs):
                shuffle(train_dataset)
                # Writing the code for training. It is not required to use a batch with size larger than one.
                for i, (vec_A, vec_B, label) in enumerate(train_dataset):
                    # Run one step of SGD to update word embeddings.

                    train_step.run(feed_dict={input_page_A: vec_A,  input_page_B: vec_B, correct_label: label, learning_rate: l_rate})
                    l_rate = l_rate * discount_rate
                # The following line computes the accuracy on the development dataset in each epoch.
                print('Epoch %d : %s .' % (epoch, compute_pre_emb_accuracy(accuracy, input_page_A, input_page_B , correct_label, self.dev_dataset_emb)))

            # uncomment the following line in the grading lab for evaluation
            # print('Accuracy on the test set : %s.' % compute_accuracy(accuracy, input_page_A, input_page_B , correct_label, self.test_dataset))
            # input_page_A, input_page_B are the placeholders for two input webpages.

            dev_results = predict(prediction, input_page_A, input_page_B, self.dev_dataset_emb)
            test_results = predict(prediction, input_page_A, input_page_B, self.test_dataset_emb)
            train_results = predict(prediction, input_page_A, input_page_B, self.train_dataset_emb)


            dev_results = calculate_y(y, input_page_A, input_page_B, self.dev_dataset_emb)
            test_results = calculate_y(y, input_page_A, input_page_B, self.test_dataset_emb)
            train_results = calculate_y(y, input_page_A, input_page_B, self.train_dataset_emb)


            print('=' * 20)
            print('Finished training the Pre-trained Embed-word classification model')
            print('=' * 20)

        return np.asarray(train_results), np.asarray(dev_results), np.asarray(test_results)

    def character_embedding(self, l_rate=0.1):

        train_dataset_2 = self.train_dataset_2
        test_results = []
        num_char = len(self.char_to_id)
        # Initialize the placeholders and Variables. E.g.
        input_page_A = tf.placeholder(tf.int32, shape=[None])
        input_page_B = tf.placeholder(tf.int32, shape=[None])
        correct_label = tf.placeholder(tf.float32, shape=[num_classes])
        learning_rate = tf.placeholder(tf.float32, shape=[])

        # Change the initialised variable smaller from -1.0 to 1 to -0.1 to 0.1 --> reduce chance of stuck at terrible local minimum
        embeddings = tf.Variable(tf.random_uniform([num_char, embedding_dim_2], -0.01, 0.01))
        weights = tf.Variable(tf.random_uniform([2 * embedding_dim_2, num_classes], -0.05, 0.05))
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

            sum_rep1 = tf.reshape(tmp_m1, [1, embedding_dim_2])
            sum_rep2 = tf.reshape(tmp_m2, [1, embedding_dim_2])

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

        return np.asarray(train_results), np.asarray(dev_results), np.asarray(test_results)

    def bi_character_embedding(self, l_rate=0.1):

        train_dataset_3 = self.train_dataset_3
        test_results = []
        num_char = len(self.bichar_to_id)
        # Initialize the placeholders and Variables. E.g.
        input_page_A = tf.placeholder(tf.int32, shape=[None])
        input_page_B = tf.placeholder(tf.int32, shape=[None])
        correct_label = tf.placeholder(tf.float32, shape=[num_classes])
        learning_rate = tf.placeholder(tf.float32, shape=[])
        # Change the initialised variable smaller from -1.0 to 1 to -0.1 to 0.1 --> reduce chance of stuck at terrible local minimum
        embeddings = tf.Variable(tf.random_uniform([num_char, embedding_dim_3], -0.1, 0.1))
        weights = tf.Variable(tf.random_uniform([2 * embedding_dim_3, num_classes], -0.5, 0.5))

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

            sum_rep1 = tf.reshape(tmp_m1, [1, embedding_dim_3])
            sum_rep2 = tf.reshape(tmp_m2, [1, embedding_dim_3])

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

        return np.asarray(train_results), np.asarray(dev_results), np.asarray(test_results)



    def main_model(self, l_rate=0.1, mode = 'weight'):


        result_tuples = []
        #result from the submodels
        result_tuples.append(self.pretrained_word_embedding(1, embedding_dim=300))
        result_tuples.append(self.word_embedding(0.5, embedding_dim=20))
        '''
        for dim in range(17, 18):
            for gamma in np.arange(1, 1.1, 0.1):
                result_tuples.append(self.pretrained_word_embedding(gamma, embedding_dim = dim))
                #result_tuples.append(self.word_embedding(gamma, embedding_dim = dim))
        '''
        #result_tuples.append(self.character_embedding(0.1))
        #result_tuples.append(self.bi_character_embedding(0.1))
        '''
        Stanford Name Entity Recogniser : mention of entity (Most hopeful idea) Do they share lots of entities in common
                                            Do they share URLs?
        Reverse engineer the biased data generation
        Potential idea: use pre trained word embeddings

        '''



        submodel_train_results, submodel_dev_results, submodel_test_results = zip(*result_tuples)
        submodel_train_results = list(submodel_train_results)
        submodel_dev_results = list(submodel_dev_results)
        submodel_test_results = list(submodel_test_results)


        num_submodels = len(submodel_train_results)
        assert len(submodel_train_results) == len(submodel_dev_results) == len(submodel_test_results) #same number of models ==> same number of length for results






        if mode == 'weight':

            train_results = np.concatenate(submodel_train_results, axis=2).transpose(0,2,1).reshape(len(submodel_train_results[0]), num_submodels*num_classes)
            print(train_results.shape)
            dev_results =np.concatenate(submodel_dev_results, axis=2).transpose(0,2,1).reshape(len(submodel_dev_results[0]), num_submodels*num_classes)
            test_results = np.concatenate(submodel_test_results, axis=2).transpose(0,2,1).reshape(len(submodel_test_results[0]), num_submodels*num_classes)




            #print(len(train_results), len(dev_results), len(test_results))
            assert train_results.shape[1] == dev_results.shape[1] == test_results.shape[1] == num_submodels*num_classes
            assert submodel_train_results[0].shape[0] == train_results.shape[0]
            assert submodel_dev_results[0].shape[0] == dev_results.shape[0]

            final_trainset = self.make_final_committee_dataset(train_results, self.train_dataset_2, mode = mode, num_submodels = num_submodels)
            final_devset = self.make_final_committee_dataset(dev_results, self.dev_dataset, mode=mode, num_submodels = num_submodels)
            final_testset = self.make_final_committee_dataset(test_results, self.test_dataset, mode=mode, num_submodels = num_submodels)


            #initialise the final committee voting model

            correct_label = tf.placeholder(tf.float32, shape=[num_classes]) #aka the target variable

            #initialise the input variable aka the big X
            submodel_prediction_matrix = tf.placeholder(tf.float32, shape=[1,
                                                                        num_submodels*num_classes])  # TODO shape will expand [num_classes, num_models]
            learning_rate = tf.placeholder(tf.float32, shape=[])
            weights = tf.Variable(tf.random_uniform([num_submodels*num_classes, num_classes], -0.5, 0.5)) #aka the big W
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
                train_step = tf.train.AdagradOptimizer(learning_rate=l_rate).minimize(cross_entropy)
                # train_step = tf.train.AdadeltaOptimizer(learning_rate = l_rate).minimize(cross_entropy)

                sess.run(tf.initialize_all_variables())
                for epoch in range(num_epochs):
                    shuffle(final_trainset)
                    # Writing the code for training. It is not required to use a batch with size larger than one.
                    for i, (input, label) in enumerate(final_trainset):
                        # Run one step of SGD to update word embeddings.
                        train_step.run(feed_dict={submodel_prediction_matrix: input, correct_label: label})
                        #print(l_rate)
                    # The following line computes the accuracy on the development dataset in each epoch.
                    print('Epoch %d : %s .' % (
                    epoch, compute_final_accuracy(accuracy, submodel_prediction_matrix, correct_label, final_devset)))

                # uncomment the following line in the grading lab for evaluation
                # print('Accuracy on the test set : %s.' % compute_accuracy(accuracy, input_page_A, input_page_B , correct_label, self.test_dataset))
                # iinput_page_A, input_page_B are the placeholders for two input webpages.
                print('Committee model weight matrix: ',sess.run(weights))
                dev_results = final_predict(prediction, submodel_prediction_matrix, final_devset)
                test_results = final_predict(prediction, submodel_prediction_matrix, final_testset)

        elif mode == 'simple_sum':
            train_results = np.concatenate(submodel_train_results, axis=1)
            print(train_results.shape)
            dev_results = np.concatenate(submodel_dev_results, axis=1)

            test_results = np.concatenate(submodel_test_results, axis=1)
            final_trainset = self.make_final_committee_dataset(train_results, self.train_dataset, mode=mode, num_submodels = num_submodels)
            final_devset = self.make_final_committee_dataset(dev_results, self.dev_dataset, mode=mode, num_submodels = num_submodels)
            final_testset = self.make_final_committee_dataset(test_results, self.test_dataset, mode=mode, num_submodels = num_submodels)

            # initialise the final committee voting model

            correct_label = tf.placeholder(tf.float32, shape=[num_classes])  # aka the target variable

            # initialise the input variable aka the big X
            submodel_prediction_matrix = tf.placeholder(tf.float32, shape=[num_submodels,
                                                                            num_classes])  # TODO shape will expand [num_classes, num_models]
            with tf.Session() as sess:
                y = tf.reduce_mean(submodel_prediction_matrix, reduction_indices=[0])

                # change y result from matrix to string labels
                prediction = tf.cast(tf.argmax(y, 0), tf.int32)
                actual = tf.cast(tf.argmax(correct_label, 0), tf.int32)
                correct_prediction = tf.equal(prediction, actual)
                accuracy = tf.cast(correct_prediction, tf.float32)


                sess.run(tf.initialize_all_variables())
                for epoch in range(1):
                    shuffle(final_trainset)
                    # The following line computes the accuracy on the development dataset in each epoch.
                    print('Epoch %d : %s .' % (
                        epoch,
                        compute_final_accuracy(accuracy, submodel_prediction_matrix, correct_label, final_devset)))

                # uncomment the following line in the grading lab for evaluation
                # print('Accuracy on the test set : %s.' % compute_accuracy(accuracy, input_page_A, input_page_B , correct_label, self.test_dataset))
                # iinput_page_A, input_page_B are the placeholders for two input webpages.
                dev_results = final_predict(prediction, submodel_prediction_matrix, final_devset)
                test_results = final_predict(prediction, submodel_prediction_matrix, final_testset)

        write_result_file(test_results, 'ALTA2016 Result' + mode)

        return dev_results, test_results

    def make_final_committee_dataset(self, submodel_results: list, dataset:list, mode:str, num_submodels: int):
        '''
        A helper method within the class to prepare for the final result
        :return:
        '''

        final_data =[]

        if mode=='weight':
            for i, (page_A, page_B, label) in enumerate(dataset):
                input_vector = submodel_results[i, :].reshape(1, num_submodels*num_classes)

                assert input_vector.shape == (1, num_submodels*num_classes)

                final_data.append((input_vector, label))
        elif mode == 'simple_sum':
            for i, (page_A, page_B, label) in enumerate(dataset):
                input_vector = submodel_results[i,:,:]
                #print(input_vector, np.sum(input_vector, 1))

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


def compute_pre_emb_accuracy(accuracy, input_page_A, input_page_B, correct_label, eval_dataset):
    num_correct = 0
    for (page_A, page_B, label) in eval_dataset:
        num_correct += accuracy.eval(feed_dict={input_page_A: page_A,  input_page_B: page_B, correct_label: label})
    print('#correct submodel classification is %s ' % num_correct)
    return num_correct / len(eval_dataset)

def calculate_y(y, input_page_A, input_page_B, dataset):
    results=[]
    for (page_A, page_B, dummy_label) in dataset:
        results.append(y.eval(feed_dict={input_page_A: page_A,  input_page_B: page_B}))
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



def write_result_file(test_results, result_file):
    with open(result_file, mode='w') as f:
         for r in test_results:
             f.write("%d\n" % r)

def main():
    tnt = ALTATrainAndTest() #train And Test instance
    tnt.main_model(mode='weight') #inside of this, it will get the train result of all the sub models
    # tnt.main_model(mode ='simple_sum')
    return

if __name__ == "__main__":
    main()
