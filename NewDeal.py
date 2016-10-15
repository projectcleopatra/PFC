import re
from os import environ
import os
from urllib.parse import urlparse
import editdistance
import gensim
import nltk
import numpy as np
from scipy.spatial.distance import cosine
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from Utils import getSumVectors, csvSmartReader

train_snippet_file_name = "alta16_kbcoref_train_search_results.csv"
test_snippet_file_name = "alta16_kbcoref_test_search_results.csv"
snippet_file_schema = ['id', 'AUrl', 'ATitle', 'ASnippet', 'BUrl', 'BTitle', 'BSnippet']
train_label_file_name = "alta16_kbcoref_train_labels.csv"
label_file_schema = ['id', 'outcome']
num_classes = 2


#load the named entity recogniser
'''
# NER is too slow
environ['CLASSPATH'] = os.path.dirname(
    os.path.abspath(__file__)) + "/stanford-ner-2015-12-09/"  # bchu: because it's originally written in java
ner = nltk.StanfordNERTagger(
    'stanford-ner-2015-12-09/classifiers/english.all.3class.distsim.crf.ser.gz')
'''


#load pretrained embedding model

# '''
stanford_embed = gensim.models.Word2Vec.load_word2vec_format(
    './pretrained_embeddings/glove.840B.300d.bin', binary=False)
print("Stanford GloVe has been loaded ")

# '''

# '''
google_embed = gensim.models.Word2Vec.load_word2vec_format(
    './pretrained_embeddings/GoogleNews-vectors-negative300.bin', binary=True)
print("Model Pretrained on Google News has been loaded ")

# '''


def getTargetValues(csvReader) -> []:
    labels = []
    for row in csvReader:
        labels.append(int(row['outcome'].strip()))
    assert len(labels) != 0
    return labels


def makeURLFeatures(url_pairs: tuple)->[]:
    features = [1]
    url_a, url_b = url_pairs

    features.append(editdistance.eval(url_a, url_b))
    assert len(features) == 2
    return features


def makeSnippetFeatures(snippet_pairs:tuple)->[]:
    features = []

    #simple vector cosine distance
    snippet_a, snippet_b = snippet_pairs

    vec_a = getSumVectors(snippet_a, embed_matrix=stanford_embed)
    vec_b = getSumVectors(snippet_b, embed_matrix=stanford_embed)
    features.append(cosine(vec_a, vec_b))

    vec_a = getSumVectors(snippet_a, embed_matrix=google_embed)
    vec_b = getSumVectors(snippet_b, embed_matrix=google_embed)
    features.append(cosine(vec_a, vec_b))


    #Word Mover distance
    stopwords = nltk.corpus.stopwords.words('english')
    snippet_a_tokens = [w for w in nltk.word_tokenize(snippet_a) if w not in stopwords]
    snippet_b_tokens = [w for w in nltk.word_tokenize(snippet_b) if w not in stopwords]
    features.append(stanford_embed.wmdistance(snippet_a_tokens, snippet_b_tokens))
    features.append(google_embed.wmdistance(snippet_a_tokens, snippet_b_tokens))

    return features


def handleQuirkyTitles(entity: str) -> str:

    if "| LinkedIn" in entity or "|LinkedIn" in entity:
        head, sep, tail = entity.partition("|")
    elif "Twitter" in entity:
        head, sep, tail = entity.partition("(@")
    elif "IMDb" in entity:
        head, sep, tail = entity.partition("-")
    else:
        head, sep, tail = entity.partition("-")
        head, sep, tail = head.partition(":")
        head, sep, tail = head.partition("...")
        head, sep, tail = head.partition("|")

    return head.strip()


def makeTitleFeatures(title_pairs:tuple)->[]:
    features = []
    title_a, title_b = title_pairs


    # Edit distance feature

    # handle linkedin, IMDB
    entity_a = handleQuirkyTitles(title_a)
    entity_b = handleQuirkyTitles(title_b)

    # print(entity_a, entity_b)
    features.append(int(editdistance.eval(entity_a, entity_b)))

    # Embedded vectors
    vec_a = getSumVectors(title_a, embed_matrix=stanford_embed)
    vec_b = getSumVectors(title_b, embed_matrix=stanford_embed)

    cosine_distance = cosine(vec_a, vec_b)
    features.append(cosine_distance)

    vec_a = getSumVectors(title_a, embed_matrix=google_embed)
    vec_b = getSumVectors(title_b, embed_matrix=google_embed)

    cosine_distance = cosine(vec_a, vec_b)
    features.append(cosine_distance)


    return features

def makeFeatures(csvReader)->[[]]:
    features = []
    for row in csvReader:

        features.append(makeURLFeatures((row['AUrl'], row['BUrl']))
                        + makeTitleFeatures((row['ATitle'], row['BTitle']))
                        + makeSnippetFeatures((row['ASnippet'], row['BSnippet'])))

    assert len(features) != 0
    print("Feature size", len(features[0]))
    return features


class NewDeal:

    def __init__(self):


        self.train_labels = getTargetValues(csvSmartReader(train_label_file_name, label_file_schema))
        self.train_features = makeFeatures(csvSmartReader(train_snippet_file_name, snippet_file_schema))


        '''
        #Experiment with different layer parameters
        all_scores = []
        for layer1 in range(30,50):
            for layer2 in range(5, 40):
                clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(layer1, layer2), random_state=1)
                scores = cross_val_score(clf, self.train_features, self.train_labels, cv=10)
                all_scores.append((scores.mean(), scores.std()*2, layer1, layer2))
        print(all_scores)
        '''

        #automatically cross validate

        dt_stump = DecisionTreeClassifier(max_depth=2, min_samples_leaf=1)

        #
        #

        # http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_hastie_10_2.html#sphx-glr-auto-examples-ensemble-plot-adaboost-hastie-10-2-py
        clf = AdaBoostClassifier(
            base_estimator=dt_stump,
            learning_rate= 0.1,
            n_estimators=1200,
            algorithm="SAMME.R")
        # Cross validation score result

        clf2 = svm.SVC(kernel='rbf', C=1)  # ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
        clf3 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(40), random_state=1)

        scores = cross_val_score(clf, self.train_features, self.train_labels, cv=10)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        scores2 = cross_val_score(clf, self.train_features, self.train_labels, cv=10)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))

        scores3 = cross_val_score(clf, self.train_features, self.train_labels, cv=10)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores3.mean(), scores3.std() * 2))

        # Fit the model on the entire dataset
        clf.fit(self.train_features, self.train_labels)
        clf2.fit(self.train_features, self.train_labels)
        clf3.fit(self.train_features, self.train_labels)
        #Apply it to the tests dataset
        self.test_features = makeFeatures(csvSmartReader(test_snippet_file_name, snippet_file_schema))
        #results = clf.predict(self.test_features)

        results = clf.predict(self.train_features)
        results2 = clf2.predict(self.train_features)
        results3 = clf3.predict(self.train_features)

        for i, (result, target) in enumerate(zip(results, self.train_labels)):
            if str(result) != str(target):
                print(i, "is predicted to be",result, "but should be", target)

        for i, (result, target) in enumerate(zip(results2, self.train_labels)):
            if str(result) != str(target):
                print(i, "is predicted to be",result, "but should be", target)

        for i, (result, target) in enumerate(zip(results3, self.train_labels)):
            if str(result) != str(target):
                print(i, "is predicted to be",result, "but should be", target)

        return


    

    







def main():
    nd = NewDeal()


if __name__ == "__main__":
    main()