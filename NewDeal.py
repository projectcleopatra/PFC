import os
from os import environ

import editdistance
import gensim
import nltk
from scipy.spatial.distance import cosine
from sklearn import svm, preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from Utils import getSumVectors, csvSmartReader, detectCountryCodeDifference, write_result_file

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
# adding mover distance from Stanford does increases accuracy

stanford_embed = gensim.models.Word2Vec.load_word2vec_format(
    './pretrained_embeddings/glove.840B.300d.bin', binary=False)
print("Stanford GloVe has been loaded ")


# '''
environ['CLASSPATH'] = os.path.dirname(
    os.path.abspath(__file__)) + "/stanford-postagger-full-2015-12-09/"
pos_tagger = nltk.StanfordPOSTagger('stanford-postagger-full-2015-12-09/models/english-left3words-distsim.tagger')
# '''


# '''
google_embed = gensim.models.Word2Vec.load_word2vec_format(
    './pretrained_embeddings/GoogleNews-vectors-negative300.bin', binary=True)
print("Model Pretrained on Google News has been loaded ")

# '''





def getTargetValues(csvReader, test) -> []:
    labels = []
    for row in csvReader:
        labels.append(int(row['outcome'].strip()))
        #reverse url and urlb, result should be the same
        if test == False:
            labels.append(int(row['outcome'].strip()))
    assert len(labels) != 0
    return labels

# ============== URL ====================


def isEducation(url: str):
    if ".ac.uk" in url or ".edu" in url:
        return 1
    else:
        return 0

def isEntertainment(url: str):
    if "imdb" in url or "allmusic" in url or "artnet" in url or "mtv.com" in url or "band" in url:
        return 1
    else:
        return 0
def isProfessional(url: str):
    if "linkedin" in url or "researchgate.com" in url:
        return 1
    else:
        return 0
def isNonProfitOrGov(url:str):
    if ".org" in url or ".gov" in url:
        return 1
    else:
        return 0
def isSportsStar(url: str):
    if "espn" in url or "ufc.com" in url or "sports" in url:
        return 1
    else:
        return 0





def careerFeatures(url_pair: tuple)->int:
    same = 0 #assume people are from different field
    url_a, url_b = url_pair
    url_a_features = [isEducation(url_a), isEntertainment(url_a), isProfessional(url_a), isNonProfitOrGov(url_a), isSportsStar(url_a) ]
    url_b_features = [isEducation(url_b), isEntertainment(url_b), isProfessional(url_b), isNonProfitOrGov(url_b), isSportsStar(url_b) ]

    return url_a_features + url_b_features



def makeURLFeatures(url_pair: tuple)->[]:
    features = [1] #Feature 1
    url_a, url_b = url_pair
    # Feature 2
    features.append(detectCountryCodeDifference(url_pair)) #1 if one url has uk and another has cz, otherwise 0
    # Feature 3
    features.extend(careerFeatures(url_pair))
    features.append(editdistance.eval(url_a, url_b))
    #assert len(features) == 3
    return features

# ============== Snippet ====================


def makeSnippetFeatures(snippet_pairs:tuple)->[]:
    features = []

    #simple vector cosine distance
    snippet_a, snippet_b = snippet_pairs
    snippet_a = snippet_a.replace("/"," ")
    snippet_b = snippet_b.replace("/", " ")

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
    """
    tagged_sequence_a = pos_tagger.tag(snippet_a_tokens)
    tagged_sequence_b = pos_tagger.tag(snippet_b_tokens)
    rel_pos = ["NNS", "NN", "NNP", "NNPS"]
    relevant_tokens_a = [token for token, tag in tagged_sequence_a if tag in rel_pos]
    relevant_tokens_b = [token for token, tag in tagged_sequence_b if tag in rel_pos]
    # print(len(relevant_tokens_a), "vs", len(snippet_a_tokens))
    # print(len(relevant_tokens_b), "vs", len(snippet_b_tokens))
    st_noun_wmd = stanford_embed.wmdistance(relevant_tokens_a, relevant_tokens_b)
    if st_noun_wmd == float('Inf') or st_noun_wmd == float('NaN'): #Find out why WMD appears unusual
        print(st_noun_wmd)
        print(relevant_tokens_a, "vs",relevant_tokens_b)
        st_noun_wmd = 0.0
    features.append(st_noun_wmd)

    features.append(stanford_embed.wmdistance(relevant_tokens_a, relevant_tokens_b))
    g_noun_wmd = google_embed.wmdistance(relevant_tokens_a, relevant_tokens_b)
    if g_noun_wmd == float('Inf') or g_noun_wmd == float('NaN'): #Find out why WMD appears unusual
        print(g_noun_wmd)
        print(relevant_tokens_a, "vs",relevant_tokens_b)
        g_noun_wmd = 0.0

    features.append(g_noun_wmd)
    """

    # ----------------------
    # New Metric -> number of tokens that are contained in the other list?
    #features.append(count_similar_words(relevant_tokens_a, relevant_tokens_b, google_embed))
    #features.append(count_similar_words(relevant_tokens_a, relevant_tokens_b, stanford_embed))

    return features

# ---------------------- Delegate method for counting similar words
def count_similar_words(relevant_tokens_a, relevant_tokens_b, embed_file):
    metric = 0
    # First, expand the tokens to include synonyms
    expanded_tokens_a = set(relevant_tokens_a)
    most_similar_words = set([])
    for token in relevant_tokens_a:
        print("Scanning similar words for", token)
        if token in embed_file.vocab:
            most_similar_words, _ = zip(*embed_file.most_similar(positive=[token], negative=[], topn=5))
            most_similar_words = set(most_similar_words)
            # if token in stanford_embed.vocab:
            # most_similar_words.union(set(zip(*stanford_embed.most_similar(positive=[token]))))
        expanded_tokens_a.union(most_similar_words)
    most_similar_words = set([])
    expanded_tokens_b = set(relevant_tokens_b)
    for token in relevant_tokens_b:
        print("Scanning similar words for", token)
        if token in embed_file.vocab:
            most_similar_words, _ = zip(*embed_file.most_similar(positive=[token], negative=[], topn=5))
            most_similar_words = set(most_similar_words)
            # if token in stanford_embed.vocab:
            # most_similar_words.union(set(zip(*stanford_embed.most_similar(positive=[token]))))
        expanded_tokens_b.union(most_similar_words)

    # Find words in the expanded list that are included in another list
    for t1 in expanded_tokens_a:
        for t2 in expanded_tokens_b:
            if t1.lower() in t2.lower():
                metric += 1
            if t2.lower() in t1.lower():
                metric += 1
    return metric


# ============== Title ====================

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
    """
    # Embedded vectors
    vec_a = getSumVectors(title_a, embed_matrix=stanford_embed)
    vec_b = getSumVectors(title_b, embed_matrix=stanford_embed)

    cosine_distance = cosine(vec_a, vec_b)
    features.append(cosine_distance)
    """
    vec_a = getSumVectors(title_a, embed_matrix=google_embed)
    vec_b = getSumVectors(title_b, embed_matrix=google_embed)



    cosine_distance = cosine(vec_a, vec_b)
    features.append(cosine_distance)


    return features


# ============== Mixed ====================
def makeMixedFeatures(pairs_of_info: tuple) ->[]:
    #TODO


    
    return

# ============== Main Feature (Tie things together) ====================

def makeFeatures(csvReader, test)->[[]]:
    features = []
    num_trained = 1
    for row in csvReader:
        new_feature = makeURLFeatures((row['AUrl'], row['BUrl'])) \
                        + makeTitleFeatures((row['ATitle'], row['BTitle'])) \
                        + makeSnippetFeatures((row['ASnippet'], row['BSnippet']))
        features.append(new_feature)


        if test==False:
            #reverse 'AUrl' and 'BUrl'
            features.append(makeURLFeatures((row['BUrl'], row['AUrl']))  \
                            + makeTitleFeatures((row['BTitle'], row['ATitle'])) \
                            + makeSnippetFeatures((row['BSnippet'], row['ASnippet'])))
            print("Trained", num_trained, "training examples.",  )
        else:
            print("Trained", num_trained, "test examples.", )

        num_trained += 1

    assert len(features) != 0
    print("Feature size", len(features[0]))
    return features


class NewDeal:

    def __init__(self):


        self.train_labels = getTargetValues(csvSmartReader(train_label_file_name, label_file_schema), test=False)
        self.train_features = makeFeatures(csvSmartReader(train_snippet_file_name, snippet_file_schema), test = False)
        self.scaler = preprocessing.StandardScaler().fit(self.train_features)

        X = self.scaler.transform(self.train_features)

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




        scores = cross_val_score(clf, X, self.train_labels, cv=10)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        clf2 = svm.SVC(kernel='rbf', C=1)  # ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
        scores2 = cross_val_score(clf2, X, self.train_labels, cv=10)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))
        clf3 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(40), random_state=1)
        scores3 = cross_val_score(clf3, X, self.train_labels, cv=10)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores3.mean(), scores3.std() * 2))

        # Fit the model on the entire dataset
        clf.fit(X, self.train_labels)

        clf2.fit(X, self.train_labels)


        clf3.fit(X, self.train_labels)

        #Apply it to the tests dataset
        self.test_features = makeFeatures(csvSmartReader(test_snippet_file_name, snippet_file_schema), test=True)
        #results = clf.predict(self.test_features)

        results = clf.predict(self.scaler.transform(self.test_features))
        write_result_file(results, 'Results/boost_result.csv')

        for i, (result, target) in enumerate(zip(results, self.train_labels)):
            if str(result) != str(target):
                #manual correction
                print(i, "is predicted to be",result, "but should be", target)
        print("="*20)


        results2 = clf2.predict(self.scaler.transform(self.test_features))
        write_result_file(results2, 'Results/SVC_result.csv')
        results3 = clf3.predict(self.scaler.transform(self.test_features))
        write_result_file(results3, 'Results/MLP_result.csv')

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