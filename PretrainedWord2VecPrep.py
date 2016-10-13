import gensim
import numpy as np

from Utils import csvSmartReader, extractURLFeatures
import nltk

num_classes = 2
pretrained_model_news = gensim.models.Word2Vec.load_word2vec_format(
    './pretrained_embeddings/GoogleNews-vectors-negative300.bin', binary=True)
print("Model Pretrained on Google News has been loaded ")

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


def read_labeled_dataset_for_pretrained_embeddings(input_file_name, input_file_schema, label_file_name, label_file_schema):
    input_file_reader = csvSmartReader(input_file_name, input_file_schema)
    label_file_reader = csvSmartReader(label_file_name, label_file_schema)

    data = []
    for input,label in zip(input_file_reader, label_file_reader):
        content_A = " ".join([extractURLFeatures(input['AUrl']), input['ATitle'], input['ASnippet']]) #TODO
        content_B = " ".join([extractURLFeatures(input['BUrl']), input['BTitle'], input['BSnippet']]) #TODO

        token_seq_A = tokenize(content_A)
        token_seq_B = tokenize(content_B)

        token_vector_A = np.empty((0, 300), float)  # newscorp has dimensionality 300
        token_vector_B = np.empty((0, 300), float)

        for token in token_seq_A:
            try:
                token_vector_A = np.row_stack((token_vector_A, pretrained_model_news[token]))
            except KeyError:
                print(token, "doesn't exist")
        fasttext_vec_A = np.mean(token_vector_A, axis=0)
        assert fasttext_vec_A.shape[0] == 300

        for token in token_seq_B:
            try:
                token_vector_B = np.row_stack((token_vector_B, pretrained_model_news[token]))
            except KeyError:
                print(token, "doesn't exist")
        fasttext_vec_B = np.mean(token_vector_B, axis=0)
        assert fasttext_vec_B.shape[0] == 300

        data.append((fasttext_vec_A, fasttext_vec_B, create_label_vec(label['outcome'].strip('\n'))))
    print("read %d sentences from %s ." % (len(data), input_file_name))
    return data


def read_unlabeled_dataset_for_pretrained_embeddings(input_file_name, input_file_schema):
    input_file_reader = csvSmartReader(input_file_name, input_file_schema)
    data = []
    for input in input_file_reader:
        content_A = " ".join([extractURLFeatures(input['AUrl']), input['ATitle'], input['ASnippet']])  # TODO
        content_B = " ".join([extractURLFeatures(input['BUrl']), input['BTitle'], input['BSnippet']])  # TODO

        token_seq_A = tokenize(content_A)
        token_seq_B = tokenize(content_B)

        token_vector_A = np.empty((0, 300), float)  # newscorp has dimensionality 300
        token_vector_B = np.empty((0, 300), float)

        for token in token_seq_A:
            try:
                token_vector_A = np.row_stack((token_vector_A, pretrained_model_news[token]))
            except KeyError:
                print(token, "doesn't exist")
        fasttext_vec_A = np.mean(token_vector_A, axis=0)
        assert fasttext_vec_A.shape[0] == 300

        for token in token_seq_B:
            try:
                token_vector_B = np.row_stack((token_vector_B, pretrained_model_news[token]))
            except KeyError:
                print(token, "doesn't exist")
        fasttext_vec_B = np.mean(token_vector_B, axis=0)
        assert fasttext_vec_B.shape[0] == 300

        #it was: data.append(word_id_seq) without tuples
        data.append((fasttext_vec_A, fasttext_vec_B, [0,0,0]))
    print("read %d sentences from %s ." % (len(data), input_file_name))
    return data