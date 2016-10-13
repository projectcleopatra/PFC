import collections
import nltk
import numpy as np

from Utils import csvSmartReader

num_classes = 2
embedding_dim = 20

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


def map_token_seq_to_word_id_seq(token_seq, word_to_id, NER_only = False, ner = None):
    if NER_only:
        tagged_seq = ner.tag(token_seq)
        ne_token_seq = []
        for token, tag in tagged_seq:
            if tag != "O":
                ne_token_seq.append(token)
        token_seq = ne_token_seq

    return [map_word_to_id(word_to_id,word, NER_only, ner) for word in token_seq]

def map_word_to_id(word_to_id, word, NER_only = False, ner = None):
    # map each word to its id.
    if word in word_to_id:

        return word_to_id[word]
    else:
        return word_to_id['$UNK$']  #TODO

def build_vocab(input_file_name, input_file_schema, NER_only = False, ner = None):
    '''
    Important! This builds a list of vocabulary indexes from file
    :param sens_file_name:
    :return:
    '''
    data = []
    csvreader = csvSmartReader(input_file_name, input_file_schema)
    for input in csvreader:
        tokens = tokenize(" ".join([input['ATitle'], input['ASnippet'], input['BTitle'], input['BSnippet']]))
        data.extend(tokens)

    if NER_only:
        ner_data = []
        tagged_data = ner.tag(data)
        for token, tag in tagged_data:
            if tag != "O":
                ner_data.append(token)

        data = ner_data


    count = [['$UNK$', 0]]
    sorted_counts = collections.Counter(data).most_common()
    count.extend(sorted_counts)
    word_to_id = dict()
    for word, _ in count:
        word_to_id[word] = len(word_to_id)
    print('size of vocabulary is %s. ' % len(word_to_id))
    return word_to_id


def read_labeled_dataset(input_file_name, input_file_schema, label_file_name, label_file_schema, word_to_id, NER_only = False, ner = None):
    input_file_reader = csvSmartReader(input_file_name, input_file_schema)
    label_file_reader = csvSmartReader(label_file_name, label_file_schema)

    data = []
    for input,label in zip(input_file_reader, label_file_reader):
        content_A = input['ATitle'] + ' '+ input['ASnippet'] #TODO
        content_B = input['BTitle'] + ' '+ input['BSnippet'] #TODO
        word_id_seq_A = map_token_seq_to_word_id_seq(tokenize(content_A), word_to_id, NER_only, ner)
        word_id_seq_B = map_token_seq_to_word_id_seq(tokenize(content_B), word_to_id, NER_only, ner)

        data.append((word_id_seq_A, word_id_seq_B, create_label_vec(label['outcome'].strip('\n'))))
    print("read %d sentences from %s ." % (len(data), input_file_name))
    return data


def read_unlabeled_dataset(input_file_name, input_file_schema, word_to_id, NER_only = False, ner = None):
    input_file_reader = csvSmartReader(input_file_name, input_file_schema)
    data = []
    for input in input_file_reader:
        content_A = input['ATitle'] + input['ASnippet']  # TODO
        content_B = input['BTitle'] + input['BSnippet']  # TODO
        word_id_seq_A = map_token_seq_to_word_id_seq(tokenize(content_A), word_to_id, NER_only, ner)
        word_id_seq_B = map_token_seq_to_word_id_seq(tokenize(content_B), word_to_id, NER_only, ner)
        #it was: data.append(word_id_seq) without tuples
        data.append((word_id_seq_A, word_id_seq_B, [0,0,0]))
    print("read %d sentences from %s ." % (len(data), input_file_name))
    return data