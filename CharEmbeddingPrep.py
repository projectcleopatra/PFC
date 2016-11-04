import string

import numpy as np

from Utils import csvSmartReader, extractURLFeatures

num_classes = 2
embedding_dim_2 = 18
monitored_chars =  string.ascii_lowercase + ' '

char_to_id = dict()

for char in monitored_chars:
    char_to_id[char] = len(char_to_id)



def create_label_vec(label: str):
    '''
    Creates target vectors from csv label '0' or '1'
    :param label: String
    :return:
    '''
   # Generate a label vector for a given classification label.
    '''
    label_vec = np.zeros(num_classes)
    label_vec[int(label)] = 1
    return label_vec
    '''
    return int(label)


def map_content_to_char_id_seq(content, char_to_id):
    try:
        assert len(char_to_id) == len(monitored_chars)
    except:
        print(len(char_to_id), 'vs', len(monitored_chars))

    id_seq= []
    content = content.lower() #optional
    for char in content:
        if char in monitored_chars:
            id_seq.append(char_to_id[char])
    return id_seq


def read_labeled_dataset_2(input_file_name, input_file_schema, label_file_name, label_file_schema, char_to_id):
    input_file_reader = csvSmartReader(input_file_name, input_file_schema)
    label_file_reader = csvSmartReader(label_file_name, label_file_schema)

    data = []
    for input,label in zip(input_file_reader, label_file_reader):
        content_A = extractURLFeatures(input['AUrl']) + input['ATitle'] + input['ASnippet'] #TODO
        content_B = extractURLFeatures(input['BUrl']) + input['BTitle'] + input['BSnippet'] #TODO
        word_id_seq_A = map_content_to_char_id_seq(content_A, char_to_id)
        word_id_seq_B = map_content_to_char_id_seq(content_B, char_to_id)

        data.append((word_id_seq_A, word_id_seq_B, create_label_vec(label['outcome'].strip('\n'))))
    print("read %d sentences from %s ." % (len(data), input_file_name))
    return data


def read_unlabeled_dataset_2(input_file_name, input_file_schema, char_to_id):
    input_file_reader = csvSmartReader(input_file_name, input_file_schema)
    data = []
    for input in input_file_reader:
        content_A = extractURLFeatures(input['AUrl']) # input['ATitle'] + input['ASnippet']  # TODO
        content_B = extractURLFeatures(input['BUrl']) # input['BTitle'] + input['BSnippet']  # TODO
        word_id_seq_A = map_content_to_char_id_seq(content_A, char_to_id)
        word_id_seq_B = map_content_to_char_id_seq(content_B, char_to_id)
        #it was: data.append(word_id_seq) without tuples
        data.append((word_id_seq_A, word_id_seq_B, [0,0,0]))
    print("read %d sentences from %s ." % (len(data), input_file_name))
    return data