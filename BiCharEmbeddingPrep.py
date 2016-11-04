import re
import string

import numpy as np

from Utils import csvSmartReader, extractURLFeatures

num_classes = 2 #bchu: same person or not, which is a binary
embedding_dim_3 = 18 #bchu: random parameter. not too big due to our training data is not large enough
monitored_chars = string.ascii_lowercase #bchu: a-z

bichar_to_id= dict() #bchu: has 26 x 26 entries. If we add " " in the monitored_chars, then it will be 27 x 27
for a in monitored_chars:
    for b in monitored_chars:
        bichar_to_id[a+b] = len(bichar_to_id)


def create_label_vec(label: str): # bchu: passed label with: '0' or '1'
    '''
    Creates target vectors from csv label '0' or '1'
    :param label: String
    :return:
    '''
   # Generate a label vector for a given classification label.
    """
    label_vec = np.zeros(num_classes) # [0,0]
    label_vec[int(label)] = 1 #[1, 0] if label = '0' otherwise [0, 1]
    return label_vec
    """
    return int(label)


def map_content_to_char_id_seq(content: str, char_to_id: dict)-> []: #bchu: "blah b" -> ['bl', 'la','ah', ...]


    #remove invalid bichar characters. Including space: r'[^a-zA-Z\s]+
    content = re.sub(r'[^a-zA-Z]+', '', content).lower().strip() #bchu: eg ':// ' would be ignored

    id_seq= []
    for i in range(0, len(content) - 1):
        bichar = content[i]+content[i+1] #bchu: sliding window size is 2
        id_seq.append(char_to_id[bichar.lower()]) #bchu; the bichar id
    return id_seq


def read_labeled_dataset_3(input_file_name: str, input_file_schema: [] , label_file_name: str, label_file_schema, bichar_to_id):
    #bchu: input_file_schema: means the table header in all the .csv file, specifically for one file's
    input_file_reader = csvSmartReader(input_file_name, input_file_schema) # bchu: read the csv file
    label_file_reader = csvSmartReader(label_file_name, label_file_schema)

    data = []
    for input,label in zip(input_file_reader, label_file_reader): #bchu: logically row by row matching
        content_A = extractURLFeatures(input['AUrl']) + input['ATitle'] + input['ASnippet'] #bchu: concat three values retrived by the keys
        content_B = extractURLFeatures(input['BUrl']) + input['BTitle'] + input['BSnippet'] #bchu: comparing if doc A and doc B are the same
        word_id_seq_A = map_content_to_char_id_seq(content_A.replace('http://|https://', ''), bichar_to_id) #bchu; map a string into bichar ids
        word_id_seq_B = map_content_to_char_id_seq(content_B.replace('http://|https://', ''), bichar_to_id)

        data.append((word_id_seq_A, word_id_seq_B, create_label_vec(label['outcome'].strip('\n')))) #bchu: if the list entry is same or not
    print("read %d sentences from %s ." % (len(data), input_file_name))
    return data


def read_unlabeled_dataset_3(input_file_name, input_file_schema, bichar_to_id): #bchu: all data except the training data
    input_file_reader = csvSmartReader(input_file_name, input_file_schema)
    data = []
    for input in input_file_reader:
        content_A = extractURLFeatures(input['AUrl']) + input['ATitle'] + input['ASnippet']
        content_B = extractURLFeatures(input['BUrl']) + input['BTitle'] + input['BSnippet']
        word_id_seq_A = map_content_to_char_id_seq(content_A.replace('http://|https://', ''), bichar_to_id)
        word_id_seq_B = map_content_to_char_id_seq(content_B.replace('http://|https://', ''), bichar_to_id)
        #it was: data.append(word_id_seq) without tuples
        data.append((word_id_seq_A, word_id_seq_B, [0,0])) #bchu; if you don't have the label assigned, for testing data, you assign garbage
    print("read %d sentences from %s ." % (len(data), input_file_name))
    return data