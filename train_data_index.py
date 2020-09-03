# encoding=utf-8

"""Converting vector to matrix"""

import pickle
import numpy as np
import re

BASE_PATH = "/home/yusu/GPCR/"
#BASE_FILE = "MyWord2vec"

embedding_size = 21

embeddings = [[0] * embedding_size]  # creates an initialized matrix of zeroes
char_ix = {'#PADDING#': 0}  # create an initialized dictionary of key=#PADDING and value=0

sequences_num = {"GPCR": 7106}  # Total number of sequences for training

#class_num = {"GPCR": 5}  # Total number of family classes of sequence
class_num = {"COG": 86}  # Total number of sub_sub family classes of sequence

# L -0.481728 -0.250347 -0.448789 0.186062 0.422846 -0.193202 -0.262873 0.434215 0.184949 0.144890 0.007216 0.007632 0.191    272 -0.219945 -0.212097 0.119401 -0.186181 0.395970 0.174094 -0.307764 -0.347458 -0.057727 0.179733 0.249983 0.003792 -0    .108843 -0.135170 0.008776 0.186450 -0.060635 -0.195090 0.023070 0.013137 0.148716 -0.132770 0.051153 0.107155 -0.219902     0.016620 0.114833 -0.045999 -0.008222 -0.043825 0.007982 -0.019792 0.047939 -0.000411 0.070530 0.016786 0.005368 -0.043    620 0.009384 -0.050344 0.147947 -0.068051 0.028818 0.020639 -0.141691 -0.110414 0.027665 -0.023237 0.087213 0.034169 -0.    034874 0.014539 -0.021719 -0.017274 -0.051435 0.064698 0.028225 0.256347 -0.117346 0.053821 0.004764 -0.037029 -0.015553     -0.196308 -0.010695 -0.156272 -0.110649 0.238234 0.013338 0.156936 -0.022130 0.129191 0.007791 0.049640 0.023007 0.1004    83 0.055872 0.146082 0.003860 0.034772 -0.077951 0.145148 0.026578 -0.177309 -0.818680 -0.238838 0.362554 0.159508 -0.08    0362 0.229137 0.044086 -0.083908 0.053543 -0.065141 -0.086398 -0.124056 0.190212 -0.088928 -0.326103 0.030491 0.031978 -    0.018375 0.090927 0.082926 0.103743 -0.136867 -0.329194 0.034571 0.066067 0.109040 0.097326 -0.146479 0.163602 -0.066253     -0.17989
with open('embeddings.21', 'r') as file:
    file.readline()
    # print(file.readline())
    # Add counter to each word vector in the embedding using the "enumerate" command
    for i, line in enumerate(file.readlines()):
        # print (i, line) #check
        cols = line.split()  # split each wordvector (sentence) embedding into a list(of words)
        # print(cols)#check
        c = cols[0]  # These are the columns of letters in the word2vec embeddings
        # print(c) #check
        v = cols[1:]  # These are the columns of vectors in the word2vec embeddings
        # print(v) #check
        char_ix[
            c] = i + 1  # Indexing the first letter in the embedding from 1. Remember: '#PADDING#': 0. So, the next key in the dictionary cant be zero.
        # print(char_ix[c])#check
        # print(char_ix) #check
        embeddings.append(v)  # appends vectors to the matrix called embeddings
    # print(embeddings) #check

# print(char_ix['L'])#check
assert len(char_ix) == len(embeddings)  # check if the number of char_ix equals number of embbeddings

"""convert list wordvector to sentence word vector"""
embeddings = np.array(embeddings, dtype=np.float32)
# print(embeddings) #check
# print(embeddings[1]) #check
np.save('embeddings_train.npy', embeddings)
# pickle char_ix

with open('char_ix_train.pkl', 'wb') as file:
    pickle.dump(char_ix, file)


# =================================
def save_file_smy(class_tag):
    print(class_tag)
    protein_file = BASE_PATH + "train_sequence.txt".format(class_tag)
    # print(protein_file)
    label_file = BASE_PATH + "train_label.pkl".format(class_tag)

    N = int(sequences_num[class_tag])
    # print(N) #check
    M = 1000  # Maximum length of a sequence
    all_sequences = np.zeros((N, M), dtype=np.int32)
    # print(all_sequences)#check

    with open(label_file, "rb") as fp_label:
        labels = pickle.load(fp_label)
        all_labels = np.array(labels, dtype=np.float32)

    with open(protein_file) as fp_seq:
        i = 0
        for line in fp_seq:
            seqence1 = line.split()[-1].replace('B', '')
            seqence2 = seqence1.replace('O', '')
            seqence3 = seqence2.replace('J', '')
            seqence4 = seqence3.replace('U', '')
            seqence5 = seqence4.replace('Z', '')
            seqence6 = seqence5.replace('b', '')
            seqence8 = seqence6.strip("'")
            seqence = seqence8.strip('_')
            #print(seqence) #check
            for j, c in enumerate(seqence[:M]):
                # print(j,c) #check
                ix = char_ix[c]
                # print(ix) #check
                all_sequences[i][j] = ix
            i += 1
    print("Total number of train sequences = " + str(len(all_sequences)))
    print(all_sequences)  # check
    # print(all_sequences[0]) #calling a row in marix
    # print(all_sequences[0,6])#calling an element in a matrix
    with open('{0}_all_train_sequences.pkl'.format(class_tag), "wb") as fp:
        pickle.dump(all_sequences, fp)

    np.save('{0}_all_train_labels.npy'.format(class_tag), all_labels)  # all_labels
    print("Total number of train Labels" + "(" + "outputs" + ")" + " = " + str(len(all_labels)))
    print(all_labels)


if __name__ == '__main__':
    # print(BASE_PATH)
    #print(embeddings)
    for key in sequences_num.keys():
        # print(key)
        save_file_smy(key)
