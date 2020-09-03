import os
import pickle
import string
import re

#os.environ["CUDA_VISIBLE_DEVICES"] = "4"

BASE_FILE = "/home/yusu/GPCR"

train_file = "{0}/cv0/train.txt".format(BASE_FILE)
test_file = "{0}/cv0/test.txt".format(BASE_FILE)

#train_file = "{0}/POG/train.txt".format(BASE_FILE)
#test_file = "{0}/POG/test.txt".format(BASE_FILE)

#train_file = "{0}/COG-500-1074/dataset0/train.txt".format(BASE_FILE)
#test_file = "{0}/COG-500-1074/dataset0/test.txt".format(BASE_FILE)

train_label_file = "{0}/train_label.pkl".format(BASE_FILE)
test_label_file = "{0}/test_label.pkl".format(BASE_FILE)

train_sequence_file = "{0}/train_sequence.txt".format(BASE_FILE)
test_sequence_file = "{0}/test_sequence.txt".format(BASE_FILE)

all_labels = []
all_labels_1 = []

# count = 0
with open(train_file, "rb") as fp, open(train_sequence_file, "w") as fp_w:
    for line in fp:
        idx, sequence = line.strip().split()
        # print(idx)#check
        # print(idx,sequence)#check
        label = [0 for i in
                 range(86)]  # create a vector of 86 zeroes. 86 is the number of classes of proteins for my practice
        # label[int(idx)-1] = 1 #Use this line of code if sequence label starts from 1
        label[int(idx)] = 1  # Use this line of code if sequence label starts from 0

        all_labels.append(label)

        fp_w.write("{0}\n".format(sequence))

        # count += 1
        # if count==3:
        # break

count = 1
for one_hot in all_labels:
    print(count, one_hot)
    count += 1
# print(len(one_hot))
with open(train_label_file, "wb") as fp_w:
    pickle.dump(all_labels, fp_w, protocol=2)

# Repeat the same for test set
with open(test_file, "rb") as fp, open(test_sequence_file, "w") as fp_w:
    for line in fp:
        idx, sequence = line.strip().split()
        # print(idx)#check
        # print(idx,sequence)#check
        label = [0 for i in
                 range(86)]  # create a vector of 86 zeroes. 86 is the number of classes of proteins for my practice
        # label[int(idx)-1] = 1 #Use this line of code if sequence label starts from 1
        label[int(idx)] = 1  # Use this line of code if sequence label starts from 0

        all_labels_1.append(label)

        fp_w.write("{0}\n".format(sequence))

        # count += 1
        # if count==3:
        # break

count_1 = 1
for one_hot_1 in all_labels_1:
    print(count_1, one_hot_1)
    count_1 += 1
# print(len(one_hot))
with open(test_label_file, "wb") as fp_w:
    pickle.dump(all_labels_1, fp_w, protocol=2)
