# -*- encoding:utf8 -*-

import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers import Flatten, concatenate,add, Embedding
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras import regularizers
from keras import initializers
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.metrics import categorical_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from inspect import signature

# set the matplotlib backend so figures can be saved in the background
import matplotlib

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


matplotlib.use("Agg")

# import the necessary packages
# from pyimagesearch.stridednet import StridedNet
from sklearn.metrics import classification_report, precision_score, roc_auc_score, f1_score, recall_score, precision_recall_curve, average_precision_score, matthews_corrcoef, balanced_accuracy_score


# fix random seed for reproducibility
seed = 1234
np.random.seed(seed)

'''Load training Sequences embedding Matrix'''
with open('GPCR_all_train_sequences.pkl', "rb") as fp_seq:
    X_train = pickle.load(fp_seq)
print('Shape of train input tensor: ', X_train.shape)

'''Load training Output'''
GPCR_all_train_labels = np.load('COG_all_train_labels.npy')
labels_train= GPCR_all_train_labels
print('Shape of train output tensor: ', labels_train.shape)
#print(labels_train)

'''Load testing Sequences embedding Matrix'''
with open('GPCR_all_test_sequences.pkl', "rb") as fp_seq:
    X_test = pickle.load(fp_seq)
print('Shape of test input tensor: ', X_train.shape)

'''Load test Output'''
GPCR_all_test_labels = np.load('COG_all_test_labels.npy')
labels_test= GPCR_all_test_labels
print('Shape of test output tensor: ', labels_test.shape)
#print(labels_test)

'''compute an index mapping words to known embeddings, by parsing the data dump of pre-trained embeddings: embeddings_index = {}'''
with open('char_ix_train.pkl', "rb") as fp_label:
    # labels = np.load(fp_label)
    # print(labels)#check
    embeddings_index = pickle.load(fp_label)
#del embeddings_index['#PADDING#']
print(embeddings_index)
print('Found %s word vectors.' % len(embeddings_index))
word_index=embeddings_index
print('Found %s unique tokens.' % len(word_index))

'''Load word2vec embeddings'''
embedding_matrix = np.load('embeddings_train.npy')
#embedding_matrix=embedding_matrix[1:]
np.save('embedding_matrix.npy', embedding_matrix)
print('Found embedding matrix with dimension: ', embedding_matrix.shape)
EMBEDDING_DIM=21
MAX_SEQUENCE_LENGTH=1000
print(embedding_matrix)
''' Load embedding matrix into an Embedding layer. Note: we set trainable=False to prevent the weights from being updated during training'''
embedding_layer = Embedding(len(word_index), EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=True)
'''Build the Neural Network'''
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
conv1 = Conv1D(250, kernel_size=20, kernel_initializer='glorot_uniform', activation='relu')(embedded_sequences)
pool1 = MaxPooling1D(pool_size=981, strides=1000)(conv1)

conv2 = Conv1D(250, kernel_size=8, kernel_initializer='glorot_uniform', activation='relu')(embedded_sequences)
pool2 = MaxPooling1D(pool_size=993, strides=1000)(conv2)

conv3 = Conv1D(250, kernel_size=19, kernel_initializer='glorot_uniform', activation='relu')(embedded_sequences)
pool3 = MaxPooling1D(pool_size=982, strides=1000)(conv3)
#Merge by concatenation
#concat=concatenate([pool1, pool2], axis=-1):
merg=add([pool1, pool2, pool3])
drop1 = Dropout(rate=0.35)(merg)
concat=concatenate([pool1, drop1], axis=-1)
flat = Flatten()(concat)
drop2 = Dropout(rate=0.35)(flat)
hidden1 = Dense(2000, activation='relu')(drop2)
batch1 = BatchNormalization(axis=-1, momentum=0.99)(hidden1)
output = Dense(86, kernel_regularizer=regularizers.l1_l2(l1=0.0005, l2=0.0005), activation='softmax')(batch1)
model = Model(inputs=sequence_input, outputs=output)
# summarize layers
model.summary()
# plot graph
plot_model(model, to_file='DeepPff.png', show_shapes=True)
# compile network
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0005), metrics=['accuracy'])
# Early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('best_model_deepPff.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
# Fit model
history = model.fit(X_train, labels_train, validation_data=(X_test, labels_test), batch_size=100, epochs=20, verbose=1,
                    callbacks=[es, mc])  # allows access to model.fit for ploting
# Load saved model
saved_model = load_model('best_model_deepPff.h5')

# Evaluate the Model
_, train_acc = saved_model.evaluate(X_train, labels_train, verbose=0)
_, test_acc = saved_model.evaluate(X_test, labels_test, verbose=0)
print('Train Acc: %.3f, Test Acc: %.3f' % (train_acc * 100, test_acc * 100))  # Present results

start_time = time.time() #Prediction start time
# evaluate the network
print("[INFO] evaluating network...")
predictions = saved_model.predict(X_test)
print(classification_report(labels_test.argmax(axis=1),predictions.argmax(axis=1), digits=5)) #Summarizes precision, recall, f1 score

#precision_val =precision_score(labels_test.argmax(axis=1), predictions.argmax(axis=1), average='macro')
#f1_val = f1_score(labels_test.argmax(axis=1),predictions.argmax(axis=1), average='macro')
#recall_val = recall_score(labels_test.argmax(axis=1),predictions.argmax(axis=1), average='macro')
mcc_val = matthews_corrcoef(labels_test.argmax(axis=1), predictions.argmax(axis=1))
bacc_val = balanced_accuracy_score(labels_test.argmax(axis=1), predictions.argmax(axis=1), adjusted=True) #Defaul, adjusted=False

#print("macro-precision = %.5f" % (precision_val* 100))
#print("macro-recall = %.5f" % (recall_val* 100))
#print("macro-f1 = %.5f" % (f1_val* 100))
print("mathew's correlation coefficient = %.5f" % (mcc_val))
print("Balanced Accuracy = %.5f" % (bacc_val))

end_time = time.time()  # CNN_1 end time
print('Prediction Time taken = ', (end_time - start_time), 'seconds')

# Plot history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Accuracy_plot_DeepPff.png')

plt.clf()
# Plot history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Loss_plot_DeepPff.png')

plt.clf()
n_classes = labels_train.shape[1]
#print(n_classes)

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(labels_test[:, i],
                                                        predictions[:, i])
    average_precision[i] = average_precision_score(labels_test[:, i], predictions[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(labels_test.ravel(),
    predictions.ravel())
average_precision["micro"] = average_precision_score(labels_test, predictions,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.5f}'
      .format(average_precision["micro"]))

"""Ploting precision-recall curve"""
# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.figure()
plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
         where='post')
#plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b',
                 #**step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Average precision score, micro-averaged over all classes: AP={0:0.5f}'
    .format(average_precision["micro"]))
plt.savefig('PR_curve_DeepPff.png')
