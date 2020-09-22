# DeepPPF
A deep learning framework for predicting protein family.
# Usage
1. Preprocess the raw dataset by running the preprocessing.py file.
2. Then, compute the word2vec embedding by running the w2v.py file.
3. Index the amino acids of the protein sequences by running the train_data_index.py and test_data_index.py files.
4. Train and test the model by running the deepPPF.py file.
5. Finetune the model by running the model_transfer.py file.
# Data
1. data.zip contains the GPCR and POG protein sequences.
2. The COG protein can be downloaded at http://epigenomics.snu.ac.kr/DeepFam/data.zip
