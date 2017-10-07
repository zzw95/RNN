from collections import Counter
import random
import numpy as np
from string import punctuation


def load_text():
    with open('data/reviews.txt') as f:
        reviews = f.read()
    with open('data/labels.txt') as f:
        labels = f.read()
    return reviews, labels

def preprocess(text):
    print("Remove punctuation......")
    text_p = text.lower()
    text_p = ''.join([c for c in text_p if c not in punctuation])
    lines = text_p.split('\n')
    words = ' '.join(lines).split()
    return lines, words

def create_lookup_tabels(words):
    word_counter = Counter(words)
    vocab_to_int = {}
    int_to_vocab = {}
    for index,(word, cnt) in enumerate(word_counter.most_common()):
        vocab_to_int[word] = index+1
        int_to_vocab[index+1] = word
    return vocab_to_int, int_to_vocab

def split_data(datasets, labels, train_size, valid_size, test_size):
    assert (train_size+valid_size+test_size)<=1
    assert datasets.shape[0]==labels.shape[0]
    num = datasets.shape[0]
    train_idx = int(num*train_size)
    valid_idx = int(num*(train_size+valid_size))
    test_idx = int(num*(train_size+valid_size+test_size))
    train_datasets = datasets[:train_idx]
    train_labels = labels[:train_idx]
    valid_datasets = datasets[train_idx:valid_idx]
    valid_labels = labels[train_idx:valid_idx]
    test_datasets = datasets[valid_idx:test_idx]
    test_labels = labels[valid_idx:test_idx]
    print("Train set: \t\t{}".format(train_datasets.shape),
      "\nValidation set: \t{}".format(valid_datasets.shape),
      "\nTest set: \t\t{}".format(test_datasets.shape),
      "\nAll set: \t\t{}".format(datasets.shape))
    return train_datasets, train_labels, valid_datasets, valid_labels, test_datasets, test_labels

def get_batches(datasets, labels, batch_size):
    n_batches = len(datasets) // batch_size
    for i in range(n_batches):
        batch_datasets = datasets[i*batch_size:(i+1)*batch_size]
        batch_labels = labels[i*batch_size:(i+1)*batch_size]
        yield  batch_datasets, batch_labels

def shuffle_list(data):
    ri=np.random.permutation(len(data))
    data=[data[i] for i in ri]
    return data

def standardize(datasets):
    #subtract mean
    centerized=datasets - np.mean(datasets, axis = 0)
    #divide standard deviation
    normalized=centerized/np.std(centerized, axis = 0)
    return normalized

def loadXY():
    X=np.load('data/X.npy',encoding="latin1")
    Y=np.load('data/Y.npy',encoding="latin1")
    print("loading data ......")
    print("Samples: {}".format(X.shape[0]))
    print("Input shape: {}".format(X[0].shape))
    print("Labels shape: {}".format(Y[0].shape))
    n_samples = X.shape[0]
    D_input = X[0].shape[1]
    D_output = Y[0].shape[1]
    data = []
    print("Standardizing ......")
    for x,y in zip(X,Y):
        data.append([standardize(x).reshape([1,-1,D_input]).astype("float32"),
                     standardize(y).astype("float32")])
    return data
