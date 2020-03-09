from __future__ import print_function

import pickle
import numpy as np
import os
import sys
import time
import importlib
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, Embedding
from keras import optimizers
from keras.layers import Activation

import argparse

#config 
one_hot = True
embedding_size = 256  # is ignored if one_hot=True
num_layers = 3
rnn_size = 512
dropout = 0.2

learning_rate = 0.003
learning_rate_decay_after = 20
learning_rate_decay = 0.97

batch_size = 64
max_epoch = 100
grad_clipping = 5
validation_fraction = 0.05
validate_every = 100  # iterations

save_every = 1  # epochs

parser = argparse.ArgumentParser()
#parser.add_argument('metadata_path',type=str, default= "metadata_pickle_99.pkl")
parser.add_argument('--rng_seed', type=int, default=42)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--ntunes', type=int, default=15)
parser.add_argument('--seed')
parser.add_argument('--terminal', action="store_true")

args = parser.parse_args()

#metadata_path = args.metadata_path
rng_seed = args.rng_seed
temperature = args.temperature
ntunes = args.ntunes
seed = args.seed

def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    rms_opt = optimizers.RMSprop(lr=learning_rate, clipvalue=grad_clipping)
    model = Sequential()
    model.add(Embedding(input_dim=n_vocab, output_dim=n_vocab, embeddings_initializer='Identity'))
    model.add(LSTM(
        rnn_size,
        input_shape =(1,),
        return_sequences=True
    ))
    model.add(Dropout(dropout))
    model.add(LSTM(rnn_size,return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(rnn_size))
    model.add(Dropout(dropout))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=rms_opt)
    #model.summary()

    # Load the weights to each node
    model.load_weights('weights.hdf5')

    return model


with open('metadata_pickle_99.pkl','rb') as f:
    metadata = pickle.load(f)

token2idx = metadata['token2idx']
idx2token = dict((v, k) for k, v in token2idx.items())
vocab_size = len(token2idx)

x = np.zeros((ntunes,100),dtype='int32')

model = create_network(x,vocab_size)

start_idx, end_idx = token2idx['<s>'], token2idx['</s>']

rng = np.random.RandomState(rng_seed)
vocab_idxs = np.arange(vocab_size)

seed_sequence = [start_idx]
if seed is not None:
    for token in seed.split(' '):
        seed_sequence.append(token2idx[token])

for i in range(ntunes):
    sequence = seed_sequence[:]
    while sequence[-1] != end_idx:
        temp = model.predict(np.array([sequence], dtype='int32'))
        temp = np.reshape(temp, (vocab_size))
        next_itoken = rng.choice(vocab_idxs, p=temp)
        sequence.append(next_itoken)

    abc_tune = [idx2token[j] for j in sequence[1:-1]]
    if not args.terminal:
        f = open('samples.txt', 'a+')
        f.write('X:' + repr(i) + '\n')
        f.write(abc_tune[0] + '\n')
        f.write(abc_tune[1] + '\n')
        f.write(' '.join(abc_tune[2:]) + '\n\n')
        f.close()
    else:
        print('X:' + repr(i))
        print(abc_tune[0])
        print(abc_tune[1])
        print(' '.join(abc_tune[2:]) + '\n')