import os
import sys
import time
import logger
import pickle
import importlib
import six
import random
from collections import defaultdict
from random import shuffle

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation,Reshape, Embedding, TimeDistributed
from keras import utils
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.utils import np_utils, Sequence
from keras import optimizers

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

save_every = 10  # epochs

class DataIterator(object):
    
    def __init__(self, tune_lens, tune_idxs, batch_size, random_lens=False):
        self.batch_size = batch_size
        self.ntunes = len(tune_lens)
        self.tune_idxs = tune_idxs

        self.len2idx = defaultdict(list)
        for k, v in zip(tune_lens, tune_idxs):
            self.len2idx[k].append(v)

        self.random_lens = random_lens
        self.rng = np.random.RandomState(42)
    
    def __iter__(self):
        if self.random_lens:
            for batch_idxs in self.__iter_random_lens():
                yield np.int32(batch_idxs)
        else:
            for batch_idxs in self.__iter_homogeneous_lens():
                yield np.int32(batch_idxs)
    
    def __iter_random_lens(self):
        available_idxs = np.copy(self.tune_idxs)
        while len(available_idxs) >= self.batch_size:
            rand_idx = self.rng.choice(range(len(available_idxs)), size=self.batch_size, replace=False)
            yield available_idxs[rand_idx]
            available_idxs = np.delete(available_idxs, rand_idx)

    def __iter_homogeneous_lens(self):
        for idxs in six.itervalues(self.len2idx):#.itervalues():
            self.rng.shuffle(idxs)

        progress = defaultdict(int)
        available_lengths = list(self.len2idx.keys())

        batch_idxs = []
        b_size = self.batch_size

        get_tune_len = lambda: self.rng.choice(available_lengths)
        k = get_tune_len()

        while available_lengths:
            batch_idxs.extend(self.len2idx[k][progress[k]:progress[k] + b_size])
            progress[k] += b_size
            if len(batch_idxs) == self.batch_size:
                yield batch_idxs
                batch_idxs = []
                b_size = self.batch_size
                k = get_tune_len()
            else:
                b_size = self.batch_size - len(batch_idxs)
                i = available_lengths.index(k)
                del available_lengths[i]
                if not available_lengths:
                    break
                if i == 0:
                    k = available_lengths[0]
                elif i >= len(available_lengths) - 1:
                    k = available_lengths[-1]
                else:
                    k = available_lengths[i + self.rng.choice([-1, 0])]


def create_network(x, n_vocab):
    """ create the structure of the neural network """
    rms_opt = optimizers.RMSprop(lr=learning_rate, clipvalue=grad_clipping)
    model = Sequential()
    model.add(Embedding(input_dim=n_vocab, output_dim=n_vocab, embeddings_initializer='Identity'))
    model.add(LSTM(
        rnn_size,
        input_shape =(100,),
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
    model.summary()

    return model

data_path = "data/allabcwrepeats_parsed"
with open(data_path, 'r', encoding='utf-8') as f:
    data = f.read()
    
def remove_title(tune):
    return ('\n').join(tune.split('\n')[1:])

tunes = data.split('\n\n')

# Remove all the titles to reduce vocab size
tunes = [remove_title(tune) for tune in tunes]

tokens_set = set('\n\n'.join(tunes).split())

#tokens_set = set(data.split())
start_symbol, end_symbol = '<s>', '</s>'
tokens_set.update({start_symbol, end_symbol})

idx2token = sorted(list(tokens_set)) # needs to be sorted to be the same after reloading the vocab
vocab_size = len(idx2token)
print('vocabulary size:', vocab_size)
token2idx = dict(zip(idx2token, range(vocab_size)))

start_symbol_id = token2idx['<s>']
end_symbol_id = token2idx['</s>']


tunes = [[token2idx[c] for c in [start_symbol] + t.split() + [end_symbol]] for t in tunes]

# set maximum length to 500
tunes = [t for t in tunes if len(t) <= 100]

tunes.sort(key=lambda x: len(x), reverse=True)
ntunes = len(tunes)

tune_lens = np.array([len(t) for t in tunes])
max_len = max(tune_lens)


nvalid_tunes = ntunes * validation_fraction
nvalid_tunes = int(batch_size * max(1, np.rint(
    nvalid_tunes / float(batch_size))))  # round to the multiple of batch_size

rng = np.random.RandomState(42)
valid_idxs = rng.choice(np.arange(ntunes), nvalid_tunes, replace=False)

ntrain_tunes = ntunes - nvalid_tunes
train_idxs = np.delete(np.arange(ntunes), valid_idxs)

print('n tunes:', ntunes)
print('n train tunes:', ntrain_tunes)
print('n validation tunes:', nvalid_tunes)
print('min, max length', min(tune_lens), max(tune_lens))

def create_batch(idxs):
    max_seq_len = max([len(tunes[i]) for i in idxs])
    x = np.zeros((batch_size, max_seq_len), dtype='float32')
    mask = np.zeros((batch_size, max_seq_len - 1), dtype='float32')
    for i, j in enumerate(idxs):
        x[i, :tune_lens[j]] = tunes[j]
        mask[i, : tune_lens[j] - 1] = 1
    return x, mask


temp_tunes = np.zeros((ntunes, max(tune_lens)), dtype='float32')
length = max(map(len, tunes))
temp_tunes=np.array([xi+[None]*(length-len(xi)) for xi in tunes], dtype='float32')

model = create_network(temp_tunes, vocab_size)

train_data_iterator = DataIterator(tune_lens[train_idxs], train_idxs, batch_size, random_lens=False)
valid_data_iterator = DataIterator(tune_lens[valid_idxs], valid_idxs, batch_size, random_lens=False)

niter = 1
losses = []
val_losses = []
train_batches_per_epoch = ntrain_tunes / batch_size
max_niter = max_epoch * train_batches_per_epoch
nvalid_batches = nvalid_tunes / batch_size
prev_time = time.clock()

for epoch in range(max_epoch):
    for i, train_batch_idxs in enumerate(train_data_iterator):
        x_batch, mask_batch = create_batch(train_batch_idxs)

        history = model.fit(x=x_batch.T,y=x_batch[i+1],batch_size=batch_size,verbose=0)
        
        current_time = time.clock()
        
        train_loss = history.history['loss'][0]
        
        print('%d/%d (epoch %.3f) train_loss=%6.8f time/batch=%.2fs' % (niter, max_niter, niter / float(train_batches_per_epoch), train_loss , current_time - prev_time))
        
        prev_time = current_time
        losses.append(train_loss)
        niter += 1

        if niter % validate_every == 0:
            print("Validating")
            for valid_batch_idx in valid_data_iterator:
                x_batch, mask_batch = create_batch(valid_batch_idx)

                val_loss = model.evaluate(x=x_batch.T,y=x_batch[i+1],batch_size=batch_size,verbose=0)

                print("Val Loss:", val_loss)
                val_losses.append(val_loss)
            

    if (epoch + 1) % save_every == 0:
        with open('metadata_pickle_'+ str(epoch) +'.pkl' , 'wb') as f:
            pickle.dump({
                'iters_since_start': niter,
                'losses_train': losses,
                'losses_eval_valid': val_losses,
                'token2idx': token2idx
            }, f)
        print("Saved to metadata.pkl file")

print('Saving the weights')
model.save_weights('weights.hdf5')






