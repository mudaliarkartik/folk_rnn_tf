import os
import sys
import time
import logger
import pickle
import importlib

import numpy as np
import tensorflow as tf
from data_iter import DataIterator
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, Bidirectional, Flatten, Reshape
from keras import utils
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras import optimizers

def train_network(network_input, network_output, vocab_size):
    """ Train a Neural Network to generate music """
    
    model = create_network(network_input, vocab_size)

    train(model, network_input, network_output)


def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    rms_opt = optimizers.rmsprop(lr=config.learning_rate, decay=config.learning_rate_decay, clipvalue=config.grad_clipping)
    model = Sequential()
    model.add(LSTM(
        config.rnn_size,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(config.dropout))
    model.add(LSTM(config.rnn_size, return_sequences=True,))
    model.add(Dropout(config.dropout))
    model.add(LSTM(config.rnn_size))
    model.add(Dropout(config.dropout))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=rms_opt)
    model.summary()

    return model

def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "./metadata/weights-improvement-{epoch:02d}-{val_loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        mode='auto'
    )
    callbacks_list = [checkpoint]

    history = model.fit(network_input, network_output, epochs=config.max_epoch, batch_size=config.batch_size, validation_split=config.validation_fraction ,callbacks=callbacks_list)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('./output/new_loss.png')

# create batch
def create_batch(idxs):
    max_seq_len = max([len(tunes[i]) for i in idxs])
    x = np.zeros((config.batch_size, max_seq_len), dtype='float32')
    mask = np.zeros((config.batch_size, max_seq_len - 1), dtype='float32')
    for i, j in enumerate(idxs):
        x[i, :tune_lens[j]] = tunes[j]
        mask[i, : tune_lens[j] - 1] = 1
    return x, mask


if __name__ == '__main__':

    if len(sys.argv) < 3:
        sys.exit("Usage: train_rnn.py <configuration_name> <train data filename>")

    # data preparation
    config_name = sys.argv[1]
    data_path = sys.argv[2]

    config = importlib.import_module('configurations.%s' % config_name)
    experiment_id = '%s-%s-%s' % (
        config_name.split('.')[-1], os.path.basename(data_path.split('.')[0]),
        time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    print(experiment_id)

    # metadata
    if not os.path.isdir('metadata'):
        os.makedirs('metadata')
    metadata_target_path = 'metadata'

    # logs
    if not os.path.isdir('logs'):
            os.makedirs('logs')
    sys.stdout = logger.Logger('logs/%s.log' % experiment_id)
    sys.stderr = sys.stdout

    # load data
    with open(data_path, 'r') as f:
        data = f.read()

    # construct symbol set
    tokens_set = set(data.split())
    start_symbol, end_symbol = '<s>', '</s>'
    tokens_set.update({start_symbol, end_symbol})

    # construct token to number dictionary
    idx2token = list(tokens_set)
    vocab_size = len(idx2token)
    print('vocabulary size:', vocab_size)
    token2idx = dict(zip(idx2token, range(vocab_size)))
    tunes = data.split('\n\n')
    del data

    # transcribe tunes from symbol to index
    tunes = [[token2idx[c] for c in [start_symbol] + t.split() + [end_symbol]] for t in tunes]
    tunes.sort(key=lambda x: len(x), reverse=True)
    ntunes = len(tunes)

    tune_lens = np.array([len(t) for t in tunes])
    max_len = max(tune_lens)

    # tunes for validation
    nvalid_tunes = ntunes * config.validation_fraction
    nvalid_tunes = int(config.batch_size * max(1, np.rint(
        nvalid_tunes / float(config.batch_size))))  # round to the multiple of batch_size

    rng = np.random.RandomState(42)
    valid_idxs = rng.choice(np.arange(ntunes), nvalid_tunes, replace=False)

    # tunes for training
    ntrain_tunes = ntunes - nvalid_tunes
    train_idxs = np.delete(np.arange(ntunes), valid_idxs)

    print('n tunes:', ntunes)
    print('n train tunes:', ntrain_tunes)
    print('n validation tunes:', nvalid_tunes)
    print('min, max length', min(tune_lens), max(tune_lens))
    
    train_network(,vocab_size)