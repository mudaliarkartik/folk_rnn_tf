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
from keras.layers import LSTM
from keras import optimizers
from keras.layers import Activation

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('metadata_path')
parser.add_argument('--rng_seed', type=int, default=42)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--ntunes', type=int, default=1)
parser.add_argument('--seed')
parser.add_argument('--terminal', action="store_true")

args = parser.parse_args()

metadata_path = args.metadata_path
rng_seed = args.rng_seed
temperature = args.temperature
ntunes = args.ntunes
seed = args.seed

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

    # Load the weights to each node
    model.load_weights('weights-improvement-04-3.2188-bigger.hdf5')

    return model

def generate_notes(model, network_input, pitchnames, n_vocab):
    start = np.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # generate 500 notes
    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output


with open('data/tokens', 'rb') as f:
    metadata = pickle.load(f)

config = importlib.import_module('configurations.%s' % metadata['configuration'])

# samples dir
if not os.path.isdir('samples'):
        os.makedirs('samples')
target_path = "samples/%s-s%d-%.2f-%s.txt" % (
    metadata['experiment_id'], rng_seed, temperature, time.strftime("%Y%m%d-%H%M%S", time.localtime()))

token2idx = metadata['token2idx']
idx2token = dict((v, k) for k, v in token2idx.items())
vocab_size = len(token2idx)

temp_tunes = metadata['tunes']

print(vocab_size)


sequence_length = 500
network_input = []
output = []
for i in range(0, len(temp_tunes) - sequence_length, 1):
    sequence_in = temp_tunes[i:i + sequence_length]
    sequence_out = temp_tunes[i + sequence_length]
    network_input.append([token2idx[char] for char in sequence_in])
    output.append(token2idx[sequence_out])

n_patterns = len(network_input)
normalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
normalized_input = normalized_input / float(vocab_size)

model = create_network(normalized_input,vocab_size)

prediction_output = generate_notes(model, network_input, token2idx, vocab_size)

print(prediction_output)

'''
start_idx, end_idx = token2idx['<s>'], token2idx['</s>']

rng = np.random.RandomState(rng_seed)
vocab_idxs = np.arange(vocab_size)

# Converting the seed passed as an argument into a list of idx
seed_sequence = [start_idx]
if seed is not None:
    for token in seed.split(' '):
        seed_sequence.append(token2idx[token])

for i in range(ntunes):
    sequence = seed_sequence[:]
    while sequence[-1] != end_idx:
        next_itoken = rng.choice(vocab_idxs, p=model.predict(np.array([sequence], dtype='int32')))
        sequence.append(next_itoken)

    abc_tune = [idx2token[j] for j in sequence[1:-1]]
    if not args.terminal:
        f = open(target_path, 'a+')
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


if not args.terminal:
    print('Saved to '+target_path)
'''