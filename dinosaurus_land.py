import numpy as np
from utils import *
import random
from random import shuffle

data = open(
    "D:/Andrew_Ng_Deep_learning/dinosaurus island/dinosaurus island/dinosaurus island/dinos.txt"
).read()
data = data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)

char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}
ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}


def clip(gradients, maxValue):
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients[
        'dWax'], gradients['dWya'], gradients['db'], gradients['dby']
    for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, -1 * maxValue, maxValue, out=gradient)

    gradients = {
        "dWaa": dWaa,
        "dWax": dWax,
        "dWya": dWya,
        "db": db,
        "dby": dby
    }

    return gradients


def sample(parameters, char_to_ix, seed):
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters[
        'Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    x = np.zeros((vocab_size, 1))
    a_prev = np.zeros((n_a, 1))
    indices = []
    idx = -1
    counter = 0

    newline_character = char_to_ix['\n']
    while (idx != newline_character and counter != 50):
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        z = np.dot(Wya, a) + by
        y = softmax(z)
        np.random.seed(counter + seed)
        idx = np.random.choice(list(range(vocab_size)), p=y.ravel())
        indices.append(idx)
        x = np.zeros((vocab_size, 1))
        x[idx] = 1

        a_prev = a
        seed += 1
        counter += 1

    if (counter == 50):
        indices.append(char_to_ix['\n'])

    return indices


def optimize(X, Y, a_prev, parameters, learning_rate=0.1):
    loss, cache = rnn_forward(X, Y, a_prev, parameters, vocab_size)

    gradients, a = rnn_backward(X, Y, parameters, cache)

    gradients = clip(gradients, 5)

    parameters = update_parameters(parameters, gradients, learning_rate)

    return loss, gradients, a[len(X) - 1]


def model(data,
          ix_to_char,
          char_to_ix,
          num_iterations=35000,
          n_a=50,
          dino_names=7,
          vocab_size=27):
    n_x, n_y = vocab_size, vocab_size
    parameters = initialize_parameters(n_a, n_x, n_y)
    loss = get_initial_loss(vocab_size, dino_names)
    with open(
            "D:/Andrew_Ng_Deep_learning/dinosaurus island/dinosaurus island/dinosaurus island/dinos.txt"
    ) as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]

    shuffle(examples)

    a_prev = np.zeros((n_a, 1))

    for j in range(num_iterations):
        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_ix['\n']]

        curr_loss, gradients, a_prev = optimize(X,
                                                Y,
                                                a_prev,
                                                parameters,
                                                learning_rate=0.01)
        loss = smooth(loss, curr_loss)
        if j % 2000 == 0:

            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')

            # The number of dinosaur names to print
            seed = 0
            for name in range(dino_names):

                # Sample indexes and print them
                sampled_indexes = sample(parameters, char_to_ix, seed)
                print_sample(sampled_indexes, ix_to_char)

                seed += 1  # To get the same result for grading purposed, increment the seed by one.

            print('\n')

    return parameters


parameters = model(data, ix_to_char, char_to_ix)