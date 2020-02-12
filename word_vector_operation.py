import numpy as np
from w2v_utils import *

words, word_to_vec_map = read_glove_vecs(
    'D:/Andrew_Ng_Deep_learning/Word_Vector_Representation/Word_Vector_Representation/data/glove.6B.50d.txt'
)


def cosine_similarity(u, v):
    distance = 0.0

    dot = np.dot(u.T, v)

    norm_u = np.sqrt(np.sum(u * u))
    norm_v = np.sqrt(np.sum(v * v))

    cosine_similarity = dot / norm_u / norm_v

    return cosine_similarity


def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[
        word_b], word_to_vec_map[word_c]
    words = word_to_vec_map.keys()

    max_cosine_sim = -100
    best_word = []

    for word in words:
        if word in [word_a, word_b, word_c]:
            continue

        cosine_sim = cosine_similarity(e_b - e_a, word_to_vec_map[word] - e_c)
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = word

    return best_word


triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'),
                 ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
for triad in triads_to_try:
    print('{} -> {} :: {} -> {}'.format(
        *triad, complete_analogy(*triad, word_to_vec_map)))
