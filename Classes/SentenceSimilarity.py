import numpy as np

from Classes.WordSimilarity import word_similarity

#from pywsd import disambiguate
#from pywsd.lesk import simple_lesk
#from pywsd.similarity import max_similarity

CONST_GAMMA = 1.8
CONST_PHI = 0.2


def sentence_similarity(q1_toks, q2_toks):
    #q1 = " ".join(q1_toks)
    #q2 = " ".join(q2_toks)
    # sense1 = disambiguate(q1)
    # sense2 = disambiguate(q2)
    max_len = max(len(q1_toks), len(q2_toks))
    sem_vector_v1 = np.zeros(max_len)
    sem_vector_v2 = np.zeros(max_len)
    c1 = 0
    c2 = 0
    i1 = 0
    i2 = 0
    sim = 0
    max_sim = 0
    for word1 in q1_toks:
        max_sim = 0
        for word2 in q2_toks:
            sim = word_similarity(word1, word2)
            max_sim = max(max_sim, sim)
        if max_sim < CONST_PHI:
            max_sim = 0
        sem_vector_v1[i1] = max_sim
        i1 = i1 + 1
        if max_sim > 0.8025:
            c1 = c1 + 1
    #print(sem_vector_v1, c1)
    sim = 0
    max_sim = 0
    for word1 in q2_toks:
        max_sim = 0
        for word2 in q1_toks:
            sim = word_similarity(word1, word2)
            max_sim = max(max_sim, sim)
        if max_sim < CONST_PHI:
            max_sim = 0
        sem_vector_v2[i2] = max_sim
        i2 = i2 + 1
        if max_sim > 0.8025:
            c2 = c2 + 1
    #print(sem_vector_v2, c2)
    c = (c1 + c2) / CONST_GAMMA

    if c == 0:
        c = max_len / 2
    #print(c)
    s = (np.linalg.norm(sem_vector_v1) * np.linalg.norm(sem_vector_v2))
    #print(s)
    sim = s / c
    return sim