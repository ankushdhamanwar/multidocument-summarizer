from nltk.corpus import wordnet as wn
import math

def proper_synset(word_one , word_two):
    pair = (None,None)
    maximum_similarity = -1
    synsets_one = wn.synsets(word_one)
    synsets_two = wn.synsets(word_two)
    if(len(synsets_one)!=0 and len(synsets_two)!=0):
        for synset_one in synsets_one:
            for synset_two in synsets_two:
                similarity = wn.path_similarity(synset_one,synset_two)
                if(similarity == None):
                    sim = -2
                elif(similarity > maximum_similarity):
                    maximum_similarity = similarity
                    pair = synset_one,synset_two
    else:
        pair = (None , None)
    return pair


CONST_ALPHA = 0.2
CONST_BETA = 0.45
def length_between_words(synset_one , synset_two):
        length = 100000000
        if synset_one is None or synset_two is None:
            return 0
        elif(synset_one == synset_two):
            length = 0
        else:
            words_synet1 = set([word.name() for word in synset_one.lemmas()])
            words_synet2 = set([word.name() for word in synset_two.lemmas()])
            if(len(words_synet1) + len(words_synet2) > len(words_synet1.union(words_synet2))):
                length = 0
            else:
                #finding the actual distance
                length = synset_one.shortest_path_distance(synset_two)
                #print(length)
                if(length is None):
                    return 0
        return math.exp( -1 * CONST_ALPHA * length)


def depth_common_subsumer(synset_one,synset_two):
    height = 100000000
    if synset_one is None or synset_two is None:
        return 0
    elif synset_one == synset_two:
        height = max([hypernym[1] for hypernym in synset_one.hypernym_distances()])
    else:
        #get the hypernym set of both the synset.
        hypernym_one = {hypernym_word[0]:hypernym_word[1] for hypernym_word in synset_one.hypernym_distances()}
        hypernym_two = {hypernym_word[0]:hypernym_word[1] for hypernym_word in synset_two.hypernym_distances()}
        common_subsumer = set(hypernym_one.keys()).intersection(set(hypernym_two.keys()))
        if(len(common_subsumer) == 0):
            height = 0
        else:
            height = 0
            for cs in common_subsumer:
                val = [hypernym_word[1] for hypernym_word in cs.hypernym_distances()]
                val = max(val)
                if val > height : height = val

    #print(height) #works
    return (math.exp(CONST_BETA * height) - math.exp(-CONST_BETA * height))/(math.exp(CONST_BETA * height) + math.exp(-CONST_BETA * height))


def word_similarity(word1, word2):
    # if word1.lower() == word2.lower():
    #     return 1
    synset_word1, synset_word2 = proper_synset(word1, word2)
    return length_between_words(synset_word1, synset_word2) * depth_common_subsumer(synset_word1, synset_word2)