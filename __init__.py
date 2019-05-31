q4 = """Cambodian leader Hun Sen on Friday rejected opposition parties demands for talks outside the country, accusing them of trying to internationalize the political crisis.
Government and opposition parties have asked King Norodom Sihanouk to host a summit meeting after a series of post-election negotiations between the two opposition groups and Hun Sen's party to form a new government failed.
Opposition leaders Prince Norodom Ranariddh and Sam Rainsy, citing Hun Sen's threats to arrest opposition figures after two alleged attempts on his life, said they could not negotiate freely in Cambodia and called for talks at Sihanouk's residence in Beijing.
Hun Sen, however, rejected that.
"""
q3 = """King Norodom Sihanouk has declined requests to chair a summit of Cambodia's top political leaders, saying the meeting would not bring any progress in deadlocked negotiations to form a government.
Cambodian leader Hun Sen's ruling party and the two-party opposition had called on the monarch to lead top-level talks, but disagreed on its location.
Papa will not preside over any summit meeting between the three parties, whether it is held in Phnom Penh or Beijing, because such a meeting will certainly achieve no result, Sihanouk wrote in an Oct. 17 letter to his son, Prince Norodom Ranariddh, leader of the senior opposition FUNCINPEC party.
"""

q1=  "Cambodian leader Hun Sen on Friday rejected opposition parties demands for talks outside the country, accusing them of trying to internationalize the political crisis."
q2 = "King Norodom Sihanouk has declined requests to chair a summit of Cambodia's top political leaders, saying the meeting would not bring any progress in deadlocked negotiations to form a government."
q5= "A gem is a jewel or stone that is used in jewellery."
q6 = "A jewel is a precious stone used to decorate valuable things that you wear, such as rings or necklaces."
q7 = """Black holes are some of the strangest and most fascinating objects found in outer space. Black holes are invisible. They are objects of extreme density, with such strong gravitational attraction that even light cannot escape from their grasp if it comes near enough."""
q8 = """A black hole is a region in space where the pulling force of gravity is so strong that light is not able to escape. The strong gravity occurs because matter has been pressed into a tiny space. Because no light can escape,black holes are invisible."""


import numpy as np
from nltk.tokenize import sent_tokenize
import pandas as pd
from Classes.Preprocessing import preprocessing,processData
from Classes.SentenceSimilarity import sentence_similarity

# q1_sents_tok = sent_tokenize(q7)
# q2_sents_tok = sent_tokenize(q8)

q1_sents_tok =[] 
q2_sents_tok =[]
with open('data/DUC2004_documents_cleaned_tokenized/d31050t_raw/NYT19981222.0021.txt','r') as f:
    doc =  f.read()
    q1_sents_tok = sent_tokenize(doc)

with open('data/DUC2004_documents_cleaned_tokenized/d31050t_raw/NYT19981221.0377.txt','r') as f:
    doc = f.read()
    q2_sents_tok = sent_tokenize(doc)


sim_matrix =np.zeros((len(q1_sents_tok),len(q2_sents_tok)))

q1_filtered_tok,q1_disamb_dic = processData(q1_sents_tok)
q2_filtered_tok,q2_disamb_dic = processData(q2_sents_tok)

# q1s = []
# for s in q1_filtered_tok:
#     q1s.append(" ".join(s))

# for s in q2_filtered_tok:
#     q1s.append(" ".join(s))

# from sklearn.feature_extraction.text import CountVectorizer

# vec = CountVectorizer()
# matrix = vec.fit_transform(q1s)
# df = pd.DataFrame(matrix.toarray(), columns=vec.get_feature_names())

# print(df['holes'])
from collections import Counter
wordCounter = Counter()
for s in q1_filtered_tok:
    wordCounter.update(s)

for s in q2_filtered_tok:
    wordCounter.update(s)


# print(df.head())

print('Preprocessing done!')
for i in range(len(q1_filtered_tok)):
    for j in range(len(q2_filtered_tok)):

        # q1s = " ".join(q1_filtered_toks[i])
        # q2s = " ".join(q2_filtered_toks[j])
        # print(q1s,"\n",q2s)
        sim = sentence_similarity(q1_disamb_dic[i], q1_filtered_tok[i], q2_disamb_dic[j], q2_filtered_tok[j])
        # print(sim,"\n")
        sim_matrix[i][j] = sim


# for x in range(len(q1_sents_tok)):
#     for y in range(len(q2_sents_tok)):
#         sense1, q1_filtered_toks, sense2, q2_filtered_toks = preprocessing(q1_sents_tok[x], q2_sents_tok[y])
#         q1s = " ".join(q1_filtered_toks)
#         q2s = " ".join(q2_filtered_toks)
#         # print(q1s,"\n",q2s)
#         sim = sentence_similarity(sense1, q1_filtered_toks, sense2, q2_filtered_toks)
#         # print(sim,"\n")
#         sim_matrix[x][y] = sim



for x in range(len(q1_sents_tok)):
    for y in range(len(q2_sents_tok)):
        print(sim_matrix[x][y],end="\t")
    print()

def getMaxWithIndex(sim):
    d = {}
    for i,similarities in enumerate(sim):
        m = -1
        index = -1
        for j,val in enumerate(similarities):
            if m < val :
                m = val
                index = j
        d[(i,j)]=m
    return d



dics = getMaxWithIndex(sim_matrix)

print("\n \n")

import operator
sorted_x = sorted(dics.items(), key=operator.itemgetter(1),reverse=True)
print(sorted_x)

# for k,v in sorted_x:
#     print(q1_sents_tok[k[0]],end=" ")

# print("\n \n")

# sorted_x = [tup for tup,v in sorted_x]
leng = int(len(sorted_x)/4)

for k,v in sorted_x[:leng]:
    if v <0.5:
        print(k,v)
        break
    q1score = 0
    for word in q1_filtered_tok[k[0]]:
        q1score += wordCounter[word]
    q2score = 0
    for word in q2_filtered_tok[k[1]]:
        q2score += wordCounter[word]
    if q1score> q2score:
        print(q1_sents_tok[k[0]],end=" ")
    else:
        print(q2_sents_tok[k[1]],end=" ")

print()