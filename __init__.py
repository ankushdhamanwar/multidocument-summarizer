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
q8 = """A black hole is a region in space where the pulling force of gravity is so strong that light is not able to escape. The strong gravity occurs because matter has been pressed into a tiny space. Some black holes are a result of dying stars. Because no light can escape,black holes are invisible."""


import numpy as np
from nltk.tokenize import sent_tokenize

from Classes.Preprocessing import preprocessing
from Classes.SentenceSimilarity import sentence_similarity

q1_sents_tok = sent_tokenize(q7)
q2_sents_tok = sent_tokenize(q8)
sim_matrix =np.zeros((len(q1_sents_tok),len(q2_sents_tok)))
print(q1_sents_tok,"\n",q2_sents_tok)

for x in range(len(q1_sents_tok)):
    for y in range(len(q2_sents_tok)):
        q1_filtered_toks, q2_filtered_toks = preprocessing(q1_sents_tok[x], q2_sents_tok[y])
        print(q1_filtered_toks,"\n",q2_filtered_toks)
        sim = sentence_similarity(q1_filtered_toks, q2_filtered_toks)
        sim_matrix[x][y] = sim



for x in range(len(q1_sents_tok)):
    for y in range(len(q2_sents_tok)):
        print(sim_matrix[x][y],"    ")
    print("\n")