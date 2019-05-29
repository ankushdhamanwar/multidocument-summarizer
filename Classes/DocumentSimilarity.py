q4 = """Cambodian leader Hun Sen on Friday rejected opposition parties demands for talks outside the country, accusing them of trying to internationalize the political crisis.
Government and opposition parties have asked King Norodom Sihanouk to host a summit meeting after a series of post-election negotiations between the two opposition groups and Hun Sen's party to form a new government failed.
Opposition leaders Prince Norodom Ranariddh and Sam Rainsy, citing Hun Sen's threats to arrest opposition figures after two alleged attempts on his life, said they could not negotiate freely in Cambodia and called for talks at Sihanouk's residence in Beijing.
Hun Sen, however, rejected that.
I would like to make it clear that all meetings related to Cambodian affairs must be conducted in the Kingdom of Cambodia, Hun Sen told reporters after a Cabinet meeting on Friday.
No-one should internationalize Cambodian affairs.
"""
q3 = """King Norodom Sihanouk has declined requests to chair a summit of Cambodia's top political leaders, saying the meeting would not bring any progress in deadlocked negotiations to form a government.
Cambodian leader Hun Sen's ruling party and the two-party opposition had called on the monarch to lead top-level talks, but disagreed on its location.
Papa will not preside over any summit meeting between the three parties, whether it is held in Phnom Penh or Beijing, because such a meeting will certainly achieve no result, Sihanouk wrote in an Oct. 17 letter to his son, Prince Norodom Ranariddh, leader of the senior opposition FUNCINPEC party.
A copy of the letter was obtained Thursday.
In it, the king called on the three parties to make compromises to end the stalemate : Papa would like to ask all three parties to take responsibility before the nation and the people.
Hun Sen used Thursday's anniversary of a peace agreement ending the country's civil war to pressure the opposition to form a coalition government with his party.
"""

q1=  "Midday is 12 oclock in the middle of the day."
q2 = "Noon is 12 oclock in the middle of the day."

import numpy as np
from nltk.tokenize import sent_tokenize

from Classes.Preprocessing import preprocessing
from Classes.SentenceSimilarity import sentence_similarity

q1_sents_tok = sent_tokenize(q3)
q2_sents_tok = sent_tokenize(q4)
sim_matrix =np.zeros((len(q1_sents_tok),len(q2_sents_tok)))


for x in range(len(q1_sents_tok)):
    for y in range(len(q2_sents_tok)):
        q1_filtered_toks, q2_filtered_toks = preprocessing(q1_sents_tok[x], q2_sents_tok[y])
        sim = sentence_similarity(q1_filtered_toks, q2_filtered_toks)
        sim_matrix[x][y] = sim



for x in range(len(q1_sents_tok)):
    for y in range(len(q2_sents_tok)):
        print(sim_matrix[x][y],"\t")
    print("\n")