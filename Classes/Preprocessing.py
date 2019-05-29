from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
eng_stops = set(stopwords.words('english'))

def preprocessing(sent1,sent2):
    q1_toks = word_tokenize(sent1)
    q2_toks = word_tokenize(sent2)
    q1_toks = [word for  word in q1_toks if word not in eng_stops]
    q2_toks = [word for  word in q2_toks if word not in eng_stops]
    q1_tagged_toks = pos_tag(q1_toks)
    #print(q1_tagged_toks)
    q2_tagged_toks = pos_tag(q2_toks)
    #print(q2_tagged_toks)
    q1_filtered_toks = [words[0] for words in q1_tagged_toks if words[1].startswith('N') or words[1].startswith('V')]
    q2_filtered_toks = [words[0] for words in q2_tagged_toks if words[1].startswith('N') or words[1].startswith('V')]

    # q1_filtered_toks = [words[0] for words in q1_tagged_toks if 'NN' in words[1]  or 'VB' in words[1]]
    # q2_filtered_toks = [words[0] for words in q2_tagged_toks if 'NN' in words[1]  or 'VB' in words[1]]
    return q1_filtered_toks, q2_filtered_toks