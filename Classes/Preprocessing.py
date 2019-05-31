from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pywsd import disambiguate
from nltk import pos_tag
eng_stops = set(stopwords.words('english'))
print('stopwords loaded')
def preprocessing(sent1,sent2):
    sense1 = disambiguate(sent1)
    dict1 = {}
    for x in sense1:
        dict1[x[0]] = x[1]
    sense2 = disambiguate(sent2)
    dict2 = {}
    for x in sense2:
        dict2[x[0]] = x[1]


    q1_toks = word_tokenize(sent1)
    q2_toks = word_tokenize(sent2)
    tagged1 = pos_tag(q1_toks)
    tagged2 = pos_tag(q2_toks)
    q1_stop_filtered_toks = [word for word in tagged1 if word[0] not in eng_stops]
    q2_stop_filtered_toks = [word for word in tagged2 if word[0] not in eng_stops]
    q1_stop_pos_filtered_toks = [word[0] for word in q1_stop_filtered_toks if 'NN' in word[1] or 'VB' in word[1]]
    q2_stop_pos_filtered_toks = [word[0] for word in q2_stop_filtered_toks if 'NN' in word[1] or 'VB' in word[1]]

    return dict1, q1_stop_pos_filtered_toks, dict2, q2_stop_pos_filtered_toks

def processData(sentences):
    p_sentences = []
    p_d_sentences = []
    for sentence in sentences:
        sent = disambiguate(sentence)
        p_d_sentences.append(dict(sent))

        q1_toks = word_tokenize(sentence)
        tagged = pos_tag(q1_toks)
        q1_stop_filtered_toks = [word for word in tagged if word[0] not in eng_stops]
        q1_stop_pos_filtered_toks = [word[0] for word in q1_stop_filtered_toks if 'NN' in word[1] or 'VB' in word[1]]

        p_sentences.append(q1_stop_pos_filtered_toks)
    return p_sentences,p_d_sentences
