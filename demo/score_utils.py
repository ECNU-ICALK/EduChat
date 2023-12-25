import torch.nn as nn
cos = nn.CosineSimilarity(dim=1, eps=1e-6)

def score(key_words, sentence,ft):
    res = 0

    for key_word in key_words.split():

        key_embedding = ft.get_word_vector(key_word)


        vector = ft.get_word_vector(sentence)  # 300-dim vector
         
        from numpy import dot
        from numpy.linalg import norm

        cos_sim = dot(key_embedding, vector)/(norm(key_embedding)*norm(vector))
        res += cos_sim
    return res 

def score_2(key_words, sentence,):
    res = 0

    for key_word in key_words.split():

        res += 1 if sentence.find(key_word) > 0 else 0

    return res 

def score_3(key_words, sentence,  measure):
    if sentence.split():
        return measure.infer(sentence, key_words)
    else:
        return -10000 


