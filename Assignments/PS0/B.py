import gensim

from collections import OrderedDict
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic

RESOURCES = ""
RESOURCES = '/home/classes/cs477/data/'

# Given a filename where each line is in the format "<word1>  <word2>  <human_score>", 
# return a dictionary of {(word1, word2): human_score), ...}
# Note that human_scores in your dictionary should be floats.
def parseFile(filename):
    similarities = OrderedDict()
    # Fill in your code here
    with open(filename) as f:
        for line in f:
            token_list = word_tokenize(line)
            similarities[(token_list[0], token_list[1])] = float(token_list[2])
    ########################
    return similarities
    

# Given a list of tuples [(word1, word2), ...] and a wordnet_ic corpus, return a dictionary 
# of Lin similarity scores {(word1, word2): similarity_score, ...}
def linSimilarities(word_pairs, ic):
    similarities = {}
    # Fill in your code here
    for pair in word_pairs:
        synset1 = wn.synsets(pair[0], pos = wn.NOUN)
        synset2 = wn.synsets(pair[1], pos = wn.NOUN)
        if synset1 and synset2: # non-empty NOUN synsets
            similarities[(pair[0], pair[1])] = 10 * synset1[0].lin_similarity(synset2[0], ic)
            continue
        # at leat one word of the pair has no noun synset
        synset1 = wn.synsets(pair[0], pos = wn.VERB)
        synset2 = wn.synsets(pair[1], pos = wn.VERB)
        if synset1 and synset2: # non-empty VERB synsets
            similarities[(pair[0], pair[1])] = 10 * synset1[0].lin_similarity(synset2[0], ic)
    #######################
    return similarities


# Given a list of tuples [(word1, word2), ...] and a wordnet_ic corpus, return a dictionary 
# of Resnik  similarity scores {(word1, word2): similarity_score, ...}
def resSimilarities(word_pairs, ic):
    similarities = {}
    # Fill in your code here
    for pair in word_pairs:
        synset1 = wn.synsets(pair[0], pos = wn.NOUN)
        synset2 = wn.synsets(pair[1], pos = wn.NOUN)
        if synset1 and synset2: # non-empty NOUN synsets
            similarities[(pair[0], pair[1])] = synset1[0].res_similarity(synset2[0], ic)
            continue
        # at leat one word of the pair has no noun synset
        synset1 = wn.synsets(pair[0], pos = wn.VERB)
        synset2 = wn.synsets(pair[1], pos = wn.VERB)
        if synset1 and synset2: # non-empty VERB synsets
            similarities[(pair[0], pair[1])] = synset1[0].res_similarity(synset2[0], ic)
    ########################
    return similarities

# def findFirstNoun(word):
#     return None

# def findFirstVerb(word):



# Given a list of tuples [(word1, word2), ...] and a word2vec model, return a dictionary 
# of word2vec similarity scores {(word1, word2): similarity_score, ...}
def vecSimilarities(word_pairs, model):
    similarities = {}
    # Fill in your code here
    for pair in word_pairs:
        similarities[(pair[0], pair[1])] = 10 * model.similarity(pair[0].lower(), pair[1].lower())
    ########################
    return similarities


def main():
    brown_ic = wordnet_ic.ic('ic-brown.dat')

    human_sims = parseFile("input.txt")
    lin_sims = linSimilarities(human_sims.keys(), brown_ic)
    res_sims = resSimilarities(human_sims.keys(), brown_ic)

    model = None
    #model = model.load_word2vec_format(RESOURCES+'glove_model.txt', binary=False)
    model = gensim.models.KeyedVectors.load_word2vec_format(RESOURCES+'glove_model.txt', binary=False)
    vec_sims = vecSimilarities(human_sims.keys(), model)
    
    lin_score = 0
    res_score = 0
    vec_score = 0

    print(f"{'word1':15} {'word2':15} {'human':10} {'Lin':20} {'Resnik':20} {'Word2Vec':20}")

    for key, human in human_sims.items():
        try:
            lin = lin_sims[key]
        except:
            lin = 0
        lin_score += (lin - human) ** 2
        try:
            res = res_sims[key]
        except:
            res = 0
        res_score += (res - human) ** 2
        try:
            vec = vec_sims[key]
        except:
            vec = 0
        vec_score += (vec - human) ** 2
        print(f"{key[0]:15} {key[1]:15} {human:10} {lin:20} {res:20} {vec:20}")

    num_examples = len(human_sims)
    print("\nMean Squared Errors")
    print(f"Lin method error: {lin_score/num_examples:.2f}")
    print(f"Resnick method error: {res_score/num_examples:.2f}")
    print(f"Vector-based method error: {vec_score/num_examples:.2f}")

if __name__ == "__main__":
    main()
