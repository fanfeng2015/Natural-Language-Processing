from collections import Counter
from collections import deque
from collections import defaultdict

import math
import nltk
import time

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    unigram_p = {}; bigram_p = {}; trigram_p = {}

    # preprocess to get n-grams
    unigrams = []; bigrams = []; trigrams = []
    for sentence in training_corpus:
	    tokens = sentence.strip().split()
	    unigrams.extend(tokens + [STOP_SYMBOL])
	    bigrams.extend(nltk.bigrams([START_SYMBOL] + tokens + [STOP_SYMBOL]))
	    trigrams.extend(nltk.trigrams([START_SYMBOL, START_SYMBOL] + tokens + [STOP_SYMBOL]))
    
    # n-gram probabilities
    unigram_counter = Counter(unigrams)
    unigram_p = { (u, ): math.log(unigram_counter[u] / len(unigrams), 2) for u in unigram_counter }

    unigram_counter[START_SYMBOL] = len(training_corpus) # assume START occurs once in each sentence
    bigram_counter = Counter(bigrams)
    bigram_p = { b: math.log(bigram_counter[b] / unigram_counter[b[0]], 2) for b in bigram_counter }

    bigram_counter[(START_SYMBOL, START_SYMBOL)] = len(training_corpus) # assume START START occurs once in each sentence
    trigram_counter = Counter(trigrams)
    trigram_p = { t: math.log(trigram_counter[t] / bigram_counter[(t[0], t[1])], 2) for t in trigram_counter }

    return unigram_p, bigram_p, trigram_p


# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    sorted(unigrams_keys)
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    sorted(bigrams_keys)
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    sorted(trigrams_keys)
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, corpus):
    scores = []
    for sentence in corpus:
        ngrams = []
        tokens = sentence.strip().split() + [STOP_SYMBOL]
        if n == 1:
            for token in tokens:
                ngrams.append((token, ))
        elif n == 2:
            ngrams = nltk.bigrams([START_SYMBOL] + tokens)
        elif n == 3:
            ngrams = nltk.trigrams([START_SYMBOL, START_SYMBOL] + tokens)

        score = 0
        for ngram in ngrams:
            if ngram not in ngram_p:
                score = MINUS_INFINITY_SENTENCE_LOG_PROB
                break
            else:
                score += ngram_p[ngram]
        scores.append(score)
    return scores


# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []
    for sentence in corpus:
        tokens = [START_SYMBOL, START_SYMBOL] + sentence.strip().split() + [STOP_SYMBOL]
        score = 0
        for trigram in nltk.trigrams(tokens):
            if trigram[2:3] not in unigrams and trigram[1:3] not in bigrams and trigram not in trigrams:
                score = MINUS_INFINITY_SENTENCE_LOG_PROB
                break
            else:
                uni = unigrams[trigram[2:3]] if trigram[2:3] in unigrams else MINUS_INFINITY_SENTENCE_LOG_PROB
                bi = bigrams[trigram[1:3]] if trigram[1:3] in bigrams else MINUS_INFINITY_SENTENCE_LOG_PROB
                tri = trigrams[trigram] if trigram in trigrams else MINUS_INFINITY_SENTENCE_LOG_PROB
                
                score += math.log((2 ** uni + 2 ** bi + 2 ** tri) / 3, 2)
        scores.append(score)
    return scores


DATA_PATH = '/home/classes/cs477/data/' # absolute path to use the shared data
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print(f"Part A time: {str(time.clock())} sec")

if __name__ == "__main__": main()


