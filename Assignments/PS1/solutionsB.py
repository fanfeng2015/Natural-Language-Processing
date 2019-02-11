from collections import defaultdict
from collections import deque
from collections import Counter

import sys
import nltk
import math
import time
import heapq

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []
    for sentence in brown_train:
        words = [START_SYMBOL, START_SYMBOL]
        tags = [START_SYMBOL, START_SYMBOL]

        tokens = sentence.strip().split()
        for token in tokens:
        	index = token.rindex('/')
        	words.append(token[ : index])
        	tags.append(token[index+1 : ])

        words.append(STOP_SYMBOL)
        tags.append(STOP_SYMBOL)

        brown_words.append(words)
        brown_tags.append(tags)
    
    return brown_words, brown_tags


# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}
    
    bigrams = []; trigrams = []
    for tags in brown_tags:
        bigrams.extend(nltk.bigrams(tags))
        trigrams.extend(nltk.trigrams(tags))

    bigram_counter = Counter(bigrams)
    trigram_counter = Counter(trigrams)
    for trigram in trigrams:
    	q_values[trigram] = math.log(trigram_counter[trigram] / bigram_counter[(trigram[0], trigram[1])], 2)

    return q_values


# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    sorted(trigrams)
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    known_words = set([])

    words = []
    for sentence in brown_words:
        words.extend(sentence)

    word_counter = Counter(words)
    for word in word_counter:
        if word_counter[word] > RARE_WORD_MAX_FREQ:
            known_words.add(word)

    return known_words


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []

    for sentence in brown_words:
    	cur = []
    	for token in sentence:
    		if token in known_words:
    			cur.append(token)
    		else:
    			cur.append(RARE_SYMBOL)
    	brown_words_rare.append(cur)

    return brown_words_rare


# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set (should not include start and stop tags)
def calc_emission(brown_words_rare, brown_tags):
    e_values = {}

    wordtag2count = {}
    tag2count = {}
    for words, tags in zip(brown_words_rare, brown_tags):
    	for word, tag in zip(words, tags):
    		wordtag2count[(word, tag)] = wordtag2count.get((word, tag), 0) + 1
    		tag2count[tag] = tag2count.get(tag, 0) + 1
    for (word, tag), count in wordtag2count.items():
    	e_values[(word, tag)] = math.log(count / tag2count[tag], 2)

    return e_values, set(tag2count.keys()) - { START_SYMBOL, STOP_SYMBOL }


# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    sorted(emissions)
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence 
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []
    # { (pos, tag_u, tag_v): prob } -- max probability of any sequence ending in (tag_u, tag_v) at position pos
    pi = { (0, START_SYMBOL, START_SYMBOL): 1.0 }
    # { (pos, tag_u, tag_v): tag } -- argmax probability of any sequence ending in (tag_u, tag_v) at position pos
    bp = {}

    for original_sentence in brown_dev_words:
        m = len(original_sentence)
        modified_sentence = [word if word in known_words else RARE_SYMBOL for word in original_sentence]

        for k in range(1, m + 1): # pos
            cur = k - 1           # current index in sentence
            # sequence order is (w, u, v)
            v_word = modified_sentence[cur]
            v_tags = [tag for tag in taglist if (v_word, tag) in e_values]
            u_word = modified_sentence[cur - 1] if cur > 0 else None
            u_tags = [tag for tag in taglist if (u_word, tag) in e_values] if cur > 0 else [START_SYMBOL]
            w_word = modified_sentence[cur - 2] if cur > 1 else None
            w_tags = [tag for tag in taglist if (w_word, tag) in e_values] if cur > 1 else [START_SYMBOL]

            for u_tag in u_tags:
                for v_tag in v_tags:
                    candidate_tag = None
                    max_prob = float("-inf")
                    for w_tag in w_tags:
                        if (v_word, v_tag) in e_values:
                            temp_prob = (pi.get((k - 1, w_tag, u_tag), LOG_PROB_OF_ZERO) + 
                                         q_values.get((w_tag, u_tag, v_tag), LOG_PROB_OF_ZERO) + 
                                         e_values.get((v_word, v_tag)))
                            if temp_prob > max_prob:
                                candidate_tag = w_tag
                                max_prob = temp_prob
                    pi[(k, u_tag, v_tag)] = max_prob
                    bp[(k, u_tag, v_tag)] = candidate_tag

        # compute the most likely tags for the last two words in sentence
        v_word = modified_sentence[m - 1]
        v_tags = [tag for tag in taglist if (v_word, tag) in e_values]
        u_word = modified_sentence[m - 2]
        u_tags = [tag for tag in taglist if (u_word, tag) in e_values]

        candidate_tags = (None, None)
        max_prob = float('-inf')
        for u_tag in u_tags:
            for v_tag in v_tags:
                temp_prob = (pi.get((m, u_tag, v_tag), LOG_PROB_OF_ZERO) + 
                             q_values.get((u_tag, v_tag, STOP_SYMBOL), LOG_PROB_OF_ZERO))
                if temp_prob > max_prob:
                    candidate_tags = (u_tag, v_tag)
                    max_prob = temp_prob
        
        # backtrack to generate the list of most-likely tags
        most_likely_tags = []
        most_likely_tags.extend([candidate_tags[1], candidate_tags[0]])
        for i, k in enumerate(range(m - 2, 0, -1)):
            most_likely_tags.append(bp[(k + 2, most_likely_tags[i + 1], most_likely_tags[i])])
        most_likely_tags.reverse()

        # generate sentence in the format of word/tag
        sentence = ''
        for k in range(m):
            sentence += original_sentence[k] + '/' + most_likely_tags[k] + ' '
        sentence += '\n'
        tagged.append(sentence)

    return tagged


# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. 
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i], brown_tags[i]) for i in range(len(brown_words)) ]
    training = [ list(x) for x in training ]

    # IMPLEMENT THE REST OF THE FUNCTION HERE
    tagged = []
    default_tagger = nltk.DefaultTagger('NOUN')
    bigram_tagger = nltk.BigramTagger(training, backoff = default_tagger)
    trigram_tagger = nltk.TrigramTagger(training, backoff = bigram_tagger)

    for original_sentence in brown_dev_words:
        sentence = ''
        for word, tag in trigram_tagger.tag(original_sentence):
            sentence += word + '/' + tag + ' '
        sentence += '\n'
        tagged.append(sentence)

    return tagged


# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()


DATA_PATH = '/home/classes/cs477/data/' # absolute path to use the shared data
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 6 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print(f"Part B time: {str(time.clock())} sec")

if __name__ == "__main__": main()
