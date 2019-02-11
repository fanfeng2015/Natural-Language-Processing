import sys
import nltk
import math
import time
import solutionsB as B

from collections import defaultdict
from collections import deque
from collections import OrderedDict

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000

DATA_PATH = '/home/classes/cs477/data/'
OUTPUT_PATH = 'output/'

#This is the same as the code in Part B, but with a Spanish corpus being evaluated 

def main():
    time.clock()

    infile = open(DATA_PATH + "wikicorpus_tagged_train.txt", "r")
    train = infile.readlines()
    infile.close()

    words, tags = B.split_wordtags(train)

    if len(sys.argv) > 1 and sys.argv[1] == "-reverse":
        q_values = B.calc_trigrams_reverse(tags)
    else:
        q_values = B.calc_trigrams(tags)

    B.q2_output(q_values, OUTPUT_PATH + 'C2.txt')

    known_words = B.calc_known(words)

    words_rare = B.replace_rare(words, known_words)

    B.q3_output(words_rare, OUTPUT_PATH + "C3.txt")

    e_values, taglist = B.calc_emission(words_rare, tags)

    B.q4_output(e_values, OUTPUT_PATH + "C4.txt")

    del train
    del words_rare

    infile = open(DATA_PATH + "wikicorpus_dev.txt", "r")
    dev = infile.readlines()
    infile.close()

    dev_words = []
    for sentence in dev:
        dev_words.append(sentence.split(" ")[:-1])

    viterbi_tagged = B.viterbi(dev_words, taglist, known_words, q_values, e_values)

    B.q5_output(viterbi_tagged, OUTPUT_PATH + 'C5.txt')

    nltk_tagged = B.nltk_tagger(words, tags, dev_words)

    B.q6_output(nltk_tagged, OUTPUT_PATH + 'C6.txt')

    print("Part C time: " + str(time.clock()) + ' sec')

if __name__ == "__main__": main()
