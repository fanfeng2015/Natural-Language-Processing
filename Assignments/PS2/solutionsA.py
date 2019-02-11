import numpy as np
import sklearn
import matplotlib.pyplot as plt

from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from sklearn.naive_bayes import MultinomialNB

# TODO: IMPLEMENT THIS FUNCTION
# Converts the movie review text into a matrix of token counts for each word
# The parameter data matrix only contains the reviews and not the sentiments or targets
# Returns the matrix of token counts
def q1_output(data):
	data_transformed = []

	return data_transformed

DATA_PATH = r'/home/classes/cs477/data/aclImdb/all_data/'
# TODO: IMPLEMENT THIS FUNCTION
# Given a training and testing data set, fit a Naive Bayes classifier to the training set 
# and return a set of predictions for the test set
def q2_output(x_train, x_test, y_train):
	y_hat = []

	return y_hat

# Do not modify the main function
def main():
	data = load_files(DATA_PATH)
	transformed_data = q1_output(data.data)
	x_train, x_test, y_train, y_test = train_test_split(transformed_data, data.target, test_size = 0.2, random_state = 477)

	y_hat = q2_output(x_train, x_test, y_train)
	print(f"Accuracy: %f" % sklearn.metrics.accuracy_score(y_test, y_hat))

if __name__ == "__main__": main()
