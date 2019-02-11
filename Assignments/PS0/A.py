import nltk
import sys

greeting = sys.stdin.read()
print(greeting)

squirrel = 0
girl = 0
token_list = nltk.word_tokenize(greeting)
print(f"The tokens in the greeting are")
for token in token_list:
    squirrel = (squirrel + 1) if (token.lower() == "squirrel") else squirrel
    girl = (girl + 1) if (token.lower() == "girl") else girl
    print(token)

print(f"There were {squirrel} instances of the word 'squirrel' and {girl} instances of the word 'girl.'")
