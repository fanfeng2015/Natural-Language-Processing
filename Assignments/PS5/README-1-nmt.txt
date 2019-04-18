Shareable Link
---------------------------------------------------------------------------------------------------
https://colab.research.google.com/drive/1qhEzZxIcC-UI_SFisn_Sex5NCk-Iy6Mq



Answers to Short Questions
---------------------------------------------------------------------------------------------------
1. 
Q: In the original paper, they find it beneficial to reverse the order of the input when feeding it to the model. Write why you think that would help. (Hint: see the original paper for their explanation.)

A: Reversing the order of the input helps becaus it introduces many short term dependencies to the dataset. Without reversing, when source sentence and target sentence are concatenated together, the distance between a word in source sentence and its corresponding word in target sentence is far. By reverting however, the average distance between corresponding words in source sentence and target sentence remains unchanged, but the first few words in source sentence are now very close to the first few words in target sentence, thus greatly reducing the problem's minimal time lag.

2. 
Q: Can you think of any additional data augmentation which could be done to improve translation results? 
A: 1. Thesaurus: Replace words and/or phrases by their synonyms.
   2. Word embeddings + cosine similarity: Find similar words for replacement. 



Outputs
---------------------------------------------------------------------------------------------------
Epoch: 01 | Time: 0m 22s
   Train Loss: 4.744 | Train PPL: 114.904
    Val. Loss: 4.944 |  Val. PPL: 140.333
Epoch: 02 | Time: 0m 22s
   Train Loss: 3.935 | Train PPL:  51.143
    Val. Loss: 4.254 |  Val. PPL:  70.382
Epoch: 03 | Time: 0m 22s
   Train Loss: 3.436 | Train PPL:  31.050
    Val. Loss: 4.059 |  Val. PPL:  57.916
Epoch: 04 | Time: 0m 22s
   Train Loss: 3.154 | Train PPL:  23.423
    Val. Loss: 3.830 |  Val. PPL:  46.042
Epoch: 05 | Time: 0m 22s
   Train Loss: 2.908 | Train PPL:  18.324
    Val. Loss: 3.737 |  Val. PPL:  41.992


| Test Loss: 3.741 | Test PPL:  42.146 |


