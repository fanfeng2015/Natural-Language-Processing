Name: Fan Feng
Email: fan.feng@yale.edu
NetID: ff242

Part A
=====================================================================
[ff242@rattlesnake Homework2]$ python3 solutionsA.py 
Accuracy: 0.839856

The main assumption of a Naive Bayes classifier is conditional independence, i.e., the feature probability is indenpendent given the class c. Thus,

P(x1, x2, ..., xn | c) = P(x1 | c) * P(x2 | c) * ... * P(xn | c), 

P(xi | c) can be estimated with the maximum-likelihood estimate, by
(number of times feature xi appears in samples of class c) / (total number of features in samples of class c).

In our case of sentiment analysis, c is either positive or negative, and xi is any word.



Part B
=====================================================================
(hw2) [ff242@raven Homework2]$ python3 solutionsB.py 
| Epoch: 01 | Train Loss: 0.700 | Train Acc: 49.90% | Val. Loss: 0.703 | Val. Acc: 49.75% |
| Epoch: 02 | Train Loss: 0.698 | Train Acc: 49.90% | Val. Loss: 0.781 | Val. Acc: 48.63% |
| Epoch: 03 | Train Loss: 0.697 | Train Acc: 50.52% | Val. Loss: 0.721 | Val. Acc: 48.78% |
| Epoch: 04 | Train Loss: 0.699 | Train Acc: 49.22% | Val. Loss: 0.708 | Val. Acc: 49.38% |
| Epoch: 05 | Train Loss: 0.696 | Train Acc: 50.36% | Val. Loss: 0.713 | Val. Acc: 49.07% |
| Test Loss: 0.719 | Test Acc: 53.31% |



Part C
=====================================================================
(hw2) [ff242@raven Homework2]$ python3 solutionsC.py
| Epoch: 01 | Train Loss: 0.696 | Train Acc: 50.31% | Val. Loss: 0.695 | Val. Acc: 51.02% |
| Epoch: 02 | Train Loss: 0.699 | Train Acc: 50.07% | Val. Loss: 0.694 | Val. Acc: 49.00% |
| Epoch: 03 | Train Loss: 0.696 | Train Acc: 49.99% | Val. Loss: 0.694 | Val. Acc: 49.54% |
| Epoch: 04 | Train Loss: 0.695 | Train Acc: 49.53% | Val. Loss: 0.694 | Val. Acc: 49.58% |
| Epoch: 05 | Train Loss: 0.696 | Train Acc: 50.07% | Val. Loss: 0.704 | Val. Acc: 48.95% |
| Test Loss: 0.701 | Test Acc: 49.46% |

Pretrained word embeddings can be useful in NLP because TODO: //



Part D
=====================================================================
[ff242@hippo Homework2]$ cat ff242-d.txt 
| Epoch: 01 | Train Loss: 0.671 | Train Acc: 60.60% | Val. Loss: 0.672 | Val. Acc: 63.30% |
| Epoch: 02 | Train Loss: 0.600 | Train Acc: 69.34% | Val. Loss: 0.694 | Val. Acc: 50.50% |
| Epoch: 03 | Train Loss: 0.553 | Train Acc: 72.23% | Val. Loss: 0.432 | Val. Acc: 82.99% |
| Epoch: 04 | Train Loss: 0.343 | Train Acc: 85.83% | Val. Loss: 0.406 | Val. Acc: 82.67% |
| Epoch: 05 | Train Loss: 0.214 | Train Acc: 92.09% | Val. Loss: 0.352 | Val. Acc: 87.05% |
| Test Loss: 0.424 | Test Acc: 83.88% |

[ff242@monkey Homework2]$ cat ff242-d.txt 
| Epoch: 01 | Train Loss: 0.668 | Train Acc: 60.21% | Val. Loss: 0.693 | Val. Acc: 50.38% |
| Epoch: 02 | Train Loss: 0.675 | Train Acc: 57.54% | Val. Loss: 0.692 | Val. Acc: 54.49% |
| Epoch: 03 | Train Loss: 0.594 | Train Acc: 67.28% | Val. Loss: 0.372 | Val. Acc: 84.84% |
| Epoch: 04 | Train Loss: 0.281 | Train Acc: 89.18% | Val. Loss: 0.281 | Val. Acc: 88.82% |
| Epoch: 05 | Train Loss: 0.159 | Train Acc: 94.67% | Val. Loss: 0.318 | Val. Acc: 88.89% |
| Test Loss: 0.408 | Test Acc: 85.63% |

[ff242@macaw Homework2]$ cat ff242-d.txt 
| Epoch: 01 | Train Loss: 0.672 | Train Acc: 59.13% | Val. Loss: 0.677 | Val. Acc: 58.66% |
| Epoch: 02 | Train Loss: 0.612 | Train Acc: 67.37% | Val. Loss: 0.610 | Val. Acc: 67.26% |
| Epoch: 03 | Train Loss: 0.383 | Train Acc: 84.01% | Val. Loss: 0.337 | Val. Acc: 86.91% |
| Epoch: 04 | Train Loss: 0.227 | Train Acc: 91.87% | Val. Loss: 0.309 | Val. Acc: 88.50% |
| Epoch: 05 | Train Loss: 0.134 | Train Acc: 95.63% | Val. Loss: 0.343 | Val. Acc: 88.44% |
| Test Loss: 0.435 | Test Acc: 84.96% |

Bidirectional LSTM is advantageous to regular RNN because TODO: // 




