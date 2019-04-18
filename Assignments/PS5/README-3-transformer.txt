Shareable Link
---------------------------------------------------------------------------------------------------
https://colab.research.google.com/drive/1apow70SkTBnMaFRsZv77z-ibi2ETNCeJ



Answers to Short Questions
---------------------------------------------------------------------------------------------------
1.
Q: Please explain what the role of the denominator in the self-attention equation is.
A: dk represents the dimensionality of the queries and keys. The dot-product attention is scaled by 1/sqrt(dk) because even though the dot-product attention is more efficient in time and space than the additive attention in practice (because of highly optimized matrix multiplication), it performs poorer than the additive attention if not scaled by larger values of dk. This is because for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. The role of the denominator is to counteract the above effect.

2.
Q: Please explain what the motivation is behind using multiple z matrices. 
A: This is referred to as multi-head attention. The motivation is that it allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

3.
Q: Please explain what the benefits of using residual connections are here (and in neural networks in general).
A: Residual connections allow gradients to flow through the network directly, without passing through non-linear activation functions. This prevents the gradients from exploding or vanishing.



Outputs
---------------------------------------------------------------------------------------------------
| Epoch: 001 | Time: 3m 21s| Train Loss: 5.948 | Train PPL: 383.051 | Val. Loss: 4.108 | Val. PPL:  60.820 |
| Epoch: 002 | Time: 3m 28s| Train Loss: 3.771 | Train PPL:  43.420 | Val. Loss: 3.200 | Val. PPL:  24.543 |
| Epoch: 003 | Time: 3m 28s| Train Loss: 3.132 | Train PPL:  22.909 | Val. Loss: 2.809 | Val. PPL:  16.598 |
| Epoch: 004 | Time: 3m 27s| Train Loss: 2.762 | Train PPL:  15.834 | Val. Loss: 2.575 | Val. PPL:  13.128 |
| Epoch: 005 | Time: 3m 27s| Train Loss: 2.503 | Train PPL:  12.218 | Val. Loss: 2.406 | Val. PPL:  11.095 |


| Test Loss: 2.392 | Test PPL:  10.939 |


