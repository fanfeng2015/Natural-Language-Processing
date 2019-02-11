Name: Fan Feng
NetID: ff242

Part A
=====================================================================
1). 
UNIGRAM natural -13.766408817044509
BIGRAM natural that -4.058893689053568
TRIGRAM natural that he -1.5849625007211563

2). 
(hw1) [ff242@swan Homework1]$ python3 perplexity.py output/A2.uni.txt data/Brown_train.txt
The perplexity is 1052.4865859021186
(hw1) [ff242@swan Homework1]$ python3 perplexity.py output/A2.bi.txt data/Brown_train.txt
The perplexity is 53.89847611982476
(hw1) [ff242@swan Homework1]$ python3 perplexity.py output/A2.tri.txt data/Brown_train.txt
The perplexity is 5.710679308201471

3). 
(hw1) [ff242@gator Homework1]$ python3 perplexity.py output/A3.txt data/Brown_train.txt
The perplexity is 12.551609488576803

4). The best model without linear interpolation performs better (i.e., has lower perplexity) than the model with linear interpolation.

This is expected because linear interpolation assigns equal weight to all three taggers, even though we know as a fact that the trigram tagger performs the best (with the lowest perplexity).

5). 
(hw1) [ff242@gator Homework1]$ python3 perplexity.py output/Sample1_scored.txt data/Sample1.txt 
The perplexity is 11.167028915779872
(hw1) [ff242@gator Homework1]$ python3 perplexity.py output/Sample2_scored.txt data/Sample2.txt 
The perplexity is 1611240282.4444103

Sample1.txt belongs to the Brown corpus, because its perplexity is much less then that of Sample2.txt. This is because the model (trained with the Brown corpus) has never seen some of the sentences in Sample2.txt, therefore is very perplexed about them.



Part B
=====================================================================
2). 
TRIGRAM CONJ ADV ADP -2.9755173148006566
TRIGRAM DET NOUN NUM -8.970052616298892
TRIGRAM NOUN PRT PRON -11.085472459181283

4). 
Night NOUN -13.881902599411106
Place VERB -15.453881489107426
prime ADJ -10.69483271828692
STOP STOP 0.0
_RARE_ VERB -3.177320850889013

5).
[ff242@hawk Homework1]$ python3 pos.py output/B5.txt data/Brown_tagged_dev.txt 
Percent correct tags: 93.32499462544219

6).
[ff242@hawk Homework1]$ python3 pos.py output/B6.txt data/Brown_tagged_dev.txt 
Percent correct tags: 88.03994762249106



Part C
=====================================================================
1). 
[ff242@hawk Homework1]$ python3 pos.py output/C5.txt data/wikicorpus_tagged_dev.txt
Percent correct tags: 84.49340990538452

Q: The Spanish dataset takes longer to evaluate. Why do you think this is the case?
A: Because the tag list is much larger in Spanish than in English (61 vs. 12).

Q: What are aspects or features of a language that may improve tagging accuracy that are not captured by the tagged sets?
A: Tense of a sentence, etc.



Part D
=====================================================================
1).
(hw1) [ff242@hawk Homework1]$ python3 solutionsD.py
0 0.9655823672419682
0 0.9672237083345857
Test Accuracy: 96.691

The result is better than the HMM based tagger, by a little over 3%. 



Timing
=====================================================================
(hw1) [ff242@hawk Homework1]$ python3 solutionsA.py 
Part A time: 12.377512 sec
(hw1) [ff242@hawk Homework1]$ python3 solutionsB.py 
Part B time: 23.101248 sec
(hw1) [ff242@hawk Homework1]$ python3 solutionsC.py 
Part C time: 293.936108 sec




