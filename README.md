# BottleSumm

Code for "BottleSum: Self-Supervised and Unsupervised Sentence Summarization using the Information Bottleneck Principle"

## Temporary instructions

We will update with more comprehensive instructions, but will include this for researchers who need to run this code immediately

Dependency note: This repo was run using pytorch 0.4 and the predecessor to the huggingface transformers package (pretrained BERT)

Example code for BottleSum^Ex:
```
python bottleEx_summarize.py -S1_path S1.txt -S2_path S2.txt -rem_words 3 -out_name out.txt
```
S1.txt and S2.txt would contain one source sentence (s1) or next sentence (s2) per line.
See code for more options.

If you are planning to use BottleSum^self, please contact us for instructions, as this process is more invovled.
