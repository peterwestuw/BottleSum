# BottleSumm

Code for "BottleSum: Self-Supervised and Unsupervised Sentence Summarization using the Information Bottleneck Principle"

## Simple

compatibility:
```
pytorch 0.4 
pytorch-pretrained-bert
sacremoses
nltk
tqdm
```

To run bottleSumEx with the same settings as in the paper

```
python bottleEx_summarize.py -S1_path <S1FILE> -S2_path <S2FILE> -rem_words 3 -out_name <OUTNAME>
```
<S1FILE> and <S2FILE> would contain one source sentence (s1) or next sentence (s2) per line.
See code for more options.

To generate data for training BottleSum^Self, use the above command (these are the same setting used in the paper. 

We are currently producing more in detailed instructions for training BottleSum^Self. If you are planning to use BottleSum^self, please contact us for instructions, as this process is more involved.


