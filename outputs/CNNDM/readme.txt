This directory contains results related to 
BottleSum: Self-Supervised and Unsupervised Sentence Summarization using the Information Bottleneck Principle (West et al.)

This directory contains the following files:

BottleSum^Self files:
cnn_test_s1.txt: This is the full set of sentences from the cnn/dm test set. These are used as input to BottleSum^Self
cnn_test_bottleSelf_out.txt: this is the output of BottleSum^Self on cnn_test_s1.txt, trained using BottleSum^Ex outputs on the cnn/dm trianing set with the same parameters as the the paper


BottleSum^Ex files:
cnn_test_ex_s1.txt: The subset (same order) of cnn_test_s1.txt sentences for which a next sentence is available (i.e. all but last sentence for each article)
cnn_text_ex_s2.txt: The 'next sentences' for cnn_test_ex_s2.txt
cnn_test_bottleEx_out.txt: The output of BottleSum^Ex on the above files using the same parameters as the paper
