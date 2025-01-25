# Data

We plan to start with English-only data for our first models (including common punctuation and whitespace characters). We will use large text datasets,
starting with the [American National Corpus](https://anc.org/data/masc/corpus/), a 500k word dataset of public domain English text. We will use all
of the text within this document to build our lookup table (see Methods), which we can easily save to a file for loading later (as a `.csv`).

For each document in this corpus, some preprocessing such as removing excess whitespace will be necessary. Then, we can get all trigrams in the text
for use in our lookup table.

This dataset is useful because it contains both written and spoken English text.

# Methods

Our proposed method is to fit a lookup table of character bigram keys to the top-three most probable next characters. (In the case we are given a )
This is a preliminary model to get a sense of the scope of the problem. 
We plan to implement this in Python.

## Future Directions

TODO

In later iterations, we plan to experiment with neural network architectures
with token-based (as opposed to character-based) feature representations.