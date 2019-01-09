# Word Embedding

Embeddings are vector representation of an entity, most commonly, with useful information stored in the vectors. Embeddings, in the sense of text corpus, can be of characters, words, phrases, sentences or even documents. Let us look at the different embeddings there.

Word embeddings are the vector representation of words in a corpus. It can be as simple as one-hot vectors, where only the word is marked 1, for the rest all the words of the vocabulary, it is marked 0 in the vector. Word embeddings can also be computed using complex neural networks, while considering the semantic similarities between the words.

There are different ways to represent words as numbers, i.e. vectors.

## Simple Frequency based practises:

One of the simple way, one-hot vector, was mentioned above. Basically, the vector size is the number of words in the vocabulary, i.e. total number of distinct, unique words in the whole corpus. A word is represented by marking 1 under its column, and rest all the points of vectors are marked 0.

Tf-Idf is another way to find the word embeddings of words in a corpus. In Tf-Idf embedding are found use 2 components Term Frequency and Inverse Document frequency. Term frequency is the number of times a word appeared in a document, divided by the total number of words in that document. It’s the measure of how important a word is to the document. 

Inverse Document frequency is the log of total number of documents in the corpus divided by the number of documents a term appeared in it. IDF is the measure of that terms importance in the document. If that term appears in almost all the documents, it must not be useful for the document. TF-IDF of a term is then found by multiplying TF value with IDF value of the term.

Another way is to find the co – occurrence matrix. Here, in the corpus, the occurrence of each pair of words in a Context Window is found. The context window is a number representing how many words to consider in the context of another word. If context window is 2, then for a word, 4 words will be co-occurring with that word (2 before and 2 after). So a co – occurrence matrix is a square matrix of size equal to total number of words in the vocabulary. A vector of the word can be taken as either of the column or row of that word in the co – occurrence matrix.

## Word2Vec

In word2vec a single layer neural network is made to predict the context words of a word. The number of context words to be predicted is determined by the context window size. This technique is also known as skip-gram models. There are another models known as CBOW models, which predict the word, given the context words.

In word2vec, the neural network is inputted the word and made to predict the context words. Once the neural network is trained, the weights of hidden layers are taken as the Word embedding for the word taken as the input. The size of hidden layer determines the vector size of the word embedding. The semantic information is embedded in the vectors as the whole pupose of neural network being trained is to predict the context words.

The Word2Vec embedding can be downloaded from -> [https://github.com/mmihaltz/word2vec-GoogleNews-vectors](https://github.com/mmihaltz/word2vec-GoogleNews-vectors)

## GloVe

The glove vectors are computed from the co-occurrence matrix and since it see’s the overall text, the GloVe vectors have information of the word distribution globally in the corpus. Also semantic relations are intact since the objective of GloVe is to consider the co-occurrence of 2 words. As given in there paper:

The training objective of GloVe is to learn word vectors such that their dot product equals the logarithm of the words’ probability of co-occurrence. Owing to the fact that the logarithm of a ratio equals the difference of logarithms, this objective associates (the logarithm of) ratios of co-occurrence probabilities with vector differences in the word vector space. Because these ratios can encode some form of meaning, this information gets encoded as vector differences as well. For this reason, the resulting word vectors perform very well on word analogy tasks, such as those examined in the word2vec package.

Link to GloVe -> [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)

## FastText

All the embeddings discussed so far are meant for words. FastText is similar to word2Vec, but they are meant for sub-word information. FastText is trained very similar to word2Vec with difference being that now the words are divided into sub words. Example: “banana” will be divided into “ba”, “na” and “na”, which will have their own embeddings (in this case both the “na” sub word are same and will have same embedding). 

Using FastText embedding even out of Vocabulary word embedding can be estimated, since the subword forming the OOV word may have useful information. FasText can be also used to get the character vectors, where each character will have their own vector.

FastText embedding file can be downloaded from ->  [https://fasttext.cc/docs/en/english-vectors.html](https://fasttext.cc/docs/en/english-vectors.html)

## Phrase, Sentence, Document Embedding’s

Some of the simple ways for getting embedding of components formed by words like phrases, sentences, or documents are:

1.	Get the mean (average) of the word embeddings of the words forming that component.

2.	Concatenate the word embeddings of the words of the component. This method may not be very useful since it creates very large sized vectors, as the sequence length increases.

3.	Store and use different distribution (max, min, avg) of the word embeddings of the words forming that component.

There is also an approach known as ‘BERT’ to get sentence embedding, which is explained more [here].



