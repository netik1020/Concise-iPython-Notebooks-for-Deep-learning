# Concise-iPython-Notebooks-for-Deep-learning-on-Images-and-text

This github repository is a collection of [iPython notebooks](https://ipython.org/ipython-doc/3/notebook/notebook.html) for solving different problems in deep learning using [keras API](https://keras.io/) and [tensorflow](https://www.tensorflow.org/) backend. The notebooks are written in as easy to read manner as possible. Everyone is welcome to openly contribute towards this repository and add/ recommend us to add more interesting notebooks.

The notebooks meant for processing/ understanding texts deals with problems like:

1.	Basic entity extraction from text using [Named Entity Recognition](./NER_tagger/) and tagging the text using [POS taggers](./POS_Tagger/).

2.	[Topic modelling using LDA](./Miscellaneous/topic_modeling.ipynb) and [converting pdf documents to readable text format](./Miscellaneous/pdf_To_doc.ipynb) in python.

3.	[Classification of text queries](./Text_Classification/) into positive or negative comments. [GloVe](https://nlp.stanford.edu/projects/glove/) and [FastText](https://fasttext.cc/docs/en/english-vectors.html) embedding were used and multiple architectures including [bidirectional GRU](https://towardsdatascience.com/introduction-to-sequence-models-rnn-bidirectional-rnn-lstm-gru-73927ec9df15),[LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/), [Attention networks](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/) and combinations of these were experimented with.

4.	[Relation extraction from a sentence](./Semantic_Relation_Extraction/), between 2 entities. A model which was aware of the entity words when finding the relation was made. Self-attention and GRU was used for feature extraction

5.	[Intent classifier](./Intent_classifier/) was made to classify incoming queries into one of the intents. This can be used to understand the type of query a user is making in a chat-bot application. Intent can be to book a cab, or to find a restaurant, etc.

## Motivation

We wanted to have this reposisitories for beginers, to have easy to read reference for some common tasks in Deep Learning.

## Packages used

Common Packages – [Tensorflow](https://www.tensorflow.org/), [keras](https://keras.io/), [sklearn](https://scikit-learn.org/), [numpy](http://www.numpy.org/), [pandas](https://pandas.pydata.org/),

Text Based Packages – [NLTK](https://www.nltk.org/), [genism](https://pypi.org/project/gensim/), [pdfminer](https://pypi.org/project/pdfminer/), [keras_self_attention](https://pypi.org/project/keras-self-attention/), [keras_multi_head](https://pypi.org/project/keras-multi-head/)

Image Based Packages –

## Code style

Standard. Python.

## Index  

### Text

1.	[POS tagger](./POS Tagger/) – POS tagging is to find the part-of-speech tag of the words in the sentence. The tags of the words can be passed as information to neural networks. 

1.1.	[NLTK POS tagger](./POS Tagger/POSTagger_NLTK.ipynb) – Using NLTK package to perform the POS tagging.

1.2.	[Stanford POS tagger](./POS Tagger/POSTagger_Stanford_NLTK.ipynb) – Using pre-trained model of Stanford for POS tagging.

2.	[NER tagger](./NER_tagger/) – NER tagging or Named Entity Recognition is to find some common entities in text like name, place, etc. or more subject dependent entity like years of experience, skills, etc. can all be entities while parsing a resume. NER are generally used to find some important words in the document and one can train their own document specific NER tagger.

2.1.	[Stanford NER tagger](./NER_tagger/NER_stanford_NLTK.ipynb) – Pre-trained NER provided by the Stanford libraries for entities – Person, organization, location.

2.2.	Self-trained keras model – An example of training your own NER model.

3.	[Text Classifier](./Text_Classification/) – Have shown examples of different models to classify IMDB dataset . Text classifiers are very useful in tasks like passing queries to relevant department, in understanding customer review like in case of this dataset. 

3.1.	[Text_classifier](./Text_Classification/classification_imdb.ipynb) – Performs text classification by using different architecture/ layers like GRU, LSTM, Sequence-self-attention, multi-head-attention, global max pooling and global average pooling. Different combinations of above layers can be used by passing arguments to a function to train the different models. GloVe and FastText embeddings have been experimented with.

3.2.	Text_Classifer_2 – Here GloVe and FastText embeddings have been used as different features and there features are concatenated just before the dense layers.\

4.	 [Relation Extraction](./Semantic_Relation_Extraction/) – Data from SemEval_task8 is used to show an example of finding relations between two entities in a sentence. A keras model is built with concatenation of RNN features, self-attention features and max pooled features of the entities in the sentence.

5.	[Intent Classifier](./Intent_classifier/) – Intent classifier is performed on a dataset containing 7 different intents. This is an example for how deep learning models can be successfully used to understand the intent of a text query. Customer intent can be determined and then a prewritten text can be generated to answer the user query. A simple yet effective chat-bot can be built this way, determining on the different intents possible and data available for each of those intients.

6.	[Miscellaneous](./Miscellaneous/) –

6.1.	[PDF to Doc](./Miscellaneous/pdf_To_doc.ipynb) – a very useful tool to read the pdf documents in python. PDFminer package is used here.

6.2.	[RegEx](./Miscellaneous/common_regex.md) – Some powerful and commonly used Regular Expressions.

6.3.	[Embeddings](./Miscellaneous/Word_Embedding.md) – A document going through different embeddings including sentence embedding.

6.4. [Topic Modelling](./Miscellaneous/topic_modeling.ipynb) – Topic modelling in text processing is to cluster document into topics according to the word frequencies, or basically, sentences in the document. Each topic are dominated by certain words which can also be extracted. Here topic modelling is done using LDA from sklearn and genism packages.


