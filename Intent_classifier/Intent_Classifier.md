# Intent Classification

Intent classification is a step in NLU, where we need to understand what does the user want, by processing the user query.

A simple yet powerful chatbot can be made by doing an intent classification and than replying with one of the pre written messages.

Example queries for a chatbot to find places around:

1.	“I need to buy groceries.”: Intent is to look for grocery store nearby
2.	“I want to have vegetarian food.”: Intent is to look for restaurants around, ideally veg.

So basically, we need to understand what is user looking for, and accordingly we need to classify his request to certain category of intent. Once the intent is known, a pre written message can be generated according to the intent. In the first example a message would be:
Chatbot reply ->	Some of the grocery store around you are GROCERY_1, address …

To understand the intent of an user query, we need to train a model to classify requests into intents using a ML algorithm, over the sentences transformed to vectors using methods like TF-IDF, word2Vec, GloVe. First, we convert sentences into array of numbers(vector).

![General flow of intent classification, setences->vectors->model](./Intent_classification.png?raw=true "General flow of intent classification, setences->vectors->model")

Tf-Idf: Give each word in sentence a number depending upon how many times that word occurs in that sentence and also upon the document. For words occurring many times in a sentence and not so many occurrences in document will have high value.	

Word2Vec: There are different methods of getting Word vectors for a sentence, but the main theory behind all the techniques is to give similar words a similar vector representation. So, for words like man and boy and girl will have similar vectors. The length of each vector can be set. Example of Word2Vec techniques:  GloVe, CBOW and skip-gram.

We can use Word2Vecs by training it for our own dataset(if we have enough data for the problem), or else we can use pretrained WordVecs available on internet. Pretrained Wordvec have been trained on huge documents like Wikipedia data, Tweets, etc. and its almost always good for the problem.


