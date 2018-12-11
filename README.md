# Signals_Systems_Amazon_NLP
Natural Language Processing project for 625.260 - Fall 2018

## What is this?
This is my final project for Introduction to Signals and Systems, Fall 2018. Using the Keras library, I've built a Recurrent Neural Network which leverages Long Short-Term Memory to perform an analysis on the Amazon Review dataset organized and deduplicated by Julian McAuley

This neural network seeks to take up to 300 words of a review and output an estimated score based on those reviews. Although we are not solving a novel problem, it provides a good framework for sentiment classification that may boil down a longwinded review (or README.md) into an easy to understand 1-5 (or in the case of our model, 0-4) score. We could likely extend this architecture into a recommender system off of the same dataset.

## Network architecture
The network architecture here is fairly simple, using the 300d glove vectors provided by [GloVe](https://nlp.stanford.edu/projects/glove/) at Stanford university to form an embedding for the [Amazon product data](http://jmcauley.ucsd.edu/data/amazon/) graciously curated and provided by Julian McAuley. The input that is mapped by the aforementioned embedding is then passed to two of Keras's Long Short-Term Memory layers. These initial two layers are followed by Dropout layer, which will "deactivate" some number of neurons (in this case, 20% of our neurons), which seeks to reduce overfitting of our model, and a third LSTM layer before being evaluated by a softmax to assess what "score" the input should receive. 

Long Short-Term Memory is particularly interesting from the perspective of feedback systems. Essentially, they are a standard neuron within a neural net, but also contain "gates" which allow the neural net to maintain some level of context around sequence data. 

![lstm block diagram](https://imgur.com/a/NmBr5Uh)
<super>Credit to Jitong Chen and DeLiang Wang for the diagram.</super>

The input gate i<sub>t</sub> takes our old cell state, C<sub>t</sub> and gives us a new candidate value. By "remembering" the state of the previous cell or cells, 

### Citations
R. He, J. McAuley. Modeling the visual evolution of fashion trends with one-class collaborative filtering. WWW, 2016

J. McAuley, C. Targett, J. Shi, A. van den Hengel. Image-based recommendations on styles and substitutes. SIGIR, 2015

J. Chen, D. Wang. Long Short-Term Memory for Speaker Generalization in Supervised Speech Separation. 2016