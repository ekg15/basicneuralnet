
A neural network I developed in my free time just to understand the concepts.
=============================================================================

It has been tested on the MNIST hand-drawn numeral dataset and has achieved notable accuracy improvements from its initial random distribution of weights.

Roughly, it is organized as such:

Node Class:
Contains simple methods for generating its activation value from it's input nodes and associated weights. This will probably be replaced with a true dot product operation soon.

Layer Class:
Centered around a list of Nodes, this object contains most of the math involved for gradient descent (most complex and important part) in the form of a number of operations on the aforementioned list.

Network Class:
Centered around a list of Layers, this object contains methods for quickly and easily setting up a network of arbitrary size and running the learning process on properly formatted training data.

The purpose of ImageFormatting.py is self explanatory. It's pretty miscellanious in there.
