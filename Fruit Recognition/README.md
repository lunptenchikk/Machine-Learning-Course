# Fruit Recognition

This project is a simple image classification model that recognizes different types of fruit from images.

The model takes an image as input, preprocesses it (resizing, normalization, tensor conversion), and feeds it into a convolutional neural network. The network extracts visual features such as edges, shapes, colors, and textures, and then predicts the most likely fruit class using a softmax layer.

The model is trained using cross-entropy loss and optimized with backpropagation. During evaluation, it outputs the predicted class and can be used to measure accuracy on a validation or test set.

The goal of this project is to demonstrate a practical computer vision pipeline — from raw image data to final classification — using deep learning techniques.
