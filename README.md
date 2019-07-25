# Planar-Data-Classifying-Neural-Network

Implemented Two-Class Planar Data Classification using a Neural Network with a single hidden layer and logistic regression to analyze the difference in performance.

## Packages

* numpy
* sklearn
* matplotlib
* planar_utils


## Dataset

The dataset contains points that form a Planar Flower. These points are either red (label y=0) or blue (label y=1), forming petals of alternate colors. The model will predict the color of a point based on its relative location.

## Logistic Regression

Using the existing regression model in the sklearn library, Logistic Regression classifies the dataset with a low accuracy.

## Neural Network

A neural network with one hidden network of size 4 is used for classification. 

    ### Neural Network Structure:
        1) Define number of inputs, dimensions of the layers.
        2) Initialize parameters randomly
        3) Loop:
            * Forward Propagation
            * Loss Computation
            * Back Propagation 
            * Update Parameters
            

## Result

The neural net achieves an accuracy of 90%, a great improvement as compared to the 47% achieved through Logistic Regression
