# Classifier Comparison
### Summary
In this work, the performance of several classifiers is assessed on two public datasets: one involving fault detection on steel plates and one on gesture recognition in motion capture data.

The classifers that are analyzed include:
* Least Squares (including Tikhonov regularization) - implemented both in analytical closed form and using gradient descent
* Support Vector Machines  - implemented using gradient descent
* Simple Feed-forward neural network using PyTorch

All of the executable code lies in the /Code/ directory and was written in Python3.

The original datasets have been converted to csv files and are preprocessed into numpy-friendly arrays in the included PreProcessData.py scripts.



#### Specific File Functions
* LSQ.py - includes implementation and testing code for least squares routines
* lSVM.py - includes implementation and testing code for support vector machines
* OnevAll.py - intended to be called by a cross validation routine. Performs one vs all testing for
a given training, regularization test, and test set.
* PreProcessData.py - reads csv files, peforms basic preprocessing such as normalization of data columns (to account for vastly different units), and stores data in numpy arrays
* test2Ddata.py - contains some contrived 2D data that is used to validate algorithms with known results.