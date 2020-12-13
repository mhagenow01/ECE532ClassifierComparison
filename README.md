# A Comparison of Classifiers for Classification ofSteel Plate Faults and Motion Capture Data
#### Mike Hagenow
Department of Mechanical Engineering - ECE 532 Final Project
### Summary
In this work, the performance of several classifiers is assessed on two public datasets from the UCI repository: one involving fault detection on steel plates and one on gesture recognition in motion capture data.

![Fault in a Steel Plate](https://www.bactoforce.com/wp-content/uploads/2014/08/figure3.jpg)
![Motion Capture on Hand](https://www.researchgate.net/profile/Xiaopeng_Yang3/publication/238693963/figure/fig1/AS:298744454107149@1448237636658/Marker-set-for-hand-motion-capture.png)

(Examples of a fault in a plate and motion capture hand tracking. Sources: [Bactoforce](https://www.bactoforce.com/detection-biofilms/) and [ResearchGate](https://www.researchgate.net/figure/Marker-set-for-hand-motion-capture_fig1_238693963))

Links to Datasets:
* [Steel Plate Faults](https://archive.ics.uci.edu/ml/datasets/Steel+Plates+Faults)
* [Motion Capture Hand Postures](https://archive.ics.uci.edu/ml/datasets/Motion+Capture+Hand+Postures)


The classifers that are analyzed include:
* Least Squares (including Tikhonov regularization) - implemented both in analytical closed form and using gradient descent
* Support Vector Machines  - implemented using gradient descent
* Simple Feed-forward neural network using PyTorch (1 hidden layer with 1000 Nodes)

All of the executable code lies in the /Code/ directory and was written in Python3.

The original datasets have been converted to csv files and are preprocessed into numpy-friendly arrays in the included PreProcessData.py script.

PDF versions of project writeups are available in the /Documents/ directory

### Requirements
* Python 3 (tested on 3.6.9)
* Numpy (tested on 1.18.5)
* PyTorch
* Cleanlab (https://github.com/cgnorthcutt/cleanlab)
* sci-kit learn

### Usage
NOTE: The code is tested using **Ubuntu 18.04**. The code for loading CSV files is based on a linux filesystem.
For a different filesystem, minor changes may be required in **/Code/PreProcessData.py**.

All of the main report results can be run using the runExperiments script in the code directory. Note: These scripts assume that your
 current directory is the /Code/ directory. In that directory, you can use the following syntax to run examples:

`python3 runExperiments.py [test]`

The tests available are:
* genclass - run five-fold cross validation for the three methods on the steel plate dataset
* regTestl2 - run a analysis of the impact of different l2 regularization coefficients and the impact
on the LSQ and SVM methods
* trainingAcc - compute training set accuracy for the 3 methods on the steel plate faults (to determine whether the data was separable)
* nncompare - compares 'one' vs 'all' binary classification networks with a single multi-class network
* mocap - analyzes the motion capture data using the 3 methods and a training/test set
* l1 - Runs LSQ with an l1 regularizer to look at what features are driven towards zero
* sota - runs cross validation with the cleanlab method using logistic regression in sklearn

**Also note: many of these scripts can take more than an hour to complete**


### Results
See the final report in the /Documents/ directory for expanded discussion (what could fit in 5 pages!).

**Regression analysis using Steel Plate faults:**

|    | wLSQ | wSVM | NN|  
| ------------- | ------------- | ------------- | ------------- |
| Training Data  | 0.684  | 0.592* | 0.933 |
| Cross Validation (n = 5)  | 0.585 | 0.567 | 0.583 |

(*) dependent on number of iterations of gradient descent. Similar to LSQ for a high number of iterations.

**Classification Accuracy Per Class:**
![Per Class Accuracy](/imgs/classification_per_class.png)

Interestingly, the SVM was comparably to the other methods except on the fourth class, dirtiness. The missed classifications were scattered
across other classes, but mainly Pastry and Other faults.

**l2 Regularization for LSQ and SVM:**
![l2_reg](/imgs/regularization_l2.png)

**l1 Regularization Feature Weights:**
![l1_weights](/imgs/features_l1_faults.png)

**Motion Capture Analysis (Test/Train Sets)**

|    | wLSQ | wSVM | NN|  
| ------------- | ------------- | ------------- | ------------- |
| Test Set  | 0.172  | 0.240 | 0.881 |


#### Specific File Functions
* LSQ.py - includes implementation and testing code for least squares routines
* lSVM.py - includes implementation and testing code for support vector machines
* simpleNN.py - includes implementation for a custom neural network structure using Pytorch (Adam Solver, ReLu + Sigmoid - 1 hidden layer)
* OnevAll.py - intended to be called by a cross validation routine. Performs one vs all testing for
a given training, regularization test, and test set.
* PreProcessData.py - reads csv files, peforms basic preprocessing such as normalization of data columns (to account for vastly different units), and stores data in numpy arrays
* test2Ddata.py - contains some contrived 2D data that is used to validate algorithms with known results.
* CrossValidationMulti.py - breaks data into folds and does the one vs all testing for each combination
* generatePlots.py - with data from the experiments script preloading, simply generate plots using matplotlib
* runExperiments.py - main entry script runs a variety of experiments