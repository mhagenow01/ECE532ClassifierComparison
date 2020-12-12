#!/usr/bin/env python

""" Starting script that uses command line arguments to run the evaluations
used in the final report
 Created: 12/5/2020
"""

__author__ = "Mike Hagenow"

import sys
import numpy as np
from PreProcessData import loadFaults, loadMocap
from CrossValidationMulti import crossValidation
from OnevAll import onevall, onevallNN
import torch

# Import the classifier models
from LSQ import lsq, wlsq, wlsqPGD
from lSVM import lsvm, wlsvm
from simpleNN import nn, nnMultliClass

# State of the art method
from cleanlab.classification import LearningWithNoisyLabels
from sklearn.linear_model import LogisticRegression


def noRegFiveFoldClassification():
    X_faults = loadFaults()
    lam = [0.0]

    print("\n\n------------------------")
    print("| Running Cross Validation |")
    print("------------------------")

    crossValidation(X_faults, 5, lam, wlsq)
    crossValidation(X_faults, 5, lam, wlsvm)
    # crossValidation(X_faults, 5, lam, nn)

def trainingAcc():
    X_faults = loadFaults()
    lam = [0.0]

    print("\n\n---------------------------------")
    print("| Running One v All on Training Set |")
    print("-------------------------------------")

    # Run everything with the neural network
    overall_acc, not_used = onevallNN(X_faults, X_faults, X_faults)
    print("Neural Network Classification Accuracy: ", overall_acc)

    # Run everything with least-squares
    overall_acc, not_used, not_used = onevall(X_faults, X_faults, X_faults, lam, wlsq)
    print("wLSQ Classification Accuracy:", overall_acc)

    # Run everything with SVM
    overall_acc, not_used, not_used = onevall(X_faults, X_faults, X_faults, lam, wlsvm)
    print("SVM Classification Accuracy: ", overall_acc)

def effectRegularizationLSQ_SVM():
    X_faults = loadFaults()
    lams = np.logspace(-6,1,25)

    print("\n\n------------------------------")
    print("| Running Regularization Analysis |")
    print("-----------------------------------")

    results_lsq = []
    results_svm = []

    for ii in range(0,len(lams)):
        print("\n\n------------------------------")
        print("| LAM ",ii," of ",len(lams)," |")
        print("-----------------------------------")
        results_lsq.append(crossValidation(X_faults, 5, [lams[ii]], wlsq))
        results_svm.append(crossValidation(X_faults, 5, [lams[ii]], wlsvm))

    print("\n\n------------------------------")
    print("| Overall Best Classification     |")
    print("-----------------------------------")
    crossValidation(X_faults, 5, lams, wlsq)
    crossValidation(X_faults, 5, lams, wlsvm)

    print("Results for Each Regularizer:")
    print(results_lsq)
    print(results_svm)


def compareNNTopology():
    X_faults = loadFaults()

    X_faults_train = []
    X_faults_test = []

    # Split into test and training sets

    for ii in range(0,len(X_faults)):
        num_samps = np.shape(X_faults[ii])[0]
        train_inds = np.random.choice(num_samps,int(0.8*num_samps),replace=False)
        test_inds = [i for i in range(0,num_samps) if i not in train_inds]

        X_faults_train.append(X_faults[ii][train_inds,:])
        X_faults_test.append(X_faults[ii][test_inds,:])

    # Neural network one vs all
    class_acc, acc_per_class = onevallNN(X_faults_train,X_faults_train,X_faults_test)

    # Neural network multiclass
    net = nnMultliClass(X_faults_train)

    num_correct = 0
    num_total = 0
    per_class_multi = []

    for ii in range(0,len(X_faults_test)):
        total_in_class = np.shape(X_faults_test[ii])[0]
        class_correct = 0
        for jj in range(0,np.shape(X_faults_test[ii])[0]):
            temp = net.forward(torch.Tensor(X_faults_test[ii][jj, :]))
            if np.argmax(temp.detach().numpy())==ii:
                class_correct = class_correct + 1.0
                num_correct = num_correct + 1.0
            num_total = num_total + 1.0
        per_class_multi.append(class_correct/total_in_class)
    class_acc_multi = num_correct/num_total

    print("\n\n------------------------------")
    print("| Neural Network Results          |")
    print("-----------------------------------")
    print("One vs All NN:")
    print("Total Acc: ",class_acc)
    print(acc_per_class)
    print(" ")
    print("MultiClass NN:")
    print("Total Acc: ", class_acc_multi)
    print(per_class_multi)

def compareMocap():
    X_faults = loadMocap()

    X_faults_train = []
    X_faults_test = []
    X_faults_reg = []

    # Split into test and training sets

    for ii in range(0,len(X_faults)):
        num_samps = np.shape(X_faults[ii])[0]
        train_inds = np.random.choice(num_samps,int(0.25*num_samps),replace=False)
        remaining_inds = np.array([i for i in range(0,num_samps) if i not in train_inds])
        reg_inds = remaining_inds[np.random.choice(len(remaining_inds),int(0.25*len(remaining_inds)),replace=False)]
        test_inds = np.array([i for i in range(0,num_samps) if i not in train_inds and i not in reg_inds])

        # # Further downsample
        # downsamp_per = 0.25
        # train_inds = train_inds[np.random.choice(len(train_inds),int(len(train_inds)*downsamp_per), replace=False)]
        # test_inds = test_inds[np.random.choice(len(test_inds),int(len(test_inds)*downsamp_per), replace=False)]

        X_faults_train.append(X_faults[ii][train_inds,:])
        X_faults_test.append(X_faults[ii][test_inds,:])
        X_faults_reg.append(X_faults[ii][reg_inds,:])

    # Run for 10 regularization parameters
    lams = np.logspace(-6, 1, 10)

    # One vs all testing for the 3 methods
    print("Calculating LSQ...")
    class_acc_lsq, acc_per_class_lsq, not_used = onevall(X_faults_train, X_faults_reg, X_faults_test,lams=lams,classfxn=wlsq)
    class_acc_lsq_training, _, _ = onevall(X_faults_train,X_faults_train,X_faults_train,lams=[0.0],classfxn=wlsq)
    print("Calculating SVM...")
    class_acc_svm, acc_per_class_svm, not_used = onevall(X_faults_train, X_faults_reg, X_faults_test, lams=lams, classfxn=wlsvm)
    class_acc_svm_training, _, _ = onevall(X_faults_train, X_faults_train, X_faults_train, lams=[0.0], classfxn=wlsvm)
    print("Calculating NN...")
    class_acc_nn, acc_per_class_nn = onevallNN(X_faults_train,X_faults_reg,X_faults_test)
    class_acc_nn_training, _ = onevallNN(X_faults_train, X_faults_train, X_faults_train)


    print("\n\n------------------------------")
    print("|       Mocap Results            |")
    print("-----------------------------------")
    print("Least Squares: ")
    print("Total Acc: ",class_acc_lsq)
    print("Training Acc: ", class_acc_lsq_training)
    print(acc_per_class_lsq)
    print(" ")
    print("SVM: ")
    print("Total Acc: ", class_acc_svm)
    print("Training Acc: ", class_acc_svm_training)
    print(acc_per_class_svm)
    print(" ")
    print("NN: ")
    print("Total Acc: ", class_acc_nn)
    print("Training Acc: ", class_acc_nn_training)
    print(acc_per_class_nn)


def l1_analysis():
    X_faults = loadFaults()

    X_faults_train = []
    X_faults_test = []
    X_faults_reg = []

    # Split into test and training sets

    for ii in range(0,len(X_faults)):
        num_samps = np.shape(X_faults[ii])[0]
        train_inds = np.random.choice(num_samps,int(0.60*num_samps),replace=False)
        remaining_inds = np.array([i for i in range(0,num_samps) if i not in train_inds])
        reg_inds = remaining_inds[np.random.choice(len(remaining_inds),int(0.50*len(remaining_inds)),replace=False)]
        test_inds = np.array([i for i in range(0,num_samps) if i not in train_inds and i not in reg_inds])

        X_faults_train.append(X_faults[ii][train_inds,:])
        X_faults_test.append(X_faults[ii][test_inds,:])
        X_faults_reg.append(X_faults[ii][reg_inds,:])

    # Run for 10 regularization parameters
    lams = [0.1]

    print("Calculating LSQ l1...")
    class_acc_lsq, acc_per_class_lsq, best_ws = onevall(X_faults_train, X_faults_reg, X_faults_test,lams=lams,classfxn=wlsqPGD)

    print("\n\n------------------------------")
    print("|       L1 Results                |")
    print("-----------------------------------")
    print("Least Squares: ")
    print("Total Acc: ",class_acc_lsq)
    print(acc_per_class_lsq)
    print("---")

    ws = best_ws[0].reshape((1,len(best_ws[0])))
    for ii in range(1,len(best_ws)):
        ws = np.concatenate((ws,best_ws[ii].reshape((1,len(best_ws[ii])))),axis=0)

    print("Best Ws (mean + stddev):\n",list(np.mean(np.abs(ws),axis=0)),"\n",list(np.std(np.abs(ws),axis=0)))

def state_of_the_art():
    X_faults = loadFaults()
    lam = [0.0]

    print("\n\n------------------------")
    print("| Clean Lab Analysis |")
    print("------------------------")

    crossValidation(X_faults, 5, lam, 'clab')

def runEvaluations(testname):
    if testname == "genclass":
        noRegFiveFoldClassification()
    elif testname == "regTestl2":
        effectRegularizationLSQ_SVM()
    elif testname == "trainingAcc":
        trainingAcc()
    elif testname == "nncompare":
        compareNNTopology()
    elif testname == "mocap":
        compareMocap()
    elif testname == "l1":
        l1_analysis()
    elif testname == 'sota':
        state_of_the_art()
    else:
        print("Test [",testname,"] Not Found")

if __name__ == "__main__":
    testname = sys.argv[1]
    runEvaluations(testname)




