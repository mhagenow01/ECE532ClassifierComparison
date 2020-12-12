#!/usr/bin/env python

""" Creates training, regularization, and validation sets from a
given set of multiclass data to be used for the one vs all testing

 Created: 11/24/2020
"""

__author__ = "Mike Hagenow"

import numpy as np
from tqdm import tqdm
from PreProcessData import loadFaults
from LSQ import lsq, wlsq
from lSVM import lsvm, wlsvm
from simpleNN import nn
from OnevAll import onevall, onevallNN, onevallCleanlab

# Create the sets for the cross validation
# divided into training segments, regularization segments, and validation segments
def crossValidation(X_faults,num_segs,lams,classfxn):
    acc_total = 0.0
    total_runs = 0
    acc_per_class = np.zeros((len(X_faults)))

    if(classfxn!='clab'):
        print("\n\nRunning Cross Validation (",num_segs," sets) for ",classfxn.__name__,"\n")
    else:
        print("\n\nRunning Cross Validation (", num_segs, " sets) for Clean Lab\n")

    for ii in tqdm(range(0, num_segs)):  # set for determining which regularization parameter
        for jj in tqdm(range(0, num_segs)):  # validation set
            if ii == jj:
                continue

            # Reset the sets for a new cross-validation run
            X_training = []
            X_reg = []
            X_test = []

            # For each one of the classification classes
            for kk in range(0,len(X_faults)):

                # Split the classification instances into the desired number of classes
                X_sections_temp = np.array_split(X_faults[kk], num_segs,axis=0)
            
                # Build training sets using list comprehension in python!!
                X_training_temp = np.vstack([X_sections_temp[kk] for kk in range(0, num_segs) if kk != ii and kk != jj])
                X_reg_temp = X_sections_temp[ii]
                X_test_temp = X_sections_temp[jj]
                
                X_training.append(X_training_temp)
                X_reg.append(X_reg_temp)
                X_test.append(X_test_temp)
             
             # Run the one vs all classification for this set
            if classfxn is nn :
                acc, acc_per_class_temp, not_used = onevallNN(X_training, X_reg, X_test)
            elif classfxn is 'clab':
                acc, acc_per_class_temp, not_used = onevallCleanlab(X_training, X_reg, X_test)
            else:
                acc, acc_per_class_temp, not_used = onevall(X_training, X_reg, X_test, lams, classfxn)
            acc_total = acc_total + acc

            for aa in range(0,len(acc_per_class_temp)):
                acc_per_class[aa] = acc_per_class[aa] + acc_per_class_temp[aa]

            total_runs = total_runs + 1

    if (classfxn != 'clab'):
        print("\nOverall Classification Accuracy (",classfxn.__name__,"): ",acc_total/total_runs)
    else:
        print("\nOverall Classification Accuracy (Clean Lab): ", acc_total / total_runs)

    for ii in range(0,len(X_faults)):
        print("   class ",ii,": ",acc_per_class[ii]/total_runs)

    return acc_total/total_runs

def main():
    X_faults = loadFaults()
    lam = np.logspace(-5,-2,20)
    lam = [0.0]
    crossValidation(X_faults,5,lam,wlsq)
    crossValidation(X_faults,5,lam,wlsvm)
    crossValidation(X_faults,5,lam,nn)


if __name__ == "__main__":
    main()




