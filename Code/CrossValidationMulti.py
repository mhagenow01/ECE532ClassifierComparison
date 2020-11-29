#!/usr/bin/env python

""" Creates training, regularization, and validation sets from a
given set of multiclass data to be used for the one vs all testing

 Created: 11/24/2020
"""

__author__ = "Mike Hagenow"

import numpy as np
from PreProcessData import loadFaults
from LSQ import lsq, wlsq
from lSVM import lsvm
from OnevAll import onevall

# Create the sets for the cross validation
# divided into training segments, regularization segments, and validation segments
def crossValidation(X_faults,num_segs,lams,classfxn):

    for ii in range(0, num_segs):  # set for determining which regularization parameter
        for jj in range(0, num_segs):  # validation set
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
            onevall(X_training, X_reg, X_test, lams, classfxn)

def main():
    X_faults = loadFaults()
    crossValidation(X_faults,4,[0.0],wlsq)


if __name__ == "__main__":
    main()




