#!/usr/bin/env python

""" Starting script that uses command line arguments to run the evaluations
used in the final report
 Created: 12/5/2020
"""

__author__ = "Mike Hagenow"

import sys
import numpy as np
from PreProcessData import loadFaults
from CrossValidationMulti import crossValidation

# Import the classifier models
from LSQ import lsq, wlsq
from lSVM import lsvm, wlsvm
from simpleNN import nn


def noRegFiveFoldClassification():
    X_faults = loadFaults()
    lam = [0.0]
    crossValidation(X_faults, 5, lam, wlsq)
    crossValidation(X_faults, 5, lam, wlsvm)
    crossValidation(X_faults, 5, lam, nn)

def runEvaluations(testname):
    if testname == "genclass":
        noRegFiveFoldClassification()
    else:
        print("Test [",testname,"] Not Found")

if __name__ == "__main__":
    testname = sys.argv[1]
    runEvaluations(testname)




