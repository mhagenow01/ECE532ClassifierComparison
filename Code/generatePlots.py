#!/usr/bin/env python

""" Scripts that have the outputs from the various tests
and are used to generate plots using matplotlib
 Created: 12/5/2020
"""

__author__ = "Mike Hagenow"

# libraries
import numpy as np
import matplotlib.pyplot as plt


def plotClassificationPerClassFiveFold():

    fig = plt.figure()
    fig.set_size_inches(10, 4)
    # set width of bar
    barWidth = 0.25

    # set height of bar
    bars_lsq = [0.4981350806451612, 0.8565789473684211, 0.8187925998052583, 0.9495238095238095, 0.5545454545454546, 0.542445987654321, 0.3812797125483693]
    bars_svm = [0.3638608870967742, 0.2118421052631579, 0.8866033755274263, 0.8607142857142855, 0.5818181818181818, 0.5897916666666667, 0.01265616362631288]
    bars_nn = [0.3534274193548387, 0.7368421052631577, 0.7944173969490427, 0.8897619047619048, 0.459090909090909, 0.360162037037037, 0.5972857932559424]

    # Set position of bar on X axis
    r1 = np.arange(len(bars_lsq))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    plt.bar(r1, bars_lsq, color='#aecfc8', width=barWidth, edgecolor='white', label='LSQ')
    plt.bar(r2, bars_svm, color='#aec6cf', width=barWidth, edgecolor='white', label='SVM')
    plt.bar(r3, bars_nn, color='#aeb6cf', width=barWidth, edgecolor='white', label='NN')

    # Add xticks on the middle of the group bars
    plt.xlabel('Plate Fault', fontweight='bold')
    plt.ylabel('Classification Accuracy')
    plt.xticks([r + barWidth for r in range(len(bars_lsq))], ['Pastry', 'Z Scratch', 'K Scratch', 'Stains', 'Dirtiness', 'Bumps', 'Other Faults'])

    # Create legend & Show graphic
    plt.legend()
    plt.show()

def plotl2reg():
    fig = plt.figure()
    fig.set_size_inches(10, 4)

    lsq_results = [0.5831989010907443, 0.5831978935823479, 0.584105629993841, 0.5833280949604791, 0.5826847401087043, 0.5823011340102149, 0.5795882440566594,
                   0.5783041932319417, 0.5761366257364016, 0.5751129913643444, 0.5681723159191836, 0.5649605708390829, 0.5625491548553893, 0.5540354413164383,
                   0.5494104071599526, 0.5495495703606512, 0.5482589295283186, 0.5437508880896474, 0.5366661313512047, 0.5267088345466047, 0.5152249223350337,
                   0.5081248561856376, 0.5000104144288514, 0.48622338312268043, 0.4765737742050941]

    svm_results = [0.5439398726979057, 0.5438103390191493, 0.5439398726979057, 0.5439398726979057, 0.5440694063766622, 0.5438103390191493, 0.5439398726979058,
                   0.5438110067185243, 0.5432971855455059, 0.5434250627833576, 0.5429115626524939, 0.5429109018013176, 0.5409835011952944, 0.5414989856574164,
                   0.5415069843535193, 0.5330088992272286, 0.528905030674168, 0.5305548512010947, 0.49787949125057496, 0.47797354131169223, 0.4664825374926007,
                   0.4531502323436194, 0.4470747845000595, 0.447203650479441, 0.33769789096216735]


    lams = np.logspace(-6,1,25)

    plt.semilogx(lams,lsq_results,color='green', label='LSQ')
    plt.semilogx(lams,svm_results,color='blue', label='SVM')

    plt.xlabel('Regularization Coefficient ($\lambda$)')
    plt.ylabel('Classification Accuracy')

    plt.legend()
    plt.show()

if __name__ == "__main__":
    # plotClassificationPerClassFiveFold()
    plotl2reg()



