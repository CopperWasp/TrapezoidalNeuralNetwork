import csv
import numpy as np
from sklearn import preprocessing

splice = './otherDatasets/splice.csv'


def readSplice():
    dataset=[]
    return_dataset=[]
    with open(splice) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            for i in range(len(row)):
                row[i] = row[i].strip()
            row.pop(1) # patient id
            
            # handle labels
            if row[0] == 'IE': 
                label = 0
            elif row[0] == 'EI': 
                label = 1
            elif row[0] == 'N': 
                label = 2

            row.pop(0)
            
            # handle attributes
            separated = list(row[0])
            encoded_row = []
            d = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
            for char in separated:
                for i in range(4):
                    if char in d:
                        if(i == d[char]):
                            encoded_row.append(1)
                        else:
                            encoded_row.append(0)
                    else:
                        encoded_row.append(0)
            
            # add the class label
            encoded_row.append(label)
            dataset.append(encoded_row)     

    # transform into dictionary
    numpy_dataset = np.array(dataset).astype(np.float)
    #numpy_dataset[:,:-1]=preprocessing.scale(numpy_dataset[:,:-1])
    for row in numpy_dataset:
        mydict = {v: k for v, k in enumerate(row)}
        mydict['class_label'] = mydict.pop(240)
        return_dataset.append(mydict)
    return return_dataset
        