import numpy as np

def read_kernel_file(filepath):
    # Open a file for reading and grab the first line
    with open(filepath, 'r') as file:
        line = filepath.readline()
    # Split the line into tokens by spaces
    tokens = line.split()
    # Grab 1st and 2nd tokens, convert to ints, store as row and column counts
    rowCnt = int(tokens[0])
    colCnt = int(tokens[1])
    # np zeros array of shape (rowCnt, colCnt) to store kernel values
    kernArray = np.zeros((rowCnt,colCnt))
    for i in range(2, kernArray.shape[0]):
        kernArray[i] = tokens[i]
    