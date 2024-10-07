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
    kernel = np.zeros((rowCnt,colCnt))
    
    # Starting at index = 2
    index = 2
    # loop through each row and column of the kernel/filter and store the correct token (converted to a float)
    for i in range(kernel.shape[0]):
        kernel[i] = float(tokens[index])
        for j in range(kernel.shape[1]):
            kernel[j] = float(tokens[index])
            index += 1
    
    return kernel
            
    
    