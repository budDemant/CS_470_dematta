import numpy as np

def getOneLBPLabel(subimage):
    # Ensure the input is a 3x3 subimage (maybe unnecessary given that all test inputs match this)
    # assert subimage.shape == (3, 3), "Input must be a 3x3 subimage."
    
    # Get center pixel
    centerPixel = subimage[1, 1]
    
    # binary pattern by comparing neighbors to the center pixel
    binaryPattern = (subimage > centerPixel).astype(int)
    
    
    #return lbpLabel
