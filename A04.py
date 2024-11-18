import numpy as np

def getOneLBPLabel(subimage):
    
    # Get center pixel
    centerPixel = subimage[1, 1]
    
    # Compares each pixel of the subimage to the center pixel, then converts boolean values to 1s and 0s
    binaryValues = (subimage > centerPixel).astype(int)
    
    # 2D to 1D array (ex: [1, 0, 0, 0, 1, 0, 0, 0]), and excludes center pixel
    binaryValues = np.delete(binaryValues.flatten(), 4)
    
    # "".join converts ['1','0,'1','0'] to '1010'
    # int(joinedString, 2) interprets binary string as base-2 number
    lbpLabel = int("".join(binaryValues.astype(str)), 2)
    
    return lbpLabel

