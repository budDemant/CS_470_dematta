import numpy as np
import cv2

def getOneLBPLabel(subimage):
    
    # Get center pixel
    centerPixel = subimage[1, 1]
    
    # Compares each pixel of the subimage to the center pixel, then converts boolean values to 1s and 0s
    binaryValues = (subimage > centerPixel).astype(int)
    
    # 2D to 1D array (ex: [1, 0, 0, 0, 1, 0, 0, 0]), and excludes center pixel
    binaryValues = np.delete(binaryValues.flatten(), 4)
    
    # "".join converts ['1','0,'1','0'] to '1010'
    binaryString = "".join(binaryValues.astype(str))
    
    # Generate all rotated versions of binaryString
    rotations = []
    for i in range(len(binaryString)):
        rotations.append(binaryString[i:] + binaryString[:i])
    
    # rotation invariance
    rotationMin = min(rotations)
    
    # binary string interpreted as base-2 integer
    minDecimal = int(rotationMin, 2)
    
    # if one binary number doesn't equal the next binary number, there's a transition
    transitions = 0
    for i in range(len(rotationMin)):
        if rotationMin[i] != rotationMin[(i + 1) % len(rotationMin)]:
            transitions +=1

    if transitions <= 2:
        # all possible uniform patterns
        uniformPatterns = [
            "00000000", "11111111",  
            "00000001", "10000000",  
            "00000111", "11100000",  
            "01111111", "11111110",  
            "00111111", "11110000"
        ]
        
        # finds which uniform pattern (if any) correlates to the binary string with the least rotations
        lbpLabel = uniformPatterns.index(rotationMin)
    
    else:
        lbpLabel = 10
    
    return lbpLabel

def getLBPImage(image):
    
    # You may use cv2.copyMakeBorder (with a padding of 1 on all sides).
    paddedImage = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value = 0)
    
    # The output label image will be the same size and type as the input image.
    height, width = image.shape
    lbpImage = np.zeros((height,width), dtype=np.uint8)
    
    