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
    
    # Loop through each pixel in the image, cut out the appropriate subimage...
    for i in range(1, height + 1):
        for j in range(1, width + 1):
            # 3x3 neighborhood extracted for each pixel
            subimage = paddedImage[i-1:i+2, j-1:j+2]
            
            lbpLabel = getOneLBPLabel(subimage)
            
            lbpImage[i-1, j-1] = lbpLabel
            
    return lbpImage




def getOneRegionLBPFeatures(subImage):
    
    # to prepare histogram values
    flattenSubImage = subImage.flatten()
    
    # labels 0-10
    labels = 11
    
    # + 1 because upper bound is exclusive for hist
    labelsArrange = np.arrange(labels + 1)
    
    hist, _ = np.histogram(flattenSubImage, bins=labelsArrange, range=(0,labels))
    
    # normalize
    totalPixels = flattenSubImage.size
    normalizedHist = hist.astype(float) / totalPixels
    
    return normalizedHist




def getLBPFeatures(featureImage, regionSideCnt):
    
    # subregion dimensions
    height, width = featureImage.shape
    subregionHeight = height // regionSideCnt
    subregionWidth = width // regionSideCnt
    
    # Start with an empty list to hold all of the individual histograms.
    allHists = []
    
    # Loop through each possible subregion, going row by row and then column by column.
    
    for i in range(regionSideCnt):
        for j in range(regionSideCnt):
            
            # extract subregion
            # row * subregionHeight gives first row of current subregion
            startRow = i * subregionHeight
            startCol = j * subregionWidth
            
            subRegion = featureImage[startRow:startRow + subregionHeight, startCol:startCol + subregionWidth]
            
            # Call getOneRegionLBPFeatures() to get the one subimage's histogram. 
            hist = getOneRegionLBPFeatures(subRegion)
            
            # Append this histogram to your list of all histograms.
            allHists.append(hist)
    
    # Convert your list of histograms to an np.array()        
    allHists = np.array(allHists)
    
    # reshape so that it is a flat array:
    allHists = np.reshape(allHists, (allHists.shape[0]*allHists.shape[1],))
    
    return allHists