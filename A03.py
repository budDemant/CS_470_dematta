


import cv2
from skimage import slic
import numpy as np

def find_WBC(image):
    #(Step 1) Get superpixel groups. Assume the image with superpixel indices is given by segments.
    segments = slic(image, nSegments=100, startLabel=0)
    # number of groups found can then be extracted using:
    cnt = len(np.unique(segments))
    
    #(Step 2) Compute mean color per superpixel.
    groupMeans = np.zeros((cnt, 3), dtype="float32")
    for specificGroup in range(cnt):
        maskImage = np.where(segments == specificGroup, 255, 0).astype("uint8") 
        maskImage = np.expand_dims(maskImage)
        groupMeans[specificGroup] = cv2.mean(image, mask=maskImage)[0:3]
        
    
    
    