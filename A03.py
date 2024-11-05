


import cv2
from skimage import slic
import numpy as np

def find_WBC(image):
    #(Step 1) Get superpixel groups. Assume the image with superpixel indices is given by segments.
    segments = slic(image, nSegments=100, startLabel=0)
    
    # number of groups found can then be extracted using:
    cnt = len(np.unique(segments))
    
    
    #(Step 2) Compute mean color per superpixel.
    
    # Create a numpy array to hold the mean color per superpixel: 
    groupMeans = np.zeros((cnt, 3), dtype="float32")
    
    # Loop through each superpixel index
    for specificGroup in range(cnt):
        maskImage = np.where(segments == specificGroup, 255, 0).astype("uint8")
        
        # Add the channel dimension back in to the mask using np.expand_dims().
        maskImage = np.expand_dims(maskImage, axis=2)
        
        # slice the result: 
        groupMeans[specificGroup] = cv2.mean(image, mask=maskImage)[0:3]
    
        
    # (Step 3) Use K-means on GROUP mean colors to group them into 4 color groups. 
    
    # stores mean colors 
    groupMeans = np.float32(groupMeans)
    
    k = 4
    # ret = not used, bestLabels = list of which superpixel mean colors map to which kmeans group, centers = kmeans group center values
    # apply kmeans clustering
    _, bestLabels, centers = cv2.kmeans(groupMeans, k, None, (cv2.TERM_CRITERIA_EPS, 
                                    cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), 10, 
                                    cv2.KMEANS_RANDOM_CENTERS)
    
    
    # (Step 4) Find the k-means group with mean closest to: 
    targetColor = np.array([255, 0, 0], dtype="float32")
    
    #  Euclidean distance color segmentation
    distances = np.sqrt(np.sum((centers - targetColor) ** 2, axis=1))
    wbcGroup = np.argmin(distances)
    
    # (Step 5) Set that k-means group to white and the rest to black.
    
    # binary mask
    centers = np.zeros_like(centers)
    
    centers[wbcGroup] = [255, 255, 255]
    
    
    
    
    
    