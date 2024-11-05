


import cv2
import skimage
import numpy as np

def find_WBC(image):
    
    #(Step 1) Get superpixel groups. Assume the image with superpixel indices is given by segments.
    segments = skimage.segmentation.slic(image, n_segments=100, start_label=0)
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
    
    
    # (Step 6) Determine the new colors for each superpixel group
    # Convert centers to unsigned 8-bit 
    centers = centers.astype(np.uint8)
    # Get the new superpixel group colors:
    colorsPerClump = centers[bestLabels.flatten()] 
    
    
    # (Step 7) Recolor the superpixels with their new group colors (which are now just white and black)
    cellMask = colorsPerClump[segments]
    # Convert cell_mask to grayscale!
    cellMask = cv2.cvtColor(cellMask, cv2.COLOR_BGR2GRAY)
    
    
    # (Step 8) Use cv2.connectedComponents to get disjoint blobs from cell_mask.
    # retval = number of connection components / blobs found 
    # labels = image with labels marked
    retval, labels = cv2.connectedComponents(cellMask)
    
    
    # (Step 9) For each blob group (except 0, which is the background):
    boundingBoxes = []
    for i in range(1, retval):
        coords = np.where(labels == i)
        # "except 0"
        if coords[0].size > 0 and coords[1].size > 0:
            ymin, ymax = coords[0].min(), coords[0].max()
            xmin, xmax = coords[1].min(), coords[1].max()
            boundingBoxes.append((ymin, xmin, ymax, xmax))
    
    return boundingBoxes
            

    
    
    
    
    
    
    