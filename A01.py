import numpy as np
import cv2

def create_unnormalized_hist(image):
    # Create a numpy array of zeros with shape (256,) and dtype float64
    hist = np.zeros(256, dtype=np.float64)
    # Iterate through each pixel in the image (rows and columns)
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            pixel = image[i, j]  
            hist[pixel] += 1  # to get intensity values
    
    return hist


def normalize_hist(hist):
    # total number of pixels in histogram
    histSum = np.sum(hist)
    # Create a new array to hold the normalized histogram
    nhist = np.zeros(256, dtype=np.float64)
    
    # Iterate through each pixel intensity, dividing each by histSum to normalize
    for i in range(len(hist)):
        nhist[i] = hist[i] / histSum  
        
    return nhist 


def create_cdf(nhist):
    # cdf is  a numpy array of shape (256,) and type "float64"
    cdf = np.zeros(256, dtype=np.float64)
    # cSum is cumulative sum
    cSum = 0
    for i in range(len(nhist)):
        cSum += nhist[i] 
        cdf[i] = cSum
        
    return cdf

def get_hist_equalize_transform(image, do_stretching):
    # Use create_unnormalized_hist to calculate the unnormalized histogram 
    hist = create_unnormalized_hist(image)
    # Normalize your histogram
    nhist = normalize_hist(hist)
    # Make the CDF
    cdf = create_cdf(nhist)
    
    int_transform = np.zeros(256, dtype=np.float64)
    
    # stretching formula is newImage = ((oldPixelIntensity - minIntensity) * 255) / maxIntensity - minIntensity
    if do_stretching:
        cdfMin = cdf.min()
        cdfMax = cdf.max()
        cdf = (cdf - cdfMin) * 255 / (cdfMax - cdfMin)
    else:
        cdf *= 255
        #cdf = cdf * 255
        
    int_transform = cv2.convertScaleAbs(cdf)[:,0]
    
    return int_transform
        
        
        
        
    
    


    


    