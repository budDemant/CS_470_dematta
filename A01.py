''' I apologize in advance for the mix of underscore and camelCase variables. I am accustomed to camelCasing
but it seems that most of the given code (from the instructions) uses underscores '''
# No problem.

import numpy as np
import cv2
import gradio as gr

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
    histSum = np.sum(hist)
    # didn't have this line before, was returning scalar oops
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


def do_histogram_equalize(image, do_stretching):
    # COPY your image → output
    outputImage = image.copy()
    # Get your transformation function
    transFunc = get_hist_equalize_transform(outputImage, do_stretching)
    
    # copied from unnormalized hist func bc it's the same logic
    height, width = image.shape
    # For each pixel in the image
    for i in range(height):
        for j in range(width): 
            pixelValue = image[i, j] # Get the value
            transValue = transFunc[pixelValue] # Use your transformation to get the new value
            outputImage[i, j] = transValue # Store it into the OUTPUT image
    
    return outputImage


# Gradio stuff
def intensity_callback(input_img, do_stretching): 
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY) 
    output_img = do_histogram_equalize(input_img, do_stretching)     
    return output_img 
     
def main(): 
    demo = gr.Interface(fn=intensity_callback,  
                        inputs=["image", "checkbox"], 
                        outputs=["image"]) 
    demo.launch()    
 
if __name__ == "__main__": 
    main()
            
            
    
    
        
        
        
    
    


    


    