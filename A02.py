import numpy as np
import cv2

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
            
    
def apply_filter(image, kernel, alpha=1.0, beta=0.0, convert_uint8=True):
    #Cast both the image and kernel to "float64"
    image, kernel = np.float64(image,kernel)
    
    # Rotate the kernel 180 degrees so that you are performing convolution: 
    kernel = cv2.flip(kernel, -1)
    
    # padding is height and width of the kernel integer-divided by 2
    padTop = kernel.shape[0] // 2
    padBottom = kernel.shape[0] // 2
    padLeft = kernel.shape[1] // 2
    padRight = kernel.shape[1] // 2
    
    # create a padded image using borderType=cv2.BORDER_CONSTANT and zero-padding
    imagePad = cv2.copyMakeBorder(image, top=padTop, bottom=padBottom,
                left=padLeft, right=padRight, borderType=cv2.BORDER_CONSTANT, value=0)
    
    # Create a FLOATING-POINT (float64) numpy array to hold our output image
    output = np.zeros(image.shape, dtype=np.float64)
    
    # Grab the subimage from the PADDED image
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            # np.array[row_start : row_end, col_start : col_end] (this comment is for me to remember)
            # if image is 3x3 and filter is 3x3, will result in 9 subimages
            subImage = imagePad[row : (row + kernel.shape[0]), col : (col + kernel.shape[1])]
    
    # Multiply the subimage by the kernel       
    filtervals = subImage * kernel
    
    # Get the sum of these values: 
    value = np.sum(filtervals) 
    
    output[row,col] = value
    
    if convert_uint8:
        output = cv2.convertScaleAbs(output, alpha=alpha, beta=beta)

        
    return output