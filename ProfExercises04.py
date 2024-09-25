# MIT LICENSE
#
# Copyright 2024 Michael J. Reale
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

###############################################################################
# IMPORTS
###############################################################################

import sys
import numpy as np
import torch
import cv2
import pandas
import sklearn
from enum import Enum

class FilterType(Enum):
    BOX = 0
    GAUSS = 1    
    MEDIAN = 2
    LAPLACIAN = 3
    SHARPEN = 4
    
def filter_image(image, filter_type, filter_width, filter_height):
    output = np.copy(image)
    
    if filter_type == FilterType.BOX:
        output = cv2.blur(output, (filter_width, filter_height))
    elif filter_type == FilterType.GAUSS:
        output = cv2.GaussianBlur(output, (filter_width, filter_height), sigmaX=0)
    elif filter_type == FilterType.MEDIAN:
        output = cv2.medianBlur(output, filter_width)
    elif filter_type == FilterType.LAPLACIAN:
        laplace = cv2.Laplacian(output, cv2.CV_64F, ksize=filter_width, scale=0.25)
        output = cv2.convertScaleAbs(laplace, alpha=0.5, beta=127)
    elif filter_type == FilterType.SHARPEN:
        laplace = cv2.Laplacian(output, cv2.CV_64F, ksize=filter_width, scale=0.25)
        fimage = image.astype("float64")
        fimage -= laplace
        output = cv2.convertScaleAbs(fimage)
        
    return output

###############################################################################
# MAIN
###############################################################################

def main():   
    
    for item in list(FilterType):
        print(item.value, "-", item.name)
    filter_type = FilterType(int(input("Enter choice: ")))
    print("Chosen One:", filter_type.name)
    
    filter_width = int(input("Enter filter width: "))
    filter_height = int(input("Enter filter height: "))
         
    ###############################################################################
    # PYTORCH
    ###############################################################################
    
    b = torch.rand(5,3)
    print(b)
    print("Do you have Torch CUDA?:", torch.cuda.is_available())
    
    ###############################################################################
    # PRINT OUT VERSIONS
    ###############################################################################

    print("Torch:", torch.__version__)
    print("Numpy:", np.__version__)
    print("OpenCV:", cv2.__version__)
    print("Pandas:", pandas.__version__)
    print("Scikit-Learn:", sklearn.__version__)
        
    ###############################################################################
    # OPENCV
    ###############################################################################
    if len(sys.argv) <= 1:
        # Webcam
        print("Opening the webcam...")

        # Linux/Mac (or native Windows) with direct webcam connection
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) # CAP_DSHOW recommended on Windows 
        # WSL: Use Yawcam to stream webcam on webserver
        # https://www.yawcam.com/download.php
        # Get local IP address and replace
        #IP_ADDRESS = "192.168.0.7"    
        #camera = cv2.VideoCapture("http://" + IP_ADDRESS + ":8081/video.mjpg")
        
        # Did we get it?
        if not camera.isOpened():
            print("ERROR: Cannot open the camera!")
            exit(1)

        # Create window ahead of time
        windowName = "Webcam"
        cv2.namedWindow(windowName)

        # While not closed...
        key = -1
        while key == -1:
            # Get next frame from camera
            _, frame = camera.read()
            
            gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            filtered = filter_image(gray_scale, filter_type, filter_width, filter_height)            
            
            # Show the image
            cv2.imshow(windowName, gray_scale)
            cv2.imshow("FILTERED", filtered)

            # Wait 30 milliseconds, and grab any key presses
            key = cv2.waitKey(30)

        # Release the camera and destroy the window
        camera.release()
        cv2.destroyAllWindows()

        # Close down...
        print("Closing application...")

    else:
        # Trying to load image from argument

        # Get filename
        filename = sys.argv[1]

        # Load image
        print("Loading image:", filename)
        image = cv2.imread(filename) # For grayscale: cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        # Check if data is invalid
        if image is None:
            print("ERROR: Could not open or find the image!")
            exit(1)

        # Show our image (with the filename as the window title)
        windowTitle = "PYTHON: " + filename
        cv2.imshow(windowTitle, image)

        # Wait for a keystroke to close the window
        cv2.waitKey(-1)

        # Cleanup this window
        cv2.destroyAllWindows()

# The main function
if __name__ == "__main__": 
    main()
    