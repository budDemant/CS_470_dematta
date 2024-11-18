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

class THRESH_TYPE(Enum):
    MANUAL = 0
    AUTOMATIC = 1
    OTSU = 2
    ADAPTIVE = 3
    
def do_threshold(image, thresh_type, val):
    output = np.copy(image)
    
    if thresh_type == THRESH_TYPE.MANUAL:
        _, output = cv2.threshold(output, thresh=val, maxval=255, type=cv2.THRESH_BINARY)
    elif thresh_type == THRESH_TYPE.AUTOMATIC:
        T = cv2.mean(image)[0]
        old_T = 800
        diff_T = abs(T - old_T)
        
        while diff_T > 1:    
            old_T = T    
            _, output = cv2.threshold(image, thresh=T, maxval=255, type=cv2.THRESH_BINARY)
        
            fore_mean = cv2.mean(image, output)[0]
            back_mean = cv2.mean(image, 255 - output)[0]
            
            T = (fore_mean + back_mean)/2
            diff_T = abs(T - old_T)
        
        _, output = cv2.threshold(image, thresh=T, maxval=255, type=cv2.THRESH_BINARY)
        print("T:", T) 
    elif thresh_type == THRESH_TYPE.OTSU:
        val, output = cv2.threshold(image, thresh=val, maxval=255, type=cv2.THRESH_OTSU)   
        print("OTSU THRESH:", val)  
    elif thresh_type == THRESH_TYPE.ADAPTIVE:
        output = cv2.adaptiveThreshold(image, maxValue=255,
                                       adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                       thresholdType=cv2.THRESH_BINARY,
                                       blockSize=val, C=0)       
    
    return output

###############################################################################
# MAIN
###############################################################################

def main():    
    
    for item in list(THRESH_TYPE):
        print(item.value, "-", item.name)
    thresh_type = THRESH_TYPE(int(input("Enter choice: ")))
    print("Chosen One:", thresh_type.name)
    
    value = 7 # 127
        
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
        ESC_KEY = 27
        while key != ESC_KEY:
            # Get next frame from camera
            _, image = camera.read()
            
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh_image = do_threshold(grayscale, thresh_type, value)
            
            # Show the image
            cv2.imshow(windowName, image)
            cv2.imshow("THRESHOLDED", thresh_image)

            # Wait 30 milliseconds, and grab any key presses
            key = cv2.waitKey(30)
            
            if key == ord('a'):
                value -= 2
                print("THRESHOLD:", value)
            if key == ord('s'):
                value += 2
                print("THRESHOLD:", value)

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
        image = cv2.imread(filename) 
        
        # Check if data is invalid
        if image is None:
            print("ERROR: Could not open or find the image!")
            exit(1)
            
        gray = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2GRAY)
        # corner_image = cv2.cornerHarris(gray, 3, 3, k=1)
        corner_list = cv2.goodFeaturesToTrack(gray, maxCorners=200,qualityLevel=0.1, 
                                               minDistance=2, blockSize=3, useHarrisDetector=False)
        
        # corner_image = np.abs(corner_image)
        # corner_image *= 10.0
        
        print(corner_list.shape)
        print(corner_list.dtype)
        print(np.amin(corner_list), np.amax(corner_list))
        
        # corner_image = np.zeros_like(image)
        corner_image = np.copy(image)
        for corner in corner_image:
            print(corner.shape)
            x = int(corner[0][0])
            y = int(corner[0][1])
            cv2.circle(corner_image, (x,y), 5, (0,0,255), -1)

        # Show our image (with the filename as the window title)
        windowTitle = "PYTHON: " + filename
        cv2.imshow(windowTitle, image)
        cv2.imshow("CORNER IMAGE", corner_image)

        # Wait for a keystroke to close the window
        cv2.waitKey(-1)

        # Cleanup this window
        cv2.destroyAllWindows()

# The main function
if __name__ == "__main__": 
    main()
    