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

counter = 0
MAX_COUNTER = 30
last_image = None

def make_ghost_image(image):
    global counter
    global last_image
    
    image = image.astype(np.float64)
    
    
    if last_image is None or counter >= MAX_COUNTER:
        counter = 0
        last_image = np.copy(image)
    
    ghost_image = cv2.convertScaleAbs(image*0.5 + last_image*0.5)
    
    counter +=1
    
    return ghost_image
        

def gray_slice(image, min_val = 100, max_val = 200):
    output = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output = np.where(output <= min_val, min_val, output)
    output = np.where(output >= max_val, max_val, output)
    return output

def my_eyes(image, scale=0.1, upinter=cv2.INTER_NEAREST):
    output = np.copy(image)
    output = cv2.resize(output, dsize=(0,0), fx=scale, fy=scale)
    inv_scale = 1.0/scale
    output = cv2.resize(output, dsize=(0,0), fx=inv_scale, fy=inv_scale, interpolation=cv2.INTER_NEAREST)
    return output

    

###############################################################################
# MAIN
###############################################################################

def main():
    
    
    '''
    image = np.zeros((600, 800, 3), dtype="uint8")
    
    
    #start at row 50 and stop right before row 100
    image [50:100,20:400,0:3] = 255
    
    #if you do [:3], 0 is implied (indicating the start)
    #if you do [3:] it goes to the end
    #if you do [:] it goes from start to end
    
    #if you want to get more particular with color do [0,0] = (128, 255, 0)
    #(128, 255, 0) = (blue, green, red)
    
    cv2.imshow("My Image", image)
    #waiting for user to move the window, resize it, hit a key, etc.
    cv2.waitKey(-1)
    cv2.destroyAllWindows()
    '''
    
    
        
    
    
    
       
    ###############################################################################
    # PYTORCH
    ###############################################################################
    
    # b = torch.rand(5,3)
    # print(b)
    # print("Torch CUDA?:", torch.cuda.is_available())
    
    ###############################################################################
    # PRINT OUT VERSIONS
    ###############################################################################

    # print("Torch:", torch.__version__)
    # print("Numpy:", np.__version__)
    # print("OpenCV:", cv2.__version__)
    # print("Pandas:", pandas.__version__)
    # print("Scikit-Learn:", sklearn.__version__)
        
    ###############################################################################
    # OPENCV
    ###############################################################################
    if len(sys.argv) <= 1:
        # Webcam
        print("Opening webcam...")

        # Linux/Mac (or native Windows) with direct webcam connection
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) # CAP_DSHOW recommended on Windows 
        # WSL: Use Yawcam to stream webcam on webserver
        # https://www.yawcam.com/download.php
        # Get local IP address and replace
        #IP_ADDRESS = "192.168.0.7"    
        #camera = cv2.VideoCapture("http://" + IP_ADDRESS + ":8081/video.mjpg")
        
        # Did we get it?
        if not camera.isOpened():
            print("ERROR: Cannot open camera!")
            exit(1)

        # Create window ahead of time
        windowName = "Webcam"
        cv2.namedWindow(windowName)

        # While not closed...
        key = -1
        while key == -1:
            # Get next frame from camera
            _, frame = camera.read()
            
            # Show the image
            cv2.imshow(windowName, frame)
            
            processed = gray_slice(frame)
            cv2.imshow("Gray slice", processed)
            
            horrors = my_eyes(frame, up_inter=cv2.INTER_LINEAR)
            cv2.imshow("THE HORROR", horrors)
            
            ghost_image = make_ghost_image(frame)
            cv2.imshow("GHOST", ghost_image)

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

if __name__ == "__main__": 
    main()