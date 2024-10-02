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
import math as m

def draw_polar_line(image, line):
    # (p, theta)
    p, theta = line
    
    # x cos theta + y sin theta = p
    # Ax + By = p
    
    A = m.cos(theta)
    B = m.sin(theta)
    
    if abs(B) >= 1e-6:
        # Ax + By = p
        # Start: x = 0
        #    By = p
        #    y = p/B
        
        # End: x = width
        #    A*width + By = p
        #    By = p - A*width
        #    y = (p - A*width)/B
        
        start_y = p / B
        end_y = (p - A*image.shape[1])/B
        
        start_y = int(start_y)
        end_y = int(end_y)
        
        cv2.line(image, (0, start_y), (image.shape[1], end_y), (0,0,255), 3)
    else:
        # Ax + By = p
        # Start: y = 0
        #     Ax = p
        #     x = p/A
        
        # End: y = height
        #     Ax + B*height = p
        #     Ax = p - B*height
        #     x = (p - B*height)/A
        
        start_x = p/A
        end_x = (p - B*image.shape[0])/A
        
        start_x = int(start_x)
        end_x = int(end_x)
        
        cv2.line(image, (start_x, 0), (end_x, image.shape[0]), (0,0,255), 3)
        

###############################################################################
# MAIN
###############################################################################

def main():        
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
        
        high_T = 200
        low_T = 100
        
        hthresh = 100
        
        while key != ESC_KEY:
            # Get next frame from camera
            _, image = camera.read()
            
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            canny = cv2.Canny(grayscale, low_T, high_T)
            
            lines = cv2.HoughLines(canny, rho=1, theta=m.pi/180.0, threshold=hthresh)
            #print(lines)
            if lines is not None:
                for line in lines:
                    draw_polar_line(image, line[0])
            
            # Show the image
            cv2.imshow(windowName, image)
            cv2.imshow("EDGES", canny)

            # Wait 30 milliseconds, and grab any key presses
            key = cv2.waitKey(30)
            
            if key == ord('e'): hthresh += 1
            if key == ord('d'): hthresh -= 1
                        
            if key == ord('q'): high_T += 1
            if key == ord('a'): high_T -= 1
            if key == ord('w'): low_T += 1
            if key == ord('s'): low_T -= 1
            if lines is not None: print("Line cnt:", len(lines))
            print(low_T, high_T, hthresh)

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
    