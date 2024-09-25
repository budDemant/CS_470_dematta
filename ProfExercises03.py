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

class IntTransform(Enum):
    ORIGINAL = 0
    NEGATIVE = 1
    SLICE = 2
    GAMMA = 3
    HISTEQ = 4


def do_int_transform(image, chosen_T, 
                     slice_low=100, slice_high=200,
                     gamma=0.4):
    output = np.copy(image)
    
    def gamma_func(x):
        x /= 255.0
        x = pow(x, gamma)
        x *= 255.0
        return x
    
    if chosen_T == IntTransform.ORIGINAL:
        # Do nothing
        return output
    elif chosen_T == IntTransform.NEGATIVE:
        output = 255 - output  
    elif chosen_T == IntTransform.SLICE:
        for row in range(output.shape[0]):
            for col in range(output.shape[1]):
                pixel = image[row,col]
                if pixel >= slice_low and pixel <= slice_high:
                    pixel = 255
                else:
                    pixel = 0
                output[row,col] = pixel
    elif chosen_T == IntTransform.GAMMA:
        vector_gamma = np.vectorize(gamma_func)
        output = output.astype("float64")
        output = vector_gamma(output)
        output = cv2.convertScaleAbs(output)
    elif chosen_T == IntTransform.HISTEQ:
        output = cv2.equalizeHist(output)
                   
    return output

###############################################################################
# MAIN
###############################################################################

def main():    
        
    for item in list(IntTransform):
        print(item.value, "-", item.name)
    chosen_T = IntTransform(int(input("Enter choice: ")))
    print("Chosen One:", chosen_T.name)
    
    # Parameters
    gamma = 0.4
    slice_low=100
    slice_high=200
             
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
            _, frame = camera.read()
            
            # Convert to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Transform            
            processed = do_int_transform(frame, chosen_T, 
                                         gamma=gamma,
                                         slice_low=slice_low,
                                         slice_high=slice_high)
            
            # Show the images
            cv2.imshow(windowName, frame)
            cv2.imshow("Processed", processed)

            # Wait 30 milliseconds, and grab any key presses
            key = cv2.waitKey(30)
            
            if key == ord('a'):
                gamma /= 2.0
                print("GAMMA:", gamma)
            elif key == ord('d'):
                gamma *= 2.0
                print("GAMMA:", gamma)
                
            if key == ord('q'):
                slice_low -= 20
                print("SLICE RANGE:", slice_low, "to", slice_high)
            if key == ord('w'):
                slice_low += 20
                print("SLICE RANGE:", slice_low, "to", slice_high)
                
            if key == ord('e'):
                slice_high -= 20
                print("SLICE RANGE:", slice_low, "to", slice_high)
            if key == ord('r'):
                slice_high += 20
                print("SLICE RANGE:", slice_low, "to", slice_high)


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
                
        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Transform        
        processed = do_int_transform(image, chosen_T)
        
        # Show the images
        cv2.imshow(windowTitle, image)
        cv2.imshow("Processed", processed)

        # Wait for a keystroke to close the window
        cv2.waitKey(-1)

        # Cleanup this window
        cv2.destroyAllWindows()

# The main function
if __name__ == "__main__": 
    main()
    
    