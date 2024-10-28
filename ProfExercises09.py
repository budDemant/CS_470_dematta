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
import skimage
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

def cluster_by_color(image, cluster_cnt, target_color):
    print("ORIGINAL:", image.shape)
    color_list = np.reshape(image, (-1, 3))
    print("NEW COLOR LIST SHAPE:", color_list.shape)
    color_list = color_list.astype("float32")
    _, label_list, center_list = cv2.kmeans(color_list, K=cluster_cnt,
                                            bestLabels=None,
                                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                                                      300, 1e-6), 
                                            attempts=3,
                                            flags=cv2.KMEANS_RANDOM_CENTERS)
    print("LABEL LIST:", label_list.shape)
    print("CENTER LIST:", center_list.shape)
    print("CENTERS:")
    print(center_list)
    
    target_color = np.array(target_color).astype("float32")
    target_color = np.expand_dims(target_color, axis=0)
    print("TARGET COLOR:", target_color.shape)
    dist_list = center_list - target_color
    dist_list = dist_list * dist_list
    dist_list = np.sum(dist_list, axis=1)
    dist_list = np.sqrt(dist_list)
    
    print("DIST LIST:", dist_list.shape)
    print(dist_list)
    
    chosen_one = np.argmin(dist_list)
    print("COLOR OF CHOICE:", chosen_one, center_list[chosen_one])
    
    new_centers = np.zeros_like(center_list)
    new_centers[chosen_one] = (255,255,255)
    
    print("NEW CENTERS:")
    print(new_centers)
    
    new_image = new_centers[label_list.flatten()]
    
    print("NEW IMAGE:", new_image.shape)
    new_image = np.reshape(new_image, image.shape)
    new_image = cv2.convertScaleAbs(new_image)
    
    return new_image   
    

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
            
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh_image = do_threshold(grayscale, thresh_type, value)
        
        cnt, region_image = cv2.connectedComponents(thresh_image, 
                                                    connectivity=8, 
                                                    ltype=cv2.CV_32S)
        
        print("COUNT REGIONS:", cnt)
        
        draw_image = np.copy(image)
        for blob_index in range(1, cnt):
            print(blob_index)
            coords = np.where(region_image == blob_index)
            #print(coords)
            #exit(1)
            ymin = np.amin(coords[0])
            ymax = np.amax(coords[0])
            
            xmin = np.amin(coords[1])
            xmax = np.amax(coords[1])
            
            width = xmax - xmin
            height = ymax - ymin
            
            area = width*height
            
            if area > 10:            
                cv2.rectangle(draw_image, (xmin, ymin), (xmax, ymax), (0,0,255), 3)
                
        segments = skimage.segmentation.slic(image, n_segments=100, sigma=5)
        super_image = skimage.segmentation.mark_boundaries(image, segments)
        
        cluster_color_image = cluster_by_color(image, cluster_cnt=6, target_color=(0,255,0))
               
        # Show our image (with the filename as the window title)
        windowTitle = "PYTHON: " + filename
        cv2.imshow(windowTitle, grayscale)
        cv2.imshow("THRESH", thresh_image)
        cv2.imshow("REGION", draw_image)
        cv2.imshow("SUPERPIXEL", super_image)
        cv2.imshow("CLUSTER", cluster_color_image)

        # Wait for a keystroke to close the window
        cv2.waitKey(-1)

        # Cleanup this window
        cv2.destroyAllWindows()

# The main function
if __name__ == "__main__": 
    main()
    