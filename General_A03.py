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

import os
import sys
import cv2
import pandas as pd
import numpy as np
import shutil
from enum import Enum

BCCD_DATA_DIR = os.path.join(".", "data", "BCCD")
BCCD_DATA_CSV = "data.csv"

base_dir = "assign03"

class BCCD_TYPES(Enum):   
    RBC = "RBC"  
    WBC = "WBC"
    PLATE = "Platelets"   

###############################################################################
# LOADS UP BCCD DATA INTO TF.DATASETS
###############################################################################
def load_and_prepare_BCCD_data():
    # Assumes that Prepare_A03 has already been run
    
    # Load up CSV file
    data = pd.read_csv(os.path.join(BCCD_DATA_DIR, BCCD_DATA_CSV))
    
    # Load up split files
    def load_split_files(filename):
        all_files = []
        with open(os.path.join(BCCD_DATA_DIR, filename), "r") as f:
            all_files = f.readlines()
            
        for i in range(len(all_files)):
            all_files[i] = all_files[i].strip() + ".jpg"
            
        return all_files
    
    train_files = load_split_files("train.txt")
    test_files = load_split_files("test.txt")
    
    def load_objects(df):
        object_data = {
            "bbox": [],
            "label": []
        }
        # For each row
        for index, row in df.iterrows():
            # Get the bounding box
            # y1, x1, y2, x2
            box = np.array([row["ymin"], row["xmin"], row["ymax"], row["xmax"]], dtype="int")
            # Get label
            label = row["cell_type"]            
            # Add to lists
            object_data["bbox"].append(box)
            object_data["label"].append(label)
            
        return object_data        
            
    def prepare_dataset(filelist):
        all_data = []
        # For each file
        for filename in filelist:    
            # Load image
            image = cv2.imread(os.path.join(BCCD_DATA_DIR, "images", filename))
            # Load up bounding box and label info            
            objects = load_objects(data[data["filename"] == filename])
            # Add to list
            all_data.append((image, objects))
        return all_data
        
    # Create "datasets" for training and testing
    train_data = prepare_dataset(train_files)    
    test_data = prepare_dataset(test_files)
    
    # Number of items
    print("Number of training images:", len(train_data))
    print("Number of testing images:", len(test_data))  

    return train_data, test_data

###############################################################################
# EXTRACTS BOUNDING BOXES FOR PARTICULAR CELL TYPE
###############################################################################
def unpack_one_cell_type_only(objects, image_shape, cell_type):
    # Get the bounding boxes and labels from the dictionary
    bboxes = objects['bbox']
    labels = objects['label']
    
    # Create an empty list to hold the bounding boxes we want to keep
    cell_boxes = []

    # For each entry...
    for i in range(len(bboxes)):
        # Get one bounding box and label
        bb = bboxes[i]     
        label = labels[i]   
                
        # If this is the cell we're looking for...        
        if label == cell_type:            
            # Add to our list
            cell_boxes.append(bb)

    return cell_boxes

###############################################################################
# COMPUTES INTERSECTION OVER UNION FOR BOUNDING BOXES
###############################################################################
def compute_one_IOU(predicted, ground):
    # Bounding box stored as (y1, x1, y2, x2)   
    def compute_area(left, right, top, bottom):
        width = right - left
        height = bottom - top
        width = max(0, width)
        height = max(0, height)        
        area = width * height
        return area

    # Get intersection
    left = max(predicted[1], ground[1])
    right = min(predicted[3], ground[3])
    top = max(predicted[0], ground[0])
    bottom = min(predicted[2], ground[2])    
    intersection = compute_area(left, right, top, bottom)
    
    # Get union
    area_pred = compute_area(predicted[1], predicted[3], predicted[0], predicted[2])
    area_ground = compute_area(ground[1], ground[3], ground[0], ground[2])
    union = area_pred + area_ground - intersection     

    # Get IOU
    iou = intersection / union
            
    return iou
    
def compute_IOU(all_predicted, all_ground):
    # For each ground box, find the nearest match
    all_IOU = 0.0
    for ground in all_ground:
        best_IOU = 0.0
        for predicted in all_predicted:
            one_IOU = compute_one_IOU(predicted, ground)
            if one_IOU < 0 or one_IOU > 1.0:
                print(one_IOU)   
                exit(1)         
            best_IOU = max(best_IOU, one_IOU)
        all_IOU += best_IOU
    
    # Average it out
    if len(all_ground) > 0:
        all_IOU /= len(all_ground)

    return all_IOU

###############################################################################
# DRAWS BOUNDING BOXES ON IMAGE
###############################################################################
def draw_bounding_boxes(image, bounding_boxes, color):
    # For each box...
    for bb in bounding_boxes:
        cv2.rectangle(image, (bb[1], bb[0]), (bb[3], bb[2]), color, thickness=2)

###############################################################################
# PREDICTS BOUNDING BOXES ON DATASET AND COMPUTES METRICS
###############################################################################
def predict_dataset(dataset, prefix, out_dir, cell_type, find_cell_func):
    # Prepare metric dictionary
    metrics = {}
    metrics["Accuracy"] = 0.0
    metrics["IOU"] = 0.0

    # Get total count
    total_cnt = len(dataset)

    # Print starting
    print("Starting on", prefix, "(" + str(total_cnt) + " samples total)")

    # For each datapoint...
    image_index = 0
    for data_pack in dataset:
        # Each item is a tuple, so separate data into image and objects
        image = data_pack[0]
        objects = data_pack[1]

        # Objects is a dictionary, so we'll unpack the bounding boxes
        # for specific cells only        
        true_bounding_boxes = unpack_one_cell_type_only(objects, image.shape, cell_type)
        true_cell_count = len(true_bounding_boxes)

        # Calculate bounding boxes using your approach
        pred_bounding_boxes = find_cell_func(image)
        pred_cell_count = len(pred_bounding_boxes)

        # Draw bounding boxes on image
        draw_bounding_boxes(image, true_bounding_boxes, (0,0,0))
        draw_bounding_boxes(image, pred_bounding_boxes, (0,255,0))

        # Show images (DEBUG)                
        #cv2.imshow("IMAGE", image)        
        #cv2.waitKey(-1)

        # Save image
        cv2.imwrite(out_dir + "/%s_%03d.png" % (prefix, image_index), image)

        # Is this correct in terms of the number of cells predicted?
        if true_cell_count == pred_cell_count:
            metrics["Accuracy"] += 1.0

        # Compute IOU
        metrics["IOU"] += compute_IOU(pred_bounding_boxes, true_bounding_boxes)

        # Increment index
        image_index += 1

        # Print progress
        percent = 100.0*image_index / total_cnt
        print("%.1f%% complete...       " % percent, end="\r", flush=True)

    # Print complete
    print(prefix, "complete!")

    # Average out metrics
    metrics["Accuracy"] /= image_index
    metrics["IOU"] /= image_index

    # Return metrics
    return metrics

###############################################################################
# PRINTS METRICS (to STDOUT or file)
###############################################################################
def print_metrics(train_metrics, test_metrics, stream=sys.stdout):
    print("TRAINING:", file=stream)
    for key in train_metrics:
        print("\t", key, "=", train_metrics[key], file=stream)
  
    print("TESTING:", file=stream)
    for key in test_metrics:
        print("\t", key, "=", test_metrics[key], file=stream)
