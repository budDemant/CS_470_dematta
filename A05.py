import torchvision.transforms.v2 as v2
import torch
import cv2

# Returns a list of the names of all combinations you will be testing.
def get_approach_names():
    return ["BasicCNN", "EnhancedCNN"]


# Given the approach_name, return a text description of what makes this approach distinct. 

def get_approach_description(approach_name):
    descriptions = {
        "BasicCNN": "A simple convolutional neural network with minimal layers and no additional features.",
        "EnhancedCNN": "A convolutional neural network with added batch normalization and dropout for better generalization."
    }

    return descriptions.get(approach_name)
