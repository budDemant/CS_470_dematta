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


def get_data_transform(approach_name, training):
    if training:
        # minimal transformations
        if approach_name == "BasicCNN":
            return v2.Compose([
                v2.ToImageTensor(),
                v2.ConvertImageDtype(torch.float32)
            ])
        elif approach_name == "EnhancedCNN":
            # data augmentation
            return v2.Compose([
                v2.RandomHorizontalFlip(),
                v2.RandomCrop(32, padding=4),
                v2.ToImageTensor(),
                v2.ConvertImageDtype(torch.float32)
            ])
    else:
        # non-training transformation (both approaches)
        return v2.Compose([
            v2.ToImageTensor(),
            v2.ConvertImageDtype(torch.float32)
        ])


# Given the approach_name, return the preferred batch size.     
def get_batch_size(approach_name):
    
    batch_sizes = {
        "BasicCNN": 64,
        "EnhancedCNN": 32
    }

    return batch_sizes.get(approach_name, 64) # defaults to 64 if not recognized

