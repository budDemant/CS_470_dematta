import torchvision.transforms.v2 as v2
import torch
import cv2

# for create_model()
import torch.nn as nn
import torch.nn.functional as F

# for train_model()
import torch.optim as optim
import torch.nn as nn

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
                v2.ToTensor(),
                v2.ConvertImageDtype(torch.float32)
            ])
        elif approach_name == "EnhancedCNN":
            # data augmentation
            return v2.Compose([
                v2.RandomHorizontalFlip(),
                v2.RandomCrop(32, padding=4),
                v2.ToTensor(),
                v2.ConvertImageDtype(torch.float32)
            ])
    else:
        # non-training transformation (both approaches)
        return v2.Compose([
            v2.ToTensor(),
            v2.ConvertImageDtype(torch.float32)
        ])


# Given the approach_name, return the preferred batch size.     
def get_batch_size(approach_name):
    
    batch_sizes = {
        "BasicCNN": 64,
        "EnhancedCNN": 32
    }

    return batch_sizes.get(approach_name, 64) # defaults to 64 if not recognized

# Taken from: https://pytorch.org/vision/0.9/transforms.html


def create_model(approach_name, class_cnt):
    if approach_name == "BasicCNN":
        class BasicCNN(nn.Module):
            def __init__(self, class_cnt):
                # convolutional layers (input channel, output channel 3x3 kernel)
                super(BasicCNN, self).__init__()
                self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                # connected layer
                self.fc1 = nn.Linear(64 * 16 * 16, 128)
                self.fc2 = nn.Linear(128, class_cnt)
                # max pooling layer with 2x2 kernel
                self.pool = nn.MaxPool2d(2, 2)
                
            def forward(self, x):
                # apply 1st layer and relu activation
                x = F.relu(self.conv1(x))
                # print(f"After conv1: {x.shape}")
                # apply max pool
                x = self.pool(F.relu(self.conv2(x)))
                # print(f"After conv2 and pool: {x.shape}")
                # flatten tensor
                x = x.view(x.size(0), -1)
                # print(f"After flattening: {x.shape}")
                x = F.relu(self.fc1(x))
                # output layer
                x = self.fc2(x)
                
                return x
            
    # Taken from: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
            
    
        return BasicCNN(class_cnt)
    
    elif approach_name == "EnhancedCNN":
        # 3 convolutional layers with batch normalization (applied to 3rd layer), dropout layer, and larger fully connected layers
        class EnhancedCNN(nn.Module):
            def __init__(self, class_cnt):
                super(EnhancedCNN, self).__init__()
                self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                # Batch norm applied after 3rd layer
                self.bn1 = nn.BatchNorm2d(128)
                self.fc1 = nn.Linear(128 * 4 * 4, 256)
                # dropout for regularization
                self.dropout = nn.Dropout(0.5)
                self.fc2 = nn.Linear(256, class_cnt)
                self.pool = nn.MaxPool2d(2, 2)
            
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = self.pool(F.relu(self.conv2(x)))
                x = self.pool(F.relu(self.bn1(self.conv3(x))))
                x = x.view(x.size(0), -1)
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                
                return x
        return EnhancedCNN(class_cnt)
            
            
            

def train_model(approach_name, model, device, train_dataloader, test_dataloader):
    
    # some pieces Taken from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    # and: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    
    model = model.to(device)
    
    # cross entropy loss required
    criterion = nn.CrossEntropyLoss()
    
    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # number of epochs
    epochs = 10
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # training phase
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            
            # zero the gradients
            optimizer.zero_grad()
            
            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # backwards pass
            loss.backward()
            
            #update weights
            optimizer.step()
            
            # Accumulate training loss
            train_loss += loss.item()
            
        train_loss /= len(train_dataloader)
        print (f"Training loss: {train_loss:.4f}")
        
        # Testing phase
        model.eval() # evaluation mode
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # forward pass
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                # calculate total and correct predictions
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        accuracy = 100 % correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
    
    return model
    
    
    