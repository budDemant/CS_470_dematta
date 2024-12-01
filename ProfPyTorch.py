import torch
import torchvision.datasets as datasets
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as v2
import cv2
import numpy as np
import os

class SimpleNetwork(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        flat_input_size = np.prod(input_shape)
        self.flatten = nn.Flatten()
        self.layer_list = nn.Sequential(
            nn.Linear(flat_input_size, 32),
            nn.Sigmoid(),
            nn.Linear(32, 10)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.layer_list(x)
        return logits

def main():
    
    stdTransform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True)
    ])
    
    train_data = datasets.MNIST(root="data", train=True, download=True, transform=stdTransform)
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=stdTransform)
    
    train_ds = DataLoader(train_data, batch_size=64, shuffle=True)
    test_ds = DataLoader(test_data, batch_size=64, shuffle=False)
    
    train_it = iter(train_ds)
    
    def show_image(X, y):
        # Grab first sample
        X = X[0]
        y = y[0]
        # Turn to numpy
        X = X.detach().cpu().numpy()
        # Convert to OpenCV
        X = np.transpose(X, [1,2,0])
        # Resize to display
        X = cv2.resize(X, dsize=None,  fx=5.0, fy=5.0)
        # Show
        print("Label:", y)
        cv2.imshow("IMAGE", X)
        cv2.waitKey(-1)
        cv2.destroyAllWindows()
    
    '''
    for _ in range(5):
        sample = next(train_it)
        X,y = sample
        print("X:", X.shape)
        print("y:", y.shape)
        show_image(X, y)
    '''
    
    device = "cuda"
    
    input_shape = next(iter(train_ds))[0][0].shape
    
    model = SimpleNetwork(input_shape)
    model = model.to(device)
    
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(model)
    
    def train_one_epoch(model, train_ds, device, loss_func, opt):
        model.train()
        num_batches = len(train_ds)
        for batch_index, (X,y)  in enumerate(train_ds):
            X = X.to(device)
            y = y.to(device)
            
            pred = model(X)
            
            y = torch.nn.functional.one_hot(y, num_classes=10).to(torch.float32)
            
            #print(pred.shape, y.shape)
            
            loss = loss_func(pred, y)
            
            loss.backward()
            opt.step()
            opt.zero_grad()
            
            if batch_index % 100 == 0:
                print("Batch", (batch_index+1), "of", num_batches, ": Loss =", loss.item())
            
            
    epoch_cnt = 10
    
    for epoch in range(epoch_cnt):
        print("** EPOCH", (epoch+1), "**************")
        train_one_epoch(model, train_ds, device, loss_func, optimizer)
    
        
        
    


if __name__ == "__main__":
    main()
    