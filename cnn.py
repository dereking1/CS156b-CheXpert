import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplitlib.pyplot as plt
from PIL import Image
import torch
from torch import nn
import torch.optim as optim
from data import *

model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=2),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout(p=0.1),
    nn.Flatten(),
    nn.Linear(25 * 16, 100),
    nn.Linear(100, 14)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters())

def train(trainloader, n_epochs=10):
    training_loss_history = np.zeros([n_epochs, 1])

    for epoch in range(n_epochs):
        print(f'Epoch {epoch+1}/{n_epochs}:', end='')
        train_total = 0
        train_correct = 0
        # train
        model.train()
        for i, data in enumerate(trainloader):
            image, labels = data
            optimizer.zero_grad()
            # forward pass
            output = model(images)
            # calculate categorical cross entropy loss
            loss = criterion(output, labels)
            # backward pass
            loss.backward()
            optimizer.step()
            
            # track training loss
            training_loss_history[epoch] += loss.item()

        training_loss_history[epoch] /= len(training_data_loader)
        print(f'\n\tloss: {training_loss_history[epoch,0]:0.4f}',end='')
    
    return training_loss_history

def main():
    X, Y = get_train_data()
    _, train_loader = trainload(X, Y)
    losses = train(train_loader)
    plt.figure()
    plt.plot(range(1,11),losses)
    plt.savefig('/home/ding/testtrain.png')

if __name__ == '__main__':
    main()
