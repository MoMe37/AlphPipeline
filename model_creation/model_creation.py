import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset

import os
import numpy as np
from numpy import loadtxt

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=4,              
                out_channels=16,            
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(
                in_channels=16,              
                out_channels=32,            
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),   
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.conv3 = nn.Sequential(         
            nn.Conv2d(
                in_channels=32,              
                out_channels=64,            
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),   
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.flatten = nn.Flatten()
        self.seq = nn.Sequential(
            nn.Linear(64 * 4 * 4, 8),
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # flatten the output of conv2 to (batch_size, 64 * 16 * 16)
        # x = x.view(x.size(0), -1) 
        x = self.flatten(x)      
        output = self.seq(x)
        return output



def train(num_epochs, cnn, loaders):
    
    cnn.train()

    loss_values = []
    accuracy_test_values = []
    accuracy_train_values = []
        
    # Train the model
    total_step = len(loaders['train'])

    for epoch in range(num_epochs):
        loss_cum = 0
        for i, (images, labels) in enumerate(loaders['train']):
            
            # gives batch data, normalize x when iterate train_loader
            output = cnn(images)              
            loss = loss_func(output, labels)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()    

            # apply gradients             
            optimizer.step() 

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        loss_values.append(loss_cum/total_step)
        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, total_step, total_step, loss.item()))

        cnn.eval()

        accuracy_test = 0
        accuracy_train = 0

        with torch.no_grad():

            for images, labels in loaders['test']:
                test_output = cnn(images)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy_test += (pred_y == labels).sum().item() / float(labels.size(0))
            
            for images, labels in loaders['train']:
                train_output = cnn(images)
                pred_y = torch.max(train_output, 1)[1].data.squeeze()
                accuracy_train += (pred_y == labels).sum().item() / float(labels.size(0))

        cnn.train()

        accuracy_test /= len(loaders['test'])
        accuracy_train /= len(loaders['train'])
        accuracy_test_values.append(accuracy_test*100)
        accuracy_train_values.append(accuracy_train*100)

    x = []
    for i in range(num_epochs):
        x.append(i+1)
    plt.figure(figsize=(10,15))
    plt.subplot(3, 1, 1)
    plt.title('Loss function')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(x ,np.array(loss_values), 'r')
    plt.subplot(3, 1, 2)
    plt.title('Accuracy test data')
    plt.xlabel('Epoch')
    plt.ylabel('%')
    plt.plot(x ,np.array(accuracy_test_values), 'g')
    plt.subplot(3, 1, 3)
    plt.title('Accuracy train data')
    plt.xlabel('Epoch')
    plt.ylabel('%')
    plt.plot(x ,np.array(accuracy_train_values), 'b')



def test():
    # Test the model
    all_preds = torch.tensor([])
    all_labels = torch.tensor([])
    cnn.eval()
    accuracy = 0
    with torch.no_grad():
        for images, labels in loaders['test']:
            test_output = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy += (pred_y == labels).sum().item() / float(labels.size(0))

            all_preds = torch.cat((all_preds, pred_y),dim=0)
            all_labels = torch.cat((all_labels, labels),dim=0)

    accuracy /= len(loaders['test'])
    print('Test Accuracy of the model on the {} test : {:.2f}%'.format(len(test_dataset), (accuracy*100)))
    return all_labels, all_preds



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



if __name__ == "__main__":
    feature_train = []
    label_train = []
    feature_test = []
    label_test = []

    train_dataset = SampleDataset(feature_train, label_train, transform=nn.BatchNorm1d(64)) #Get dataset Enzo
    test_dataset = SampleDataset(feature_test, label_test, transform=nn.BatchNorm1d(64))

    batch_size = 32
    loaders = {
        'train' : torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True),
        
        'test'  : torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True),
    }

    
    cnn = CNN()
    loss_func = nn.CrossEntropyLoss()   
    optimizer = optim.Adam(cnn.parameters(), lr = 0.001) 

    num_epochs = 3

    train(num_epochs, cnn, loaders)
    print('Model has been trained {} times with {} datas'.format(num_epochs, len(train_dataset)))

    all_labels, all_preds = test()

    all_labels = all_labels.type(torch.int64)
    all_preds = all_preds.type(torch.int64)
    confusion_mat = confusion_matrix(all_labels, all_preds)
    labels = np.array(['1', '2', '3', '4', '5', '6', '7', '8'])
    plt.figure(figsize=(10,10))
    plot_confusion_matrix(confusion_mat, labels)