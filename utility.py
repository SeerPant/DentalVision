#for modularity

import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class dentalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        csv_path = os.path.join(root_dir, "_classes.csv")
        self.df = pd.read_csv(csv_path)

        self.image_names = self.df.iloc[:, 0].values
        self.class_names = list(self.df.columns[1:])
        self.labels = self.df.iloc[:, 1:].values.argmax(axis=1)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_names[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


#function for accuracy calculation 
import torch
from torch import nn
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() #torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc
loss_fn = nn.CrossEntropyLoss()
#function to calculate precision, recall and f1score 
def precision_recall_f1(y_pred, y_true, classes = 6):
    precisionList = [] 
    recallList = [] 
    f1List = []

    for cls in range(classes):
        TP = ((y_pred == cls) & (y_true == cls)).sum().item()
        FP = ((y_pred == cls) & (y_true != cls)).sum().item()
        FN = ((y_pred !=cls) & (y_true == cls)).sum().item()

        precision = TP/(TP+FP) if (TP+FP) >0 else 0.0
        recall = TP/(TP+FN) if (TP + FN) >0 else 0.0
        f1_score = 2*((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
    
        precisionList.append(precision)
        recallList.append(recall)
        f1List.append(f1_score)
        
    precision = sum(precisionList) /classes
    recall = sum(recallList) / classes
    f1 = sum(f1List)/classes

    return precision,recall,f1


device = "cuda" if torch.cuda.is_available() else "cpu"
#Function of train loop for model 
"""Things needed: 
model, dataloader, 
model, optimizer, 
accuracy function""" 

def train_step(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device):
    """Performs training step of model"""
    train_loss, train_accuracy = 0,0
    train_preds, train_labels = [],[]

    #setting model to training mode 
    model.train()

    #looping through the training batches 
    for batch,(X,y) in enumerate(data_loader):
        #set data in taget device
        X,y = X.to(device), y.to(device)

        #conduct forward pass 
        y_pred = model(X)

        #accuracy and loss calculation per batch
        loss = loss_fn(y_pred,y)
        train_loss += loss.item()
        train_accuracy += accuracy_fn(y_true = y, y_pred = y_pred.argmax(dim=1)) #logits to prediction labels
        optimizer.zero_grad()
        loss.backward() 
        #optimize model's parameter once per batch
        optimizer.step()
    
        train_preds.append(y_pred.argmax(dim=1))
        train_labels.append(y)

    train_preds = torch.cat(train_preds)
    train_labels = torch.cat(train_labels)

    precision,recall,f1 = precision_recall_f1(train_labels,train_preds)

    train_loss /= len(data_loader)
    train_accuracy /= len(data_loader)


    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_accuracy:.2f}% | Precision:{precision:.4f} | Recall :{recall:.4f} | f1-score: {f1:.4f}")

#test loop of model
def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    """Performs test step of the model"""
    test_loss, test_accuracy = 0, 0
    predictions, labels, probabilities = [],[],[]
    model.to(device)
    #Set model to evaluation mode
    model.eval() 
    #Turn on inference context manager
    with torch.inference_mode():
        #image,label 
        for X, y in data_loader:
            #Send data to GPU
            X, y = X.to(device), y.to(device)
            
            #Perform forward pass
            test_pred = model(X)

            #calculation of probaility using softtmax, for ROC curves
            probs = torch.softmax(test_pred, dim=1) 
            
            
            #Calculate testing loss and accuracy
            test_loss += loss_fn(test_pred, y).item()
            test_accuracy += accuracy_fn(y_true=y,
                y_pred=test_pred.argmax(dim=1) 
            )

            predictions.append(test_pred.argmax(dim=1))
            labels.append(y)
            probabilities.append(probs) 

        predictions = torch.cat(predictions)
        labels = torch.cat(labels)
        probabilities = torch.cat(probabilities)

        precision, recall, f1 = precision_recall_f1(labels, predictions)
        test_loss /= len(data_loader)
        test_accuracy /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_accuracy:.2f}% | Precision:{precision:.4f} | Recall :{recall:.4f} | f1-score: {f1:.4f}") 
    return predictions, labels, probabilities