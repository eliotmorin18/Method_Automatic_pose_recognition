'''
Created on Apr 24, 2022

@author: deckyal
'''
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

from dataset import *
from models import *
from utils import * 
from metrics import * 
from config import device
import os

def train(model = None,SavingName=None, train_loader = None, val_loader=None, optimizer = None):
    # training
    print('training')
    
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        losses = []
        for i, (images, labels) in enumerate(train_loader):

            #print(images)

            images = images.to(device)
            labels = labels.to(device)

            labels_f = labels.reshape(labels.size(0), -1)
            
            # Forward pass
            outputs = model.forward(images)
            
            loss = torch.sqrt(torch.mean((outputs - labels_f) ** 2))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 2 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                
            
            image = images.cpu().numpy()

            """
            for k in range(images.shape[0]):
                labels_f_k = np.array(labels_f[k]).reshape(14,3)
                outputs_k = np.array(outputs[k].detach().numpy()).reshape(14,3)
                image = np.array(images[k])
                image = image.transpose(1, 2, 0) #(128, 128, 3)
                plt.imshow(image)
                plt.scatter(outputs_k[:,0], outputs_k[:,1], color='red')
                plt.scatter(labels_f_k[:,0], labels_f_k[:,1], color='blue')
                plt.legend()
                plt.title("test")
                plt.show()
                    #print(pred)
            """
            
                
            
            #do validations every 10 epoch 
            if i%10 == 0:
                with torch.no_grad():
                    
                    model.eval()        
                    pred,gt = [],[]
                    
                    for imagesV, labelsV in val_loader:
                        
                        imagesV = imagesV.to(device)
                        labelsV = labelsV.to(device)

                        labelsV = labelsV.reshape(labelsV.size(0), -1)
                        
                        # Forward pass
                        outputsV = model(imagesV).round()

                        #print(outputsV.shape)
                        #this one works
                        
                        """
                        for k in range(imagesV.shape[0]):
                            gt_k = np.array(labelsV[k]).reshape(14,3)
                            pred_k = np.array(outputsV[k]).reshape(14,3)
                            image = np.array(imagesV[k])
                            image = image.transpose(1, 2, 0) #(128, 128, 3)
                            print(pred_k)
                            print(gt_k)
                            plt.imshow(image)
                            plt.scatter(pred_k[:,0], pred_k[:,1], color='red')
                            plt.scatter(gt_k[:,0], gt_k[:,1], color='blue')
                            plt.legend()
                            plt.title("test")
                            plt.show()
                        """

                        gt.extend(labelsV.cpu().numpy())
                        pred.extend(outputsV.cpu().numpy())
                    
                    gt = np.asarray(gt,np.float32)
                    pred = np.asarray(pred)

                    imagesV = imagesV.cpu().numpy()

                    #not working
                    """
                    for k in range(imagesV.shape[0]):
                        gt_k = np.array(gt[k]).reshape(14,3)
                        pred_k = np.array(pred[k]).reshape(14,3)
                        image = np.array(imagesV[k])
                        image = image.transpose(1, 2, 0) #(128, 128, 3)
                        plt.imshow(image)
                        plt.scatter(pred_k[:,0], pred_k[:,1], color='red')
                        plt.scatter(gt_k[:,0], gt_k[:,1], color='blue')
                        plt.legend()
                        plt.title("test")
                        plt.show()
                    """
                    
                
                    print('loss : ', np.sum(np.sqrt(np.square(gt - pred))))
                    losses.append(np.sum(np.sqrt(np.square(gt - pred))))
                    
                model.train()
        

    # Save the model checkpoint
    checkDirMake(os.path.dirname(SavingName))
    torch.save(model.state_dict(), SavingName)
    # to load : model.load_state_dict(torch.load(save_name_ori))

def plotting_loss(model = None,SavingName=None, train_loader = None, val_loader=None, optimizer = None):
    # training
    print('training')
    
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        losses = []
        for i, (images, labels) in enumerate(train_loader):

            #print(images)

            images = images.to(device)
            labels = labels.to(device)

            labels_f = labels.reshape(labels.size(0), -1)
            
            # Forward pass
            outputs = model.forward(images)
            
            loss = torch.sqrt(torch.mean((outputs - labels_f) ** 2))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 2 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                
            
            image = images.cpu().numpy()
                   
            #do validations every 10 epoch 
            if i%10 == 0:
                with torch.no_grad():
                    
                    model.eval()        
                    pred,gt = [],[]
                    
                    for imagesV, labelsV in val_loader:
                        
                        imagesV = imagesV.to(device)
                        labelsV = labelsV.to(device)

                        labelsV = labelsV.reshape(labelsV.size(0), -1)
                        
                        # Forward pass
                        outputsV = model(imagesV).round()

                        gt.extend(labelsV.cpu().numpy())
                        pred.extend(outputsV.cpu().numpy())
                    
                    gt = np.asarray(gt,np.float32)
                    pred = np.asarray(pred)

                    imagesV = imagesV.cpu().numpy()
                
                    print('loss : ', np.sum(np.sqrt(np.square(gt - pred))))
                    losses.append(np.sum(np.sqrt(np.square(gt - pred))))
                    
                model.train()

        np.save('loss.npy',losses)
        np.load('loss.npy')
    
def test(model = None,SavingName=None, test_loader=None):
    # Test the model
    model.load_state_dict(torch.load(SavingName, map_location=torch.device('cpu')))

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
         
        pred,gt = [],[]
        
        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).round()
            
            gt.extend(labels.squeeze().cpu().numpy())
            pred.extend(outputs.squeeze().cpu().numpy())

            images = images.cpu().numpy()

            print(pred[0])
            print(gt[0])

            for k in range(images.shape[0]):
                pred_k = pred[k]
                gt_k = gt[k]

                pred_k_x = []
                pred_k_y = []
                for j in range(len(pred_k)):
                    if j%3==0:
                        pred_k_x.append(pred_k[j])
                    if j%3==1 : 
                        pred_k_y.append(pred_k[j])
                plt.imshow(images[k].transpose(1, 2, 0))
                plt.scatter(pred_k_x,pred_k_y, color='red')
                plt.scatter(gt_k[:,0], gt_k[:,1], color='blue')

                #plt.legend()
                plt.title("test")
                
                plt.show()
        
        gt = np.asarray(gt,np.float32)
        pred = np.asarray(pred)

        print('Test Accuracy of the model on test images: {} %'.format(accuracy(pred,gt)))
        
        
if __name__ == '__main__':
    
    
    tr = transforms.Compose([
        #transforms.Resize((224,224)),
        transforms.ToTensor()
        ])
    
    
    batch_size = 16
    
    PoseTrain = LSPPE(transform=tr)
    train_loader = torch.utils.data.DataLoader(dataset=PoseTrain,
                                               batch_size=batch_size,
                                               shuffle=True)
    
    PoseVal = LSPPE(transform=tr)
    val_loader = torch.utils.data.DataLoader(dataset=PoseVal,
                                               batch_size=batch_size,
                                               shuffle=True)

    
    
    #FCI = FC(inputNode=3*64*64).to(device)
    CNNI = CNN(num_classes=42).to(device)
    
    learning_rate = .0001
    num_epochs = 3
    optimizer = torch.optim.Adam(CNNI.parameters(), lr=learning_rate)

    operation = 3
    
    if operation ==0 or operation==2: 
        train(model = CNNI,SavingName='./checkpoints/CNN-pose-Baseline.ckpt', train_loader = train_loader, val_loader=val_loader, optimizer = optimizer)
    if operation ==1 or operation==2: 
        test(model = CNNI,SavingName='./checkpoints/CNN-pose-Baseline.ckpt', test_loader=val_loader)
    if operation == 3:
        plotting_loss(model = CNNI,SavingName='./checkpoints/CNN-pose-Baseline.ckpt', train_loader = train_loader, val_loader=val_loader, optimizer = optimizer)
        
        
        