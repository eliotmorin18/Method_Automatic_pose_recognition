'''
Created on Apr 24, 2022

@author: deckyal
'''
import torch
from torchvision import transforms
from PIL import Image
from _operator import truediv
import os 
from scipy.io import loadmat
import matplotlib as plt
import numpy as np 

"""class CatDog(torch.utils.data.Dataset):
    def __init__(self, dataDir='../../../Data/IC-CatDog/train-small/', transform=None, crossNum=None,crossIDs=None):
        # Initialize the data and label list
        self.labels = []
        self.data = []
        temp = []
        tempL = []

        # First load all images data
        import os
        listImage = os.listdir(dataDir)
        listImage = sorted(listImage)

        for x in listImage:

            #print(x)

            # Read the data using PIL
            temp.append(Image.open(dataDir + x).convert('RGB'))

            # Second filter according name for labelling : cat : 1, dog : 0
            if 'dog' in x:
                tempL.append(torch.FloatTensor([0]))
            else:
                tempL.append(torch.FloatTensor([1]))"""
"""         if crossNum is not None: 
            totalLength = len(temp)
            length = int(truediv(totalLength,crossNum))
            
            
            for crossID in crossIDs : 
                lowR = crossID-1
                if crossID == crossNum: 
                    self.data.extend(temp[(lowR)*length:])
                    self.labels.extend(tempL[(lowR)*length:])
                else: 
                    self.data.extend(temp[(lowR)*length:(crossID)*length])
                    self.labels.extend(tempL[(lowR)*length:(crossID)*length])
                
        else: 
            self.data = temp
            self.labels = tempL
            
        self.transform = transform"""

def __getitem__(self, index):
        
        data = self.data[index]
        lbl = self.labels[index]



        if self.transform is not None:
            data = self.transform(data)

        return data, lbl

        pass

def __len__(self):
        #print(len(self.labels))
        return len(self.data) 



class LSP(torch.utils.data.Dataset):

    def __init__(self, dataDir='./dataset1/lsp/images/', transform=None):
        self.labels = []
        self.data = []

        # Charger la liste des images
        listImage = os.listdir(dataDir)
        listImage = sorted(listImage)
        print(len(listImage))

        # Charger les annotations depuis joints.mat
        data = loadmat('./dataset1/lsp/joints.mat')
        joints = data['joints']
        # print(joints.shape)  # Print the shape of joints
        for i in range(len(listImage)):
            img = listImage[i]
            joint = joints[:, :, i]  # Extraire les joints associés à l'image

            # Ajouter l'image et ses annotations
            self.data.append(dataDir + img)
            self.labels.append(joint)

        self.transform = transform

    def __getitem__(self, index):
        import matplotlib.pyplot as plt  # Assurez-vous que pyplot est importé

        data = self.data[index]
        lbl = self.labels[index]
       
        # Ouvrir l'image
        data = Image.open(data).convert('RGB')

        # Extraire les joints x et y
        x_joint = lbl[:, 0]
        y_joint = lbl[:, 1]




        min_x =max(0,np.min(x_joint)-15)
        min_y=max(0,np.min(y_joint)-15)
        max_x=max(0,np.max(x_joint)+15)
        max_y= max(0,np.max(y_joint)+15)

        data_crop = data.crop((min_x,min_y,max_x,max_y))

        lbl_crop = np.copy(lbl)

        lbl_crop[:,0] = lbl[:,0] - min_x
        lbl_crop[:,1]= lbl[:,1] -  min_y
            # Afficher l'image et les joints

        #Resize
        Size_X = 128
        Size_Y = 128

        data_resize = data_crop.resize(Size_X,Size_Y)
        scale_x = truediv(Size_X, data_crop.width)
        scale_y = truediv(Size_Y, data_crop.width)

        lbl_resize = np.copy(lbl_crop)
        lbl_resize[:,0] *=scale_x
        lbl_resize[:,1] *=scale_y

        plt.imshow(data_resize)
        plt.scatter(lbl_resize[:,0], lbl_resize[:,1], c='red', label='Joints')  # Ajout d'une couleur pour les points
        plt.legend()
        plt.title("Image crop"+ str(index))
        plt.show()


        if self.transform is not None:
             data = self.transform(data_resize)

        return data_resize, lbl_resize
    


    def __len__(self):
        return len(self.data)







if __name__ == '__main__':
    
    
    tr = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
        ])

    lsppe = LSP(transform=tr)
    print(len(lsppe))

    images,labels =next(iter(lsppe))

for i,(iamge,labels) in enumerate(lsppe):
    print(images,labels)
    print(images.shape)
    
    
