import torch
from torchvision import transforms
from PIL import Image
from _operator import truediv
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

class LSPPE(torch.utils.data.Dataset):
    def __init__(self, dataDir='./lsp', transform=None, crossNum=None,crossIDs=None):

        self.labels = []
        self.data = []

        # First load all images data
        import os
        listImage = os.listdir(dataDir + '/images')
        listImage = sorted(listImage)

        #print(len(listImage))

        data = loadmat(dataDir + '/joints.mat')
        joints = data['joints']

        #print(joints.shape) #Print the shape of joints

        for i in range(len(listImage)):
            img = listImage[i]
            joint = joints[:,:,i]
            self.data.append(dataDir + '/images' + '/' + img)
            self.labels.append(joint)
        
        self.transform = transform
        
        """
        a = joints[:,:,1] #the shape is (14, 3, 10000): 14 joints; 3 for axes x,y,z (z is always 1) ; 10000 images in data (0 to 9999)

        #run all joints to print them on the terminal to see what it looks like
        for x in a :
            print(x)

        """

        """
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
                tempL.append(torch.FloatTensor([1]))
            
        self.transform = transform
        """

    def __getitem__(self, index):
        
        data = self.data[index]
        lbl = self.labels[index]

        data = Image.open(data).convert('RGB')

        x_joint = lbl[:, 0]
        y_joint = lbl[:, 1]

        """
        plt.imshow(data)
        plt.scatter(x_joint,y_joint)
        plt.legend()
        plt.title("First image")
        plt.show()
        """

        #crop the image using min_x, min_y and max_y, max_x
        min_x = max(0,np.min(x_joint) - 15)
        min_y = max(0,np.min(y_joint) - 15)
        max_x = max(0,np.max(x_joint) + 15)
        max_y = max(0,np.max(y_joint) + 15)

        #print((min_x,min_y,max_x,max_y))

        data_crop = data.crop((min_x,min_y,max_x,max_y))

        #adjust the ground truth using min_x and min_y
        lbl_crop = np.copy(lbl)

        lbl_crop[:, 0] = lbl[:, 0] - min_x
        lbl_crop[:, 1] = lbl[:, 1] - min_y

        """
        plt.imshow(data_crop)
        plt.scatter(lbl_crop[:,0], lbl_crop[:,1])
        plt.legend()
        plt.title("crop")
        plt.show()
        """

        #resize the image using conventional size of 128 and 128
        SIZE_X = 128
        SIZE_Y = 128

        data_resize = data_crop.resize((SIZE_X,SIZE_Y))

        lbl_resize = np.copy(lbl_crop)
        lbl_resize[:,0] = (lbl_crop[:,0]/data_crop.width) * SIZE_X
        lbl_resize[:,1] = (lbl_crop[:,1]/data_crop.height) * SIZE_Y
        
        """
        plt.imshow(data_resize)
        plt.scatter(lbl_resize[:,0], lbl_resize[:,1])
        plt.legend()
        plt.title("resize")
        plt.show()
        """

        if self.transform is not None:
            data_resize = self.transform(data_resize)

        return data_resize, torch.tensor(lbl_resize)

    def __len__(self):
        #print(len(self.labels))
        return len(self.data)

if __name__ == '__main__':
    
    
    tr = transforms.Compose([
        #transforms.Resize((64,64)),
        transforms.ToTensor()
        ])

    
    lsppe = LSPPE(transform=tr,crossNum=5, crossIDs=[5])
    print(len(lsppe))

    images,labels = next(iter(lsppe))
    print(images,labels)
    print(images.shape)
    print(labels.shape)

    for i, (imaages,labels) in enumerate(lsppe):
        print(images.shape)
        print(labels.shape)

    exit(0)