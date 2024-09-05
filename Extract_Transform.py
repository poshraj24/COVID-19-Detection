import os
import glob
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Define paths for the datasets
COVID_FOLDER = os.getcwd() + r'\DataSet\COVID'
NON_COVID_FOLDER = os.getcwd() + r'\DataSet\non-COVID'

# imdir= os.getcwd() + COVID_FOLDER
# ext=['png']
# files=[]
# [files.extend(glob.glob(imdir + '/*.' + e)) for e in ext]
# images=[cv2.imread(file) for file in files]

#Perform Data Preprocessing 
class CovidDataset(Dataset):
    def __init__(self,COVID_FOLDER,NON_COVID_FOLDER,transform=None):
        self.COVID_FOLDER = COVID_FOLDER
        self.NON_COVID_FOLDER = NON_COVID_FOLDER
        self.covid_images=glob.glob(COVID_FOLDER + '/*.png')
        self.non_covid_images=glob.glob(NON_COVID_FOLDER + '/*.png')
        self.labels= [1]*len(self.covid_images) + [0]*len(self.non_covid_images)
        self.images= self.covid_images + self.non_covid_images
        self.transform=transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        img_path=self.images[idx]
        image=Image.open(img_path)
        label=self.labels[idx]


        if self.transform:
            image=self.transform(image)

        return image,label
    
    # Define the transformations
transform= transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        transforms.Resize((224,224)),   #Resizing to 224*224 pixels
        transforms.RandomHorizontalFlip(), #Random Horizontal Flip
        transforms.RandomRotation(20), #Random Rotation
        transforms.ToTensor(), #Convert to tensor
        transforms.Normalize(mean=[0.5],std=[0.5]) #Normalize the image
        ])

#Initialize the dataset 
dataset=CovidDataset(COVID_FOLDER,NON_COVID_FOLDER,transform=transform)

#Split the dataset into training and testing
train_size=int(0.8*len(dataset))
test_size=len(dataset)-train_size
train_dataset,test_dataset=random_split(dataset,[train_size,test_size])

#Initialize the dataloaders
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=32,shuffle=True, drop_last=True)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=32,shuffle=False, drop_last=True)

# #Visualize the images for verification 
# def show_image(img,label):
#     img=img/2 + 0.5 #Unnormalize the image
#     npimg=img.numpy()
#     plt.imshow(np.transpose(npimg,(1,2,0)))
#     plt.title("Label: "+str(label))
#     plt.show()

# #Get some random training images(batch)
# data_iter=iter(train_loader)
# images,labels=next(data_iter)


# #Show images
# for i in range(4):
#     show_image(images[i],labels[i])