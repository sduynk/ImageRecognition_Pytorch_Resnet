import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import glob
import numpy as np
import shutil
from tqdm import tqdm
import random
from numpy import genfromtxt
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import train_test_split


class ScatterData(Dataset):
    def __init__(self,
                 path=r'C:\Users\bdutta\work\pys\AI_algos\HSP_ml',
                 transform=transforms.Compose([
                                                  transforms.ToTensor()
                                              ]),
                 verbose=False
                ):
        '''
        load microscopy data & transform to lr
        # path : directory
        # transform: list of transforms for HR/original images
        # verbose : print and plot

        '''
        super(ScatterData,self).__init__()
        self.path=path
        
        os.chdir(path)
        ids=glob.glob('*') #s2/s1 files
        self.ids=ids
    
        self.transform=transform
        self.verbose=verbose
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self,idx):
        basename=self.ids[idx]
        if self.verbose : print(basename)
        (image,label)=self.get_data(basename)
        
        #print(image.shape)
        #plt.imshow(image)
        #print(type(image))
        
        if self.transform : image=self.transform(image)
        
        if self.verbose:
            plt.figure(figsize=(12,4))
            #dummy=image
            plt.imshow(image.permute(1,2,0).numpy())
            plt.axis('off');
            dummy='soluble' if label else 'insoluble'
            plt.title(basename+'label :'+dummy)

        return (image,torch.tensor(label,dtype=torch.float32))
    
    def get_data(self, basename=None):
    
        x=Image.open(os.path.join(self.path,basename))
        
        x = np.array(x)
        x=(x-127.5)/127.5
    
        y=basename.split('_')[-2]
        
        y = 0 if y =='insoluble' else 1

        return (x,y)


# Creating data indices for training and validation splits
def train_test_split_torch(dataset,
                     validation_split=0.2,
                     shuffle_dataset=True,
                     batch_size=10
                    ):

    
    dataset_size=len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset :
        np.random.seed(112)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    
    return (train_loader,validation_loader)
    
    
# Creating data indices for training and validation splits
def train_test_split_torch_stratify(dataset,
                     test_split=0.1,
                     shuffle_dataset=True,
                     batch_size=5
                    ):

    
    dataset_size=len(dataset)
    indices = list(range(dataset_size))
    #split = int(np.floor(validation_split * dataset_size))
    # get labels
    targets=[]
    print('wait: retrieving targets!')
    for k in range(len(dataset)):
        _,y=dataset[k]
        targets.append(y.item())
    
    train_indices,test_indices,_,_=train_test_split(indices,
                                                    targets,
                                                    stratify=targets,
                                                    test_size=test_split,
                                                    random_state=11)

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=test_sampler)
    
    return (train_loader,test_loader)
  