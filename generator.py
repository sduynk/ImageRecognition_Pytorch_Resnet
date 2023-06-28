import torch
from torch import nn
import numpy as np
import torchvision


# generator & discriminator
class ResidualBlock(nn.Module):
    def __init__(self,num_filters=64):
        super(ResidualBlock,self).__init__()
        self.layers=nn.Sequential(
            nn.Conv2d(in_channels=num_filters,out_channels=num_filters,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=num_filters),
            nn.PReLU(),
            nn.Conv2d(in_channels=num_filters,out_channels=num_filters,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=num_filters)
        )
        
    def forward(self,x):
        return self.layers(x)+x
        
    
class Generator(nn.Module):
    def __init__(self,
                 num_residual_block=16,
                 residual_channels=64):
        super(Generator,self).__init__()
        self.num_residual_block=num_residual_block
        
        self.conv1=nn.Sequential(nn.Conv2d(in_channels=3,
                                         out_channels=residual_channels,
                                         kernel_size=9,stride=1,
                                         padding=4),
                                 nn.PReLU())
        
        conv2=nn.ModuleList()
        for n in range(self.num_residual_block):
            #conv2.append(self._residual_block(num_filters=residual_channels))
            conv2.append(ResidualBlock(num_filters=residual_channels))
        self.conv2=conv2
        
        self.conv3=nn.Sequential(nn.Conv2d(in_channels=residual_channels,
                                      out_channels=residual_channels,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1),
                            nn.BatchNorm2d(num_features=residual_channels))
        
        #self.conv5=nn.Sequential(nn.Conv2d(in_channels=64,
        #                              out_channels=256, kernel_size=3, stride=1, padding=1), 
        #                    nn.PixelShuffle(upscale_factor=2),
        #                    nn.PReLU())
        self.conv4=nn.Sequential(nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1,padding=0,stride=1))
                                 #nn.Tanh())
                                 
        self.pool=nn.AdaptiveAvgPool2d(output_size=(1,1))
        
        self.fc = nn.Linear(64,1)
                        
    def forward(self,x):
        x=self.conv1(x)
        old_x=x
        for layer in self.conv2:
            x=layer(x)
        
        x=self.conv3(x)+old_x
        
        #print(x.shape)
        x=self.conv4(x)
        
        x=self.pool(x)
        
        x= x.reshape(-1,64)
        
        x = torch.sigmoid(self.fc(x))
        
        
        return x.flatten()
    
    
    
    
def get_model(model_name='resnet18',feat=256):
    if model_name == 'resnet50':
        model = torchvision.models.resnet50(weights='DEFAULT')
    elif model_name == 'resnet18':
        model = torchvision.models.resnet18(weights='DEFAULT')
    else:
        model = torchvision.models.resnet18(weights='DEFAULT')
        
        
    for  name, params in model.named_parameters():
        if  ('bn' not in name):
            params.required_grad = False
    model.fc=nn.Sequential(nn.Linear(model.fc.in_features,feat),
                          nn.ReLU(),
                          nn.Dropout(),
                          nn.Linear(feat,1),
                          nn.Sigmoid())
    return model
