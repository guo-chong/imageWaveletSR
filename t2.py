from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import numpy as np
k=[-1,1]
kernel=np.zeros((3,3,1,2),dtype=np.float32)
kernel[0,0]=k
kernel[1,1]=k
kernel[2,2]=k

def getKernel():
    k={
        "l":[0.5,0.5],
        "h":[-0.5,0.5],
    }
    kernel={}
    for key in ["lc","hc"]:
        kernel[key]=np.zeros((3,3,1,2),dtype=np.float32)
    for i in range(3):
        kernel["lc"][i,i]=k["l"]
        kernel["hc"][i,i]=k["h"]
    kernel["lr"]=torch.nn.parameter.Parameter(torch.FloatTensor(kernel["lc"].transpose(0,1,3,2)))
    kernel["hr"]=torch.nn.parameter.Parameter(torch.FloatTensor(kernel["hc"].transpose(0,1,3,2)))
    kernel["lc"] = torch.nn.parameter.Parameter(torch.FloatTensor(kernel["lc"]))
    kernel["hc"] = torch.nn.parameter.Parameter(torch.FloatTensor(kernel["hc"]))
    return kernel




#
# kernelCol=torch.FloatTensor(kernel)
# kernelRow=torch.FloatTensor(kernel.transpose(0,1,3,2))

kernel=getKernel()
class Wavelet(nn.Module):
    def __init__(self):
        super(Wavelet, self).__init__()
        self.lc=nn.Conv2d(3,3,kernel_size=(1,2),stride=(1,2),padding=0,bias=False) #lc
        self.hc=nn.Conv2d(3,3,kernel_size=(1,2),stride=(1,2),padding=0,bias=False) #hc
        self.llr=nn.Conv2d(3,3,kernel_size=(2,1),stride=(2,1),padding=0,bias=False) #lr
        self.lhr=nn.Conv2d(3,3,kernel_size=(2,1),stride=(2,1),padding=0,bias=False) #hr
        self.hlr=nn.Conv2d(3,3,kernel_size=(2,1),stride=(2,1),padding=0,bias=False) #lr
        self.hhr=nn.Conv2d(3,3,kernel_size=(2,1),stride=(2,1),padding=0,bias=False) #hr
        self.training=False

    def forward(self,x):
        xlc=self.lc(x)
        xhc=self.hc(x)
        xllr=self.llr(xlc)
        xlhr=self.lhr(xlc)
        xhlr=self.hlr(xhc)
        xhhr=self.hhr(xhc)
        data=torch.cat([xllr,xlhr,xhlr,xhhr],dim=1)
        return data

image=cv2.imread("K:\LS3D-W\\300VW-3D\Trainset\\013\\0541.jpg")
image=np.array([image],dtype=np.float32)
image=image.transpose(0,3,1,2)
model=Wavelet()
model.lc._parameters["weight"]=kernel["lc"]
model.hc._parameters["weight"]=kernel["hc"]
model.llr._parameters["weight"]=kernel["lr"]
model.lhr._parameters["weight"]=kernel["hr"]
model.hlr._parameters["weight"]=kernel["lr"]
model.hhr._parameters["weight"]=kernel["hr"]


# model.col._parameters["weight"]=torch.nn.parameter.Parameter(kernelCol)
# model.row._parameters["weight"]=torch.nn.parameter.Parameter(kernelRow)
x = torch.Tensor(image)
x=Variable(x)
y=model(x)
t=y.data.numpy()[0,0]
plt.imshow(t,"gray")




z=1



