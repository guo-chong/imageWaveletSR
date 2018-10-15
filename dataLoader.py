from __future__ import print_function
import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset

class DataSet(Dataset):
    def __init__(self,datadir,phase='train'):
        self.phase=phase
        self.datadir=datadir
        self.fileList=glob.glob(datadir+"*.jpg")
    def __getitem__(self, idx):
        images=cv2.imread(self.datadir+str(idx*25)+".jpg")
        image = np.array([images], dtype=np.float32)/255
        image = image.transpose(0, 3, 1, 2)
        image=image[:,:,256:512,256:512]
        data=self.wl(image)
        data1=self.wl(data)
        imageL=image
        dataL=data1[:,3:48]
        input=data1[:,0:3,:,:]
        return  torch.from_numpy(input)[0] ,torch.from_numpy(dataL)[0],torch.from_numpy(imageL)[0]



    def show(img):
        imgrgb=np.array([img[2],img[1],img[0]]).transpose(1,2,0)
        plt.imshow(imgrgb)
        plt.show()

    def write(img):
        import cv2
        img*=255
        img=np.array(img.transpose(1,2,0),np.uint8)
        print (cv2.imwrite("/home/user/x.jpg",img))
    def __len__(self):
        return int(len(self.fileList)/25)-1

    def wl(self,image):
        lc = (image[:, :, :, 0::2] + image[:, :, :, 1::2]) / 2
        hc = (image[:, :, :, 0::2] - image[:, :, :, 1::2]) / 2
        llr = (lc[:, :, 0::2, :] + lc[:, :, 1::2, :]) / 2
        lhr = (lc[:, :, 0::2, :] - lc[:, :, 1::2, :]) / 2
        hlr = (hc[:, :, 0::2, :] + hc[:, :, 1::2, :]) / 2
        hhr = (hc[:, :, 0::2, :] - hc[:, :, 1::2, :]) / 2
        data=np.concatenate((llr,lhr,hlr,hhr),1)
        return data

    def dWl(self,data):
        sp=data.shape
        c=data.shape[1]
        llr=data[:,0:int(c/4),:,:]
        lhr=data[:,int(c/4):int(c/2),:,:]
        hlr=data[:,int(c/2):int(c/4*3),:,:]
        hhr=data[:,int(c/4*3):c,:,:]
        lc=np.zeros((sp[0],int(sp[1]/4),sp[2]*2,sp[3]),dtype=np.float32)
        hc=np.zeros((sp[0],int(sp[1]/4),sp[2]*2,sp[3]),dtype=np.float32)
        image=np.zeros((sp[0],int(sp[1]/4),sp[2]*2,sp[3]*2),dtype=np.float32)
        lc[:,:,0::2,:]=llr+lhr
        lc[:,:,1::2,:]=llr-lhr
        hc[:,:,0::2,:]=hlr+hhr
        hc[:,:,1::2,:]=hlr-hhr
        image[:,:,:,0::2]=lc+hc
        image[:,:,:,1::2]=lc-hc
        return image



class DataSetVal(Dataset):
    def __init__(self,datadir,phase='train'):
        self.phase=phase
        self.datadir=datadir
        self.fileList=glob.glob(datadir+"*.jpg")
    def __getitem__(self, idx):
        images=cv2.imread(self.datadir+str(idx*50)+".jpg")
        image = np.array([images], dtype=np.float32)/255
        image = image.transpose(0, 3, 1, 2)
        image=image[:,:,512:512+256,512:512+256]
        data=self.wl(image)
        data1=self.wl(data)
        imageL=image
        dataL=data1[:,3:48]
        input=data1[:,0:3,:,:]
        return  torch.from_numpy(input)[0] ,torch.from_numpy(dataL)[0],torch.from_numpy(imageL)[0]



    def show(img):
        imgrgb=np.array([img[2],img[1],img[0]]).transpose(1,2,0)
        plt.imshow(imgrgb)
        plt.show()

    def write(img):
        import cv2
        img*=255
        img=np.array(img.transpose(1,2,0),np.uint8)
        print (cv2.imwrite("/home/user/x.jpg",img))
    def __len__(self):
        return int(len(self.fileList)/50)-1

    def wl(self,image):
        lc = (image[:, :, :, 0::2] + image[:, :, :, 1::2]) / 2
        hc = (image[:, :, :, 0::2] - image[:, :, :, 1::2]) / 2
        llr = (lc[:, :, 0::2, :] + lc[:, :, 1::2, :]) / 2
        lhr = (lc[:, :, 0::2, :] - lc[:, :, 1::2, :]) / 2
        hlr = (hc[:, :, 0::2, :] + hc[:, :, 1::2, :]) / 2
        hhr = (hc[:, :, 0::2, :] - hc[:, :, 1::2, :]) / 2
        data=np.concatenate((llr,lhr,hlr,hhr),1)
        return data

    def dWl(self,data):
        sp=data.shape
        c=data.shape[1]
        llr=data[:,0:int(c/4),:,:]
        lhr=data[:,int(c/4):int(c/2),:,:]
        hlr=data[:,int(c/2):int(c/4*3),:,:]
        hhr=data[:,int(c/4*3):c,:,:]
        lc=np.zeros((sp[0],int(sp[1]/4),sp[2]*2,sp[3]),dtype=np.float32)
        hc=np.zeros((sp[0],int(sp[1]/4),sp[2]*2,sp[3]),dtype=np.float32)
        image=np.zeros((sp[0],int(sp[1]/4),sp[2]*2,sp[3]*2),dtype=np.float32)
        lc[:,:,0::2,:]=llr+lhr
        lc[:,:,1::2,:]=llr-lhr
        hc[:,:,0::2,:]=hlr+hhr
        hc[:,:,1::2,:]=hlr-hhr
        image[:,:,:,0::2]=lc+hc
        image[:,:,:,1::2]=lc-hc
        return image
