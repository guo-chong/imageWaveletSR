import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torch import nn
import math
from torch.autograd import Variable

class PostRes(nn.Module):
    def __init__(self, n_in, n_out, stride=1):
        super(PostRes, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv2d(n_in, n_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self, n_in, n_out,kernelSize=3, stride=1,padding=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size=kernelSize, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out





class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet,self).__init__()
        self.f0=ConvBlock(15,64,5,1,2)
        self.d1=ConvBlock(64,64,3,2,1)
        self.f1=ConvBlock(64,128,3,1,1)
        self.f2=ConvBlock(128,128,3,1,1)
        self.d2=ConvBlock(128,256,3,2,1)

        self.wlNet=WLnet()


    def forward(self, x):
        pre=self.preBlock(x) #1
        of1=self.f1(pre)
        op1,id1=self.maxpool1(of1) #1/2
        of2=self.f2(op1)
        op2,id2=self.maxpool2(of2) #1/4
        of3=self.f3(op2)
        op3,id3=self.maxpool3(of3) #1/8
        of4=self.f4(op3)
        opath3=self.path3(of4) #1/4
        ob3=self.b3(opath3)
        comb3=torch.cat((of3,ob3),1)
        opath2=self.path2(comb3) #1/2
        ob2=self.b2(opath2)
        comb2=torch.cat((of2,ob2),1)
        opath1=self.path1(comb2) #1
        ob1=self.b1(opath1)
        comb1=torch.cat((of1,ob1),1)
        outwl=self.wlNet(comb1)
        return outwl


class WLnet(nn.Module):
    def __init__(self):
        super(WLnet,self).__init__()
        for i in range(16):
            block=[]
            block.append(ConvBlock(640,512))
            block.append(ConvBlock(512,64))
            block.append(ConvBlock(64,3))
            setattr(self,"wl"+str(i),nn.Sequential(*block))

    def forward(self, x):
        # owl0 = self.wl0(x)
        owl1=self.wl1(x)
        owl2=self.wl2(x)
        owl3=self.wl3(x)
        owl4=self.wl4(x)
        owl5=self.wl5(x)
        owl6=self.wl6(x)
        owl7=self.wl7(x)
        owl8=self.wl8(x)
        owl9=self.wl9(x)
        owl10=self.wl10(x)
        owl11=self.wl11(x)
        owl12=self.wl12(x)
        owl13=self.wl13(x)
        owl14=self.wl14(x)
        owl15=self.wl15(x)

        outwl=torch.cat((owl1,owl2,owl3,owl4,owl5,owl6,owl7,owl8,
                   owl9,owl10,owl11,owl12,owl13,owl14,owl15),1)
        return outwl

class Loss(nn.Module):
    def __init__(self):
        super(Loss,self).__init__()
        self.wlLoss=nn.MSELoss()
        self.imageLoss=nn.MSELoss()


    def dWl(self,data):
        sp = data.size()
        c = data.size()[1]
        llr = data[:, 0:int(c / 4), :, :]
        lhr = data[:, int(c / 4):int(c / 2), :, :]
        hlr = data[:, int(c / 2):int(c / 4 * 3), :, :]
        hhr = data[:, int(c / 4 * 3):c, :, :]
        lc=Variable(torch.zeros(sp[0], int(sp[1] / 4), sp[2] * 2, sp[3]),requires_grad=True).cuda(async=True)
        hc=Variable(torch.zeros(sp[0], int(sp[1] / 4), sp[2] * 2, sp[3]),requires_grad=True).cuda(async=True)


        # lc = np.zeros((sp[0], int(sp[1] / 4), sp[2] * 2, sp[3]), dtype=np.float32)
        # hc = np.zeros((sp[0], int(sp[1] / 4), sp[2] * 2, sp[3]), dtype=np.float32)
        # lc=torch.from_numpy(lc)
        # hc=torch.from_numpy(hc)
        image=Variable(torch.zeros(sp[0], int(sp[1] / 4), sp[2] * 2, sp[3] * 2),requires_grad=True).cuda(async=True)


        # image = np.zeros((sp[0], int(sp[1] / 4), sp[2] * 2, sp[3] * 2), dtype=np.float32)
        lc[:, :, 0::2, :] = llr + lhr
        lc[:, :, 1::2, :] = llr - lhr
        hc[:, :, 0::2, :] = hlr + hhr
        hc[:, :, 1::2, :] = hlr - hhr
        image[:, :, :, 0::2] = lc + hc
        image[:, :, :, 1::2] = lc - hc
        return image

    def forward(self, data,imageL,dataL,input,i,epoch,pahse):
        toDwl=torch.cat((input,data),1)
        data1=self.dWl(toDwl)
        imageO=self.dWl(data1)
        wlLoss=self.wlLoss(data,dataL)
        imageLoss=self.imageLoss(imageO,imageL)
        if(i%90==0):
            imageOWrite=imageO[0].cpu().data.numpy()
            imageOWrite[imageOWrite>=1]=1
            imageOWrite[imageOWrite<=0]=0
            imageOWrite*=255
            imageOWrite=np.array(imageOWrite.transpose(1,2,0),np.uint8)
            imageIWrite=imageL[0].cpu().data.numpy()
            imageIWrite[imageIWrite>=1]=1
            imageIWrite[imageIWrite<=0]=0
            imageIWrite*=255
            imageIWrite=np.array(imageIWrite.transpose(1,2,0),np.uint8)

            LR = input[0].cpu().data.numpy()
            LR[LR >= 1] = 1
            LR[LR <= 0] = 0
            LR *= 255
            LR = np.array(LR.transpose(1, 2, 0), np.uint8)

            cv2.imwrite("/home/user/disk2/video/"+pahse+"JPGV2/"+str(epoch)+"_"+str(i)+"_Out.jpg",imageOWrite)
            cv2.imwrite("/home/user/disk2/video/"+pahse+"JPGV2/"+str(epoch)+"_"+str(i)+"_In.jpg",imageIWrite)
            cv2.imwrite("/home/user/disk2/video/"+pahse+"JPGV2/"+str(epoch)+"_"+str(i)+"_LR.jpg",LR)
        loss=0.3*imageLoss+wlLoss
        return loss



