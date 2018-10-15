# -*- coding:utf-8 -*-
import numpy as np
import dataLoader
import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
import nets
from torch.autograd import Variable

import time
import os
import pynvml
def getFreeId():


    pynvml.nvmlInit()
    def getFreeRatio(id):
        handle = pynvml.nvmlDeviceGetHandleByIndex(id)
        use = pynvml.nvmlDeviceGetUtilizationRates(handle)
        ratio = 0.5*(float(use.gpu+float(use.memory)))
        return ratio

    deviceCount = pynvml.nvmlDeviceGetCount()
    available = []
    for i in range(deviceCount):
        if getFreeRatio(i)<70:
            available.append(i)
    gpus = ''
    for g in available:
        gpus = gpus+str(g)+','
    gpus = gpus[:-1]
    return gpus
def setgpu(gpuinput):
    freeids = getFreeId()
    if gpuinput=='all':
        gpus = freeids
    else:
        gpus = gpuinput
    print('using gpu '+gpus)
    os.environ['CUDA_VISIBLE_DEVICES']=gpus
    return len(gpus.split(','))

def train(dataLoader,net,loss,epoch,optimizer,getlr,save_dir,logfile,epochs):
    startTime=time.time()
    net.train()
    lr=getlr(epoch,epochs)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    l=0
    aveLoss=0
    print "train epoch=, " + str(epoch)
    for i, (input,dataL,imageL) in enumerate(dataLoader):
        input=Variable(input.cuda(async =True))
        # print input.size(), dataL.size(),imageL.size() #######################
        dataL=Variable(dataL.cuda(async =True))
        imageL=Variable(imageL.cuda(async =True))

        outwl=net(input)
        lossOut=loss(outwl,imageL,dataL,input,i,epoch,"train")
        optimizer.zero_grad()
        lossOut.backward()
        optimizer.step()
        aveLoss += lossOut.data[0]
        l += 1
    aveLoss /= l
    timeLen = time.time() - startTime
    print "time: " + str(timeLen)
    print("train ,     epoch , " + str(epoch) + " loss, " + str(aveLoss)+" l, "+str(l))
    file = open(logfile, "a")
    file.write("i ,   epoch ," + str(epoch) + " loss , " + str(aveLoss) + "\n")
    file.close()





def val(dataLoader,net,loss,epoch,getlr,save_dir,logfile,epochs):
    startTime=time.time()
    aveLoss=0
    l=0
    print "val epoch=, "+str(epoch)
    for i, (input,dataL,imageL) in enumerate(dataLoader):
        input=Variable(input.cuda(async =True))
        # print input.size(), dataL.size(),imageL.size() #######################
        dataL=Variable(dataL.cuda(async =True))
        imageL=Variable(imageL.cuda(async =True))

        outwl=net(input)
        lossOut=loss(outwl,imageL,dataL,input,i,epoch,"val")
        aveLoss+=lossOut.data[0]
        l+=1
    aveLoss/=l
    timeLen=time.time()-startTime
    print "time: "+str(timeLen)
    print("val ,     epoch , "+str(epoch)+" loss, "+str(aveLoss)+" l, "+str(l))
    file=open(logfile,"a")
    file.write("i ,   epoch ,"+str(epoch)+" loss , "+str(aveLoss)+"\n")
    file.close()


def main():
    torch.manual_seed(0)
    # torch.cuda.set_device(1)
    setgpu("all")

    epochs=1000
    def getlr(epoch,epochs):
        lr=0.01
        if epoch <= epochs * 0.5:
            lr = lr
        elif epoch <= epochs * 0.8:
            lr = 0.1 * lr
        else:
            lr = 0.01 * lr
        return lr

    datadir = "/home/user/disk2/video/2017/"
    savedir = "/home/user/disk2/video/saveV2/"
    logfile = os.path.join(savedir, 'log.txt')
    logfileVal = os.path.join(savedir, 'logVal.txt')

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    dataset=dataLoader.DataSet(datadir)
    datasetVal=dataLoader.DataSetVal(datadir)

    net=nets.EmbeddingNet()
    # checkpoint = torch.load(savedir+"428.ckpt")
    # net.load_state_dict(checkpoint)

    net = DataParallel(net)
    net=net.cuda()
    loss = nets.Loss()
    loss = loss.cuda()
    trainLoader=DataLoader(dataset,batch_size=48,shuffle=True,num_workers=12,pin_memory=True)
    valLoader=DataLoader(datasetVal,batch_size=6,shuffle=True,num_workers=18,pin_memory=True)


    cudnn.benchmark=True

    lr=0.01
    optimizer=torch.optim.SGD(
        net.parameters(),
        lr,
        momentum=0.9,
        weight_decay=1e-4
    )
    for epoch in range(epochs):
        train(trainLoader,net,loss,epoch,optimizer,getlr,savedir,logfile,epochs)
        if epoch%10==0:
            val(valLoader,net,loss,epoch,getlr,savedir,logfileVal,epochs)
            state_dict=net.module.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
            torch.save(state_dict,
                    os.path.join(savedir, '%03d.ckpt' % epoch))
            print "save "+str(epoch)
        file=open(logfile,"a")
        file.write("save "+str(epoch))
        file.close()





def mainVal():
    torch.manual_seed(0)
    # torch.cuda.set_device(1)
    setgpu("all")

    epochs=500
    def getlr(epoch,epochs):
        lr=0.01
        if epoch <= epochs * 0.5:
            lr = lr
        elif epoch <= epochs * 0.8:
            lr = 0.1 * lr
        else:
            lr = 0.01 * lr
        return lr

    datadir = "/home/user/disk2/video/2017/"
    savedir = "/home/user/disk2/video/save/"
    logfile = os.path.join(savedir, 'log.txt')
    logfileVal = os.path.join(savedir, 'logVal.txt')

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    dataset=dataLoader.DataSet(datadir)
    datasetVal=dataLoader.DataSetVal(datadir)

    loss = nets.Loss()
    loss = loss.cuda()
    # trainLoader=DataLoader(dataset,batch_size=16,shuffle=True,num_workers=3,pin_memory=True)
    valLoader=DataLoader(datasetVal,batch_size=6,shuffle=True,num_workers=3,pin_memory=True)


    cudnn.benchmark=True

    lr=0.01

    for epoch in range(20,epochs,20):

        net = nets.EmbeddingNet()
        checkpoint = torch.load(savedir + '%03d.ckpt' % epoch)
        net.load_state_dict(checkpoint)

        net = DataParallel(net)
        net = net.cuda()
        # train(trainLoader,net,loss,epoch,optimizer,getlr,savedir,logfile,epochs)
        val(valLoader,net,loss,epoch,getlr,savedir,logfileVal,epochs)


        # state_dict=net.module.state_dict()
        # file=open(logfile,"a")
        # file.write("save "+str(epoch))
        # file.close()



main()