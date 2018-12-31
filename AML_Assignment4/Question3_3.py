
# coding: utf-8

# In[1]:


from __future__ import print_function, division


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import os
import torch.nn.functional as  F
import copy


# In[2]:


model = models.resnet50(pretrained=True)


# In[3]:


print(model)


# In[4]:


model.fc = nn.Sequential(nn.Linear(2048,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,10),
            nn.Softmax())
# for child in model.children():
#     ct+=1
#     if ct <= 9:
#         for param in child.parameters():
#             param.requires_grad = False


# In[5]:


print(model.parameters)


# In[6]:


### CIFAR Loading ######
path = "../AML_Assignment2/cifar/"
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import skimage.transform as skt

def extractImagesAndLabels(path, file):
    f = open(path+file, 'rb')
    dicte = pickle.load(f,encoding='bytes')
    images = dicte[b'data']
    Matrix=[]
    for i in images:
        ingle_img_reshaped =  np.reshape(i,(3, 32,32))
        Matrix.append(ingle_img_reshaped)
    Matrix = np.array(Matrix)
    labels = dicte[b'labels']
    labels = np.array(labels)
    return Matrix, labels

def extractCategories(path, file):
    f = open(path+file, 'rb')
    dict = pickle.load(f,encoding='bytes')
    return dict[b'label_names']

def saveCifarImage(array, path, file):
    # array is 3x32x32. cv2 needs 32x32x3
    array = array.asnumpy().transpose(1,2,0)
    # array is RGB. cv2 needs BGR
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # save to PNG file
    return cv2.imwrite(path+file+".png", array)


Train_data, Train_labels = extractImagesAndLabels(path, "data_batch_1")
Train_data_2, Train_labels_2 = extractImagesAndLabels(path,"data_batch_2")
Train_data_3, Train_labels_3 = extractImagesAndLabels(path, "data_batch_3")
Train_data_3, Train_labels_3 = extractImagesAndLabels(path, "data_batch_3")
Train_data_4, Train_labels_4 = extractImagesAndLabels(path, "data_batch_3")
Train_data_5, Train_labels_5 = extractImagesAndLabels(path, "data_batch_3")
Train_data_t = np.concatenate((Train_data, Train_data_2,Train_data_3,Train_data_4,Train_data_5), axis=0)
Train_labels_t = np.concatenate((Train_labels,Train_labels_2,Train_labels_3,Train_labels_4,Train_labels_5), axis=0)
Test_data, Test_labels = extractImagesAndLabels(path, "test_batch")


# In[7]:


import torch.utils.data as data_utils

Train_data_t = torch.from_numpy(Train_data_t)
Train_labels_t = torch.from_numpy(Train_labels_t)
Train_labels_t = Train_labels_t.type(torch.FloatTensor)
Train_data_t = Train_data_t.type(torch.FloatTensor)
train_l = data_utils.TensorDataset(Train_data_t, Train_labels_t)
trainLoader= data_utils.DataLoader(train_l, batch_size=16, shuffle=True)
Test_data = torch.from_numpy(Test_data)
Test_labels = torch.from_numpy(Test_labels)
Test_data  = Test_data.type(torch.FloatTensor)
Test_labels = Test_labels.type(torch.FloatTensor)
testLoader = data_utils.TensorDataset(Test_data,Test_labels)


# In[8]:


testLoader= data_utils.DataLoader(testLoader, batch_size=16, shuffle=True)


# In[9]:


# Test_smaple = Test_data[0]

# Test_smaple = Test_smaple.data.resize_(3,224,224)


# In[10]:


criteria  = nn.CrossEntropyLoss()
def train(model_d, device, train_loader, optimizer,epoch):
    model_d.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.type(torch.FloatTensor)
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        data = F.upsample(data, size=(224,224), mode="bilinear")
#         print(data.size())
#         target = target-1
        data,target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model_d(data)
        loss = criteria(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            

def test(model_d, device, test_loader, set_name, contain,acc):
    model_d.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.type(torch.FloatTensor)
            target = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device)
            data = F.upsample(data, size=(224,224), mode="bilinear")
            data,target = Variable(data), Variable(target)
#             target = target-1
            output = model_d(data)
            test_loss += criteria(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    contain.append(test_loss)
    acc.append(correct/len(test_loader.dataset))
    print("Accuracy......")
    print('\n'+set_name+': Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# In[11]:


def model_activate(m):
    learning_rate = 0.00005
    loss_train=[]
    acc_train=[]
    loss_test=[]
    acc_test=[]
    ep=[]
    criterion = nn.CrossEntropyLoss()
    
    params = list(m.parameters()) + list(m.fc.parameters())
    optimizer = torch.optim.Adam(params,lr=learning_rate)
    num_epochs=10
    total_step =len(Train_data_t)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     device='cpu'
    print("Starting............")
    for epoch in range(1,num_epochs+1):
        ep.append(epoch)
        train(m, device, trainLoader, optimizer, num_epochs)
        test(m, device, trainLoader, "Training_set",loss_train, acc_train)
        test(m, device, testLoader, "Test_Set",loss_test,acc_test)
    return loss_train,loss_test,acc_test,acc_train,ep


# In[12]:


def loss_plot(ep, loss_train, loss_test, subject):
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(subject)
    plt.plot(ep,loss_train)
    plt.plot(ep,loss_test)
    plt.legend(["Loss-Train", "Loss-Test"], loc='upper right')
    plt.savefig("Figures/"+subject+"_loss_plot.png")

def acc_plot(ep,acc_train,acc_test,subject):
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(subject)
    plt.plot(ep,acc_train)
    plt.plot(ep,acc_test)
    plt.legend(["Acuracy-Train", "Accuracy-Test"], loc='lower right')
    plt.savefig("Acc_plot_(3).png")
    
def gradient_plot(gradient,name):
    plt.xlabel("Epoch")
    es = [i for i in range(len(gradient))]
    plt.ylabel("Gradient")
    plt.title(name)
    plt.plot(es,gradient)
    # plt.plot(ep,acc_test)
    plt.legend(["Gradient"], loc='upper right')
    plt.savefig("Figures/"+subject+"_grad_plot.png")
    
    
def plot_kernels(tensor, name,num_cols=6):
    print (tensor.shape)
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    if not tensor.shape[-1]==3:
        raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    print(num_cols,num_rows)
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        p = tensor[i,0]
        p = p.flatten()
        fig = plt.gcf()
        fig.set_size_inches(4, 3)
#         fig.savefig('test2png.png', dpi=10, forward=True)
        plt.hist(p, normed=True, bins=9)
        plt.ylabel("Weights-Dist"+name) 
        plt.savefig("Figures/"+"hist_layer("+name+").png")
        break
    
def model_hist_maker(model_name,subject):
    filters = model_name.modules
    body_model = [i for i in model_name.children()]
#     print(body_model)
    p = len(body_model)-5
    for i in range(p):
        layer1 = body_model[i][0]
        tensor = layer1.weight.data.cpu().numpy()
        plot_kernels(tensor, subject+"_"+str(i))


# In[ ]:


model = model.cuda()
loss_train,loss_test,acc_test,acc_train,ep = model_activate(model)


# In[ ]:



import pickle as pkl
pkl.dump(acc_test,open("acc_test_(3).pkl","wb"))
pkl.dump(acc_train,open("acc_train_(3).pkl","wb"))
pkl.dump(ep,open("ep.pkl_full_train_(3).pkl.","wb"))
pkl.dump(loss_test,open("loss_test_(3).pkl.","wb"))
pkl.dump(loss_train,open("loss_train_(3).pkl","wb"))


# In[ ]:


acc_plot(ep,acc_train,acc_test,"Acc_plot")

