path = "/home/mohit/Desktop/Nptel/Week1_Deep Learning For Visual Learning/cifar10/cifar-10-batches-py/"
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt

def extractImagesAndLabels(path, file):
    f = open(path+file, 'rb')
    dicte = pickle.load(f,encoding='bytes')
    images = dicte[b'data']
    Matrix=[]
    for i in images:
        ingle_img_reshaped = np.transpose(np.reshape(i,(3, 32,32)), (1,2,0))
        #x=np.dot(ingle_img_reshaped[...,:3], [0.299, 0.587, 0.114])
        x= ingle_img_reshaped.flatten()
        Matrix.append(x)
    #images = np.transpose(images, (1,2,0))
   
    Matrix = np.array(Matrix)
    labels = dicte[b'labels']
    labels = np.array(labels)
    #print labels.shape
    return Matrix, labels
 
#     labels = dict['labels']

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

