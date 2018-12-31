import numpy as np
from keras.engine import  Model
from keras.layers import Input
import h5py
from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Dense
import keras
import scipy.io as io
from keras.datasets import cifar10
import cv2

nb_class = 10
hidden_dim = 512

def one_hot(x,n):    
    x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x),n))
    o_h[np.arange(len(x)),x] = 1
    return o_h

def createVGGModel():
    vgg_model_conv = ResNet50(include_top=False, input_shape=(32,32,3), pooling='avg') # pooling: None, avg or max
    x = Dense(256, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu')(vgg_model_conv.output)
    x = Dense(nb_class, activation='softmax')(x)
    model = Model(input = vgg_model_conv.input, output = x)

    for layer_num in range(48):
        model.layers[layer_num].trainable = False
    return model

'''h5f = h5py.File('Data/TrainingdataVGG16WHPF.h5','r')
TrainData = h5f['dataset_1'][:]
h5f.close()

print(TrainData.shape)'''

#TrainData = np.swapaxes(TrainData, 2,3) 
#TrainData = np.swapaxes(TrainData, 1,2)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train_new = np.arange(7526400000)
x_train_new = x_train_new.reshape(50000, 224, 224, 3)

x_train = np.array(x_train,dtype=np.float32)
x_train = x_train/255.0
x_train = np.swapaxes(x_train, 1,2) 
x_train = np.swapaxes(x_train, 2,3)
for i in range(50000):
     qqq = cv2.resize(x_train[i,:,:,:],(224,224))
     x_train_new[i,:,:,:] = qqq

x_test = np.array(x_test,dtype=np.float32)
x_test = x_test/255.0
x_test = np.swapaxes(x_test, 1,2) 
x_test = np.swapaxes(x_test, 2,3)

y_train = y_train[:,0]
y_train = one_hot(y_train,10)

y_test = y_test[:,0]
y_test = one_hot(y_test,10)


model = createVGGModel()

epochs = 50
learning_rate = 0.00001
decay_rate = learning_rate / epochs
momentum = 0.8
adam = keras.optimizers.Adam(lr=learning_rate)
model.compile(optimizer=adam, loss= keras.losses.categorical_crossentropy, metrics=['accuracy'])
model.fit(x_train,y_train, nb_epoch=epochs,batch_size=128, shuffle=True, verbose=1, validation_split=0.15)

model.save('Data/ResNet50Finetune.h5')