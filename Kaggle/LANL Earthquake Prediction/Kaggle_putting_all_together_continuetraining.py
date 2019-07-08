
# coding: utf-8

# # Kaggle - LANL Earthquake Prediction - putting all together

# ## Import packages

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
matplotlib.use('Agg')

from libs.clr_callback import *

# spectrogram with scipy
from scipy import signal
from IPython.display import clear_output
import json

from tqdm import tqdm_notebook as tqdm
import gc

import glob, os, sys

from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, CuDNNGRU, Dropout, Dense
from tensorflow.keras.layers import Flatten, TimeDistributed, AveragePooling1D, Embedding, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LeakyReLU

plt.ioff()

# get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# confirm TensorFlow sees the GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print('gpu name:',tf.test.gpu_device_name())


# Any results you write to the current directory are saved as output in
print(os.listdir("./data"))

input_dir = './data/train/'

#garbage collect
gc.collect()


# In[2]:

if 1:
  # SAFETY GUARD TO PREVENT FROM RUNNING BY MISTAKE
  print("Do you want to load the extracted features from the Training data and split it into training/validation? ")
  # if answer != 'yes': sys.exit(0)

  df_train2 = pd.read_csv(input_dir+'/df_training_features.csv').values
  # df_train2 = df_train2.reshape((df_train2.shape[0], df_train2.shape[1], 1))

  # y2 = pd.read_csv(input_dir+'/y.csv').values
  #print(y2[0][1])
  # y2 = y2.min(axis=1)
  # only the last time from failure value is relevant!
  y2 = pd.read_csv(input_dir+'/y.csv')['223'].values
  # y2 = y2.reshape((y2.shape[0], y2.shape[1], 1))
  # df_train.drop(['Unnamed: 0'])
  print(y2)
  # print(y2.mean(axis=1))
  np.savetxt('./data/test_miny.csv', y2, delimiter=",")

  X_train, X_test, y_train, y_test = train_test_split(df_train2, y2, test_size=0.1, random_state=42)
  X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
  X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
  print('X_train.shape',X_train.shape, 'X_test.shape',X_test.shape, 'y_train.shape',y_train.shape, 'y_test.shape',y_test.shape)
  #print(X_train.head(5))
  #print(y_train.head(5))


# In[3]:


# class PlotLearning(tf.keras.callbacks.Callback):
    # def on_train_begin(self, logs={}):
        # self.i = 0
        # self.x = []
        # self.losses = []
        # self.val_losses = []
        # self.acc = []
        # self.val_acc = []
        # self.fig = plt.figure()
        
        # self.logs = []

    # def on_epoch_end(self, epoch, logs={}):
        
        # self.logs.append(logs)
        # self.x.append(self.i)
        # self.losses.append(logs.get('loss'))
        # self.val_losses.append(logs.get('val_loss'))
        # self.acc.append(logs.get('acc'))
        # self.val_acc.append(logs.get('val_acc'))
        # self.i += 1
        # f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        
        # clear_output(wait=True)
        
        # ax1.set_yscale('log')
        # ax1.plot(self.x, self.losses, label="loss")
        # ax1.plot(self.x, self.val_losses, label="val_loss")
        # ax1.legend()
        
        # ax2.plot(self.x, self.acc, label="accuracy")
        # ax2.plot(self.x, self.val_acc, label="validation accuracy")
        # ax2.legend()
        
        # # plt.show();
        # plt.savefig('./data/train.png')

        
# plot_learning_loss = PlotLearning()

# updatable plot
# a minimal example (sort of)
pngname = ''
historyname = ''

class PlotLosses(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(epoch)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss",color='blue')
        plt.plot(self.x, self.val_losses, label="val_loss", color='red')
        if self.i == 0: plt.legend()
        # plt.show();
        plt.savefig(pngname)
        with open(historyname, 'w') as f:
          # f.write('epoch'+str(epoch))
          json.dump(self.logs, f)

        
gc.collect()


# # In[8]:


# SAFETY GUARD TO PREVENT FROM RUNNING BY MISTAKE
print("Do you want to load a saved model and weights? ")
# if answer != 'yes': sys.exit(0)

print('gpu name:',tf.test.gpu_device_name())

model_name = "triple_CuDNNGRU_reg"
epoch_init = 0
model = "tmp"


if os.path.isfile('./data/model/model-'+model_name+'.json'): 

  print("load model from json and weights from hdf5")
  json_file = open('./data/model/model-'+model_name+'.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  model = model_from_json(loaded_model_json)
  print("Loaded model "+model_name+" from disk")
  # load latest weights into new model
  list_of_files = glob.glob('./data/model/weights-'+model_name+"*.hdf5")
  if len(list_of_files)>0:
    latest_file = max(list_of_files, key=os.path.getmtime)
    model.load_weights(latest_file)
    epoch_init = int(latest_file.split('epoch')[1].split("-")[0])
    print("loading weights from file",latest_file,"starting from epoch",epoch_init)

elif model_name == "basic_CuDNNGRU":
  model = Sequential()
  model.add(CuDNNGRU(10, input_shape=(4096,1)))
  model.add(Dense(1, activation='linear'))

elif model_name == "basic_CuDNNGRU_reg":
  model = Sequential()
  model.add(CuDNNGRU(10, input_shape=(4096,1), kernel_regularizer=regularizers.l2(0.01),))
  model.add(Dense(1, activation='linear'))

elif model_name == "basic_big_CuDNNGRU_reg":
  model = Sequential()
  model.add(CuDNNGRU(100, input_shape=(4096,1), kernel_regularizer=regularizers.l2(0.01),))
  model.add(Dense(1, activation='linear'))

elif model_name == "double_CuDNNGRU_reg":
  model = Sequential()
  model.add(CuDNNGRU(50, input_shape=(4096,1), kernel_regularizer=regularizers.l2(0.01), return_sequences=True))
  model.add(CuDNNGRU(10, kernel_regularizer=regularizers.l2(0.01),))
  model.add(Dense(1, activation='linear'))

elif model_name == "triple_small_CuDNNGRU_reg":
  model = Sequential()
  model.add(CuDNNGRU(40, input_shape=(4096,1), kernel_regularizer=regularizers.l2(0.01), return_sequences=True))
  model.add(Dropout(0.3))
  model.add(CuDNNGRU(20, input_shape=(4096,1), kernel_regularizer=regularizers.l2(0.01), return_sequences=True))
  model.add(Dropout(0.3))
  model.add(CuDNNGRU(10, kernel_regularizer=regularizers.l2(0.01),))
  model.add(Dropout(0.3))
  model.add(Dense(1, activation='linear'))

elif model_name == "triple_CuDNNGRU_reg":
  model = Sequential()
  model.add(CuDNNGRU(128, input_shape=(4096,1), kernel_regularizer=regularizers.l2(0.01), return_sequences=True))
  model.add(Dropout(0.3))
  model.add(CuDNNGRU(64, kernel_regularizer=regularizers.l2(0.01), return_sequences=True))
  model.add(Dropout(0.3))
  model.add(CuDNNGRU(32, kernel_regularizer=regularizers.l2(0.01),))
  model.add(Dropout(0.3))
  model.add(Dense(16, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(1, activation='linear'))

model.summary()
# serialize model to JSON
if not os.path.isfile('./data/model/model-'+model_name+'.json'): 
  model_json = model.to_json()
  with open('./data/model/model-'+model_name+'.json', "w") as json_file:
    json_file.write(model_json)
    print('model '+model_name+' written to file')
 
# sys.exit(1)
  
# plot_losses = PlotLosses('./data/model/model-'+model_name+'-'+str(epoch_init)+'.png')
pngname = './data/model/model-'+model_name+'-'+str(epoch_init)+'.png'
historyname = './data/model/model-'+model_name+'-history-epoch'+str(epoch_init)+'.json'
plot_losses = PlotLosses()

model.compile(optimizer=Adam(lr=0.0001), loss="mae", 
                  # metrics=['accuracy']
                  )

filepath='./data/model/weights-'+model_name+"-epoch{epoch:02d}-val_loss{val_loss:.3f}.hdf5"
checkpoint = ModelCheckpoint(filepath, 
                             # monitor='val_acc', 
                             verbose=1, save_best_only=True
                             #, mode='max'
                            )

clr_triangular = CyclicLR(mode='triangular')
clr_triangular_exp = CyclicLR(mode='exp_range', gamma=0.99994, base_lr=0.0001, max_lr=0.0005, step_size=45288)


callbacks_list = [checkpoint,
                 # plot_learning_loss,
                 plot_losses,
                 # clr_triangular_exp
                 ]
# Fit the model

history = model.fit( 
                            X_train, 
                            y_train, 
                            epochs = 200, 
                            validation_data = (X_test, y_test),
                            batch_size = 64,
                            callbacks=callbacks_list,
                            initial_epoch = epoch_init
                        )

                        
# evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))



######################

  # model = Sequential()

  # model.add(CuDNNGRU(250, input_shape=(4096,1), kernel_regularizer=regularizers.l2(), return_sequences=True))
  # model.add(BatchNormalization())

  # # model.add(CuDNNGRU(500, kernel_regularizer=regularizers.l2(), return_sequences=True))
  # # model.add(BatchNormalization())

  # model.add(CuDNNGRU(100, kernel_regularizer=regularizers.l2()))
  # model.add(BatchNormalization())
  # model.add(Dropout(0.1))
  # # model.add(Dense(50, activation='relu'))
  # # model.add(Dense(50, activation='relu'))

  # model.add(Dense(100))
  # model.add(BatchNormalization())
  # model.add(LeakyReLU(alpha=0.05))
  # model.add(Dropout(0.5))

  # model.add(Dense(50))
  # model.add(BatchNormalization())
  # model.add(LeakyReLU(alpha=0.05))
  # model.add(Dropout(0.5))
  # model.add(Dense(1, activation='linear'))

  # model.summary()

######################

# SAFETY GUARD TO PREVENT FROM RUNNING BY MISTAKE
# print("Create a new model and train? ")
# if answer != 'yes': sys.exit(0)
# sys.exit(1)

# model = Sequential()

# model.add(CuDNNGRU(250, input_shape=(4096,1), kernel_regularizer=regularizers.l2(), return_sequences=True))
# model.add(BatchNormalization())

# # model.add(CuDNNGRU(500, kernel_regularizer=regularizers.l2(), return_sequences=True))
# # model.add(BatchNormalization())

# model.add(CuDNNGRU(100, kernel_regularizer=regularizers.l2()))
# model.add(BatchNormalization())
# model.add(Dropout(0.1))
# # model.add(Dense(50, activation='relu'))
# # model.add(Dense(50, activation='relu'))

# model.add(Dense(100))
# model.add(BatchNormalization())
# model.add(LeakyReLU(alpha=0.05))
# model.add(Dropout(0.5))

# model.add(Dense(50))
# model.add(BatchNormalization())
# model.add(LeakyReLU(alpha=0.05))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='linear'))

# model.summary()

# # Compile and fit model
# model.compile(optimizer=Adam(lr=0.0005), loss="mae", 
                  # # metrics=['accuracy']
                  # )

# # serialize model to JSON
# model_json = model.to_json()
# with open("./data/model/model-CuDNNGRU-2.json", "w") as json_file:
    # json_file.write(model_json)

# # checkpoint
# # filepath="./data/train/weights3-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
# filepath="./data/model/weights-CuDNNGRU-2-improvement-{epoch:02d}.hdf5"
# checkpoint = ModelCheckpoint(filepath, 
                             # # monitor='val_acc', 
                             # verbose=1, save_best_only=True
                             # #, mode='max'
                            # )
# callbacks_list = [checkpoint,
                 # # plot_learning_loss,
                 # plot_losses
                 # ]
# # Fit the model

# history = model.fit( 
                            # X_train, 
                            # y_train, 
                            # epochs = 50, 
                            # validation_data = (X_test, y_test),
                            # batch_size = 64,
                            # callbacks=callbacks_list
                        # )

######################################################################
                        
# model3 = Sequential()

# model3.add(CuDNNGRU(128, input_shape=(4096,1), activity_regularizer=regularizers.l2(), return_sequences=True))
# model3.add(BatchNormalization())

# model3.add(CuDNNGRU(128, activity_regularizer=regularizers.l2()))
# model3.add(BatchNormalization())

# # model3.add(CuDNNGRU(64, activity_regularizer=regularizers.l2()))
# # model3.add(BatchNormalization())
# # model3.add(Dropout(0.1))
# # model3.add(Dense(50, activation='relu'))
# # model3.add(Dense(50, activation='relu'))

# # model3.add(Dense(128))
# # model3.add(BatchNormalization())
# # model3.add(LeakyReLU(alpha=0.05))
# # model3.add(Dropout(0.5))

# model3.add(Dense(32))
# model3.add(BatchNormalization())
# model3.add(LeakyReLU(alpha=0.03))
# model3.add(Dropout(0.5))
# model3.add(Dense(1, activation='linear'))

# model3.summary()

# # Compile and fit model
# model3.compile(optimizer=Adam(lr=0.001, decay = 0.1), loss="mae"
                  # # metrics=['accuracy']
                  # )

# # serialize model to JSON
# model_json = model3.to_json()
# with open("./data/model/model-CuDNNGRU-4.json", "w") as json_file:
    # json_file.write(model_json)

# # checkpoint
# # filepath="./data/train/weights3-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
# filepath="./data/model/weights-CuDNNGRU-4-improvement-{epoch:02d}.hdf5"
# checkpoint = ModelCheckpoint(filepath, 
                             # # monitor='val_acc', 
                             # verbose=1, save_best_only=True
                             # #, mode='max'
                            # )
# callbacks_list = [checkpoint,
                 # # plot_learning_loss,
                 # plot_losses
                 # ]
# # Fit the model

# history = model3.fit( 
                            # X_train, 
                            # y_train, 
                            # epochs = 10, 
                            # validation_data = (X_test, y_test),
                            # batch_size = 64,
                            # callbacks=callbacks_list
                        # )


######################################################################


# loaded_model = Sequential()

# loaded_model.add(CuDNNGRU(100, input_shape=(4096,1), 
                        # # regularizers.l1_l2(l1=0.01, l2=0.01), 
                        # return_sequences=True))
# loaded_model.add(CuDNNGRU(100,return_sequences=True))
# loaded_model.add(CuDNNGRU(100))
# # loaded_model.add(Dense(50, activation='relu'))
# # loaded_model.add(Dense(50, activation='relu'))
# loaded_model.add(Dropout(0.5))
# loaded_model.add(Dense(10, activation='relu'))
# loaded_model.add(Dense(1, activation='linear'))

# loaded_model.summary()

# # Compile and fit model
# loaded_model.compile(optimizer=Adam(lr=0.0005), loss="mae", 
                  # # metrics=['accuracy']
                  # )

# loaded_model.load_weights('./data/model/weights-CuDNNGRU-improvement-16.hdf5')

# # checkpoint
# # filepath="./data/train/weights3-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
# filepath="./data/model/weights_reloaded-CuDNNGRU-improvement-{epoch:02d}.hdf5"
# checkpoint = ModelCheckpoint(filepath, 
                             # # monitor='val_acc', 
                             # verbose=1, save_best_only=True
                             # #, mode='max'
                            # )
# callbacks_list = [checkpoint,
                  # # plot_learning_loss,
                  # plot_losses
                 # ]
# # Fit the model

# history = loaded_model.fit( 
                            # X_train, 
                            # y_train, 
                            # epochs = 2, 
                            # validation_data = (X_test, y_test),
                            # batch_size = 64,
                            # # batch_size = 256,
                            # callbacks=callbacks_list
                        # )



# # In[ ]:


# import matplotlib.gridspec as gridspec

# Y_test_hat = loaded_model.predict(X_train)
# Y_test_hat = np.reshape(Y_test_hat, (1,np.product(Y_test_hat.shape)))

# residuals = np.subtract(Y_test_hat,y_train)

# print(Y_test_hat.shape, residuals.shape, y_train.shape)

# figure, axes1 = plt.subplots(figsize=(18,10))

# plt.scatter(y_train, residuals)
# plt.xlabel("y_train")
# plt.ylabel("Y_test_hat residuals")


# # In[ ]:


# import matplotlib.gridspec as gridspec

# Y_test_hat = loaded_model.predict(X_train)
# y_test1 = np.reshape(y_train, (np.product(y_train.shape)))

# Y_test_hat = np.reshape(Y_test_hat, (np.product(Y_test_hat.shape)))
# residuals = np.subtract(Y_test_hat,y_test1)

# print(Y_test_hat.shape, residuals.shape, y_test.shape)
# figure, axes1 = plt.subplots(figsize=(18,10))
# plt.hist2d(y_test1, residuals,100)
# plt.xlabel("y_train")
# plt.ylabel("Y_test_hat residuals")

