# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:16:07 2019
drive end fault at 1750rpm, 0.021in, 12ksps
@author: caiom
"""
#%%
# Imports
import numpy as np
import pandas as pd
from keras.utils import np_utils
import scipy.io
import scipy.signal
import math
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from sklearn.model_selection import train_test_split
#%%
#normal_sampling_rate = 48000
#og_sampling_rate = 12000
#downsample_rate = 6000
#%% import normal baseline at 48kHz
normal = scipy.io.loadmat('normal_1750.mat')
normal_1 = normal['X098_DE_time']
normal_2 = normal['X099_DE_time']
#%% rolling element (ball)
ball = scipy.io.loadmat('b021')
ball = ball['X224_DE_time']
#%% inner race
inner_race = scipy.io.loadmat('ir021')
inner_race = inner_race['X211_DE_time']
#%% outer race at different angles
outer_race_at3 = scipy.io.loadmat('or021at3')
outer_race_at3 = outer_race_at3['X248_DE_time']

outer_race_at6 = scipy.io.loadmat('or021at6')
outer_race_at6 = outer_race_at6['X236_DE_time']

outer_race_at12 = scipy.io.loadmat('or021at12')
outer_race_at12 = outer_race_at12['X260_DE_time']
#%%
inner_race_rs = scipy.signal.resample(inner_race, 480000)
#%%

ds1 = pd.DataFrame(inner_race)
ds2 = pd.DataFrame(inner_race_rs)
ds1[0].plot(figsize = (600,6))
plt.show()
ds2[0].plot(figsize = (600,6))
plt.show()
#%% downsample to 12kHz (sample rate I'm going to use)
normal_1 = normal_1[::4]
normal_2 = normal_2[::4]

#%%
img_size = 28
A = np.zeros((0, img_size, img_size))
img_length = img_size*img_size
N = img_length*2
samples = 483903//N
#%% fft

for i in range(0, samples):
    y = normal_2[(i*N):((i+1)*N)]
    yf = fft(y)
#plt.figure(figsize = (150,4))
#plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
#plt.grid()
#plt.show()
    ffty = 2.0/N * np.abs(yf[0:N//2])
    ffty[ffty == 0] = 8.82619e-05

# B2LS
    S = []
    
    for x in range(0, len(ffty)):
        S.append(abs(math.log2(ffty[x]/img_length)))
    S = np.asarray(S)
    m = max(S)
    I = S/m
    I.shape = (1, I.size//img_size, img_size)
    A = np.insert(A, 0, I, axis = 0)
#    plt.imshow(I, cmap="gray")
#    plt.show()
#%% fft

for i in range(0, samples):
    y = normal_1[(i*N):((i+1)*N)]
    yf = fft(y)
#plt.figure(figsize = (150,4))
#plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
#plt.grid()
#plt.show()
    ffty = 2.0/N * np.abs(yf[0:N//2])
    ffty[ffty == 0] = 8.82619e-05

# B2LS
    S = []
    
    for x in range(0, len(ffty)):
        S.append(abs(math.log2(ffty[x]/img_length)))
    S = np.asarray(S)
    m = max(S)
    I = S/m
    I.shape = (1, I.size//img_size, img_size)
    A = np.insert(A, 0, I, axis = 0)

#%% fft
for i in range(0, samples):
    y = outer_race_at3[(i*N):((i+1)*N)]
    yf = fft(y)
#plt.figure(figsize = (150,4))
#plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
#plt.grid()
#plt.show()
    ffty = 2.0/N * np.abs(yf[0:N//2])
    ffty[ffty == 0] = 8.82619e-05

# B2LS
    S = []
    
    for x in range(0, len(ffty)):
        S.append(abs(math.log2(ffty[x]/img_length)))
    S = np.asarray(S)
    m = max(S)
    I = S/m
    It = I
    It.shape = (It.size//img_size, img_size)
    plt.imshow(It, cmap="gray")
    plt.show()
    I.shape = (1, I.size//img_size, img_size)
    A = np.insert(A, 0, I, axis = 0)

#%% fft

for i in range(0, samples):
    y = outer_race_at6[(i*N):((i+1)*N)]
    yf = fft(y)
#plt.figure(figsize = (150,4))
#plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
#plt.grid()
#plt.show()
    ffty = 2.0/N * np.abs(yf[0:N//2])
    ffty[ffty == 0] = 8.82619e-05

# B2LS
    S = []
    
    for x in range(0, len(ffty)):
        S.append(abs(math.log2(ffty[x]/img_length)))
    S = np.asarray(S)
    m = max(S)
    I = S/m
    I.shape = (1, I.size//img_size, img_size)
    A = np.insert(A, 0, I, axis = 0)
#    plt.imshow(I, cmap="gray")
#    plt.show()
#%% fft

for i in range(0, samples):
    y = outer_race_at12[(i*N):((i+1)*N)]
    yf = fft(y)
#plt.figure(figsize = (150,4))
#plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
#plt.grid()
#plt.show()
    ffty = 2.0/N * np.abs(yf[0:N//2])
    ffty[ffty == 0] = 8.82619e-05

# B2LS
    S = []
    
    for x in range(0, len(ffty)):
        S.append(abs(math.log2(ffty[x]/img_length)))
    S = np.asarray(S)
    m = max(S)
    I = S/m
    I.shape = (1, I.size//img_size, img_size)
    A = np.insert(A, 0, I, axis = 0)
#    plt.imshow(I, cmap="gray")
#    plt.show()

#%% fft

for i in range(0, samples):
    y = ball[(i*N):((i+1)*N)]
    yf = fft(y)
#plt.figure(figsize = (150,4))
#plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
#plt.grid()
#plt.show()
    ffty = 2.0/N * np.abs(yf[0:N//2])
    ffty[ffty == 0] = 8.82619e-05

# B2LS
    S = []
    
    for x in range(0, len(ffty)):
        S.append(abs(math.log2(ffty[x]/img_length)))
    S = np.asarray(S)
    m = max(S)
    I = S/m
    I.shape = (1, I.size//img_size, img_size)
    A = np.insert(A, 0, I, axis = 0)
#    plt.imshow(I, cmap="gray")
#    plt.show()
#%% fft

for i in range(0, samples):
    y = inner_race[(i*N):((i+1)*N)]
    yf = fft(y)
#plt.figure(figsize = (150,4))
#plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
#plt.grid()
#plt.show()
    ffty = 2.0/N * np.abs(yf[0:N//2])
    ffty[ffty == 0] = 8.82619e-05

# B2LS
    S = []
    
    for x in range(0, len(ffty)):
        S.append(abs(math.log2(ffty[x]/img_length)))
    S = np.asarray(S)
    m = max(S)
    I = S/m
    I.shape = (1, I.size//img_size, img_size)
    A = np.insert(A, 0, I, axis = 0)
#    plt.imshow(I, cmap="gray")
#    plt.show()
#%%
A = A.reshape(A.shape[0], 28, 28, 1)
#%%
label1 = np.zeros(539)
label1[0:395] = 1
label1[395:539] = 0

label2 = np.zeros(539)
label2[0:77] = 5
label2[77:154] = 4
label2[154:231] = 3
label2[231:308] = 2
label2[308:385] = 1
label2 = np_utils.to_categorical(label2, 6)

#%%
X_train, X_test, y_train, y_test = train_test_split(A, label2, test_size=0.25)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#%%
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras import models
from keras.optimizers import Adam

model = models.Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
#%%
model.fit(X_train, y_train, batch_size=1, nb_epoch=100, validation_data=(X_test, y_test))

#%%
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(28,28,1)))
model.add(Activation('relu'))
BatchNormalization(axis=-1)
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

BatchNormalization(axis=-1)
model.add(Conv2D(64,(3, 3)))
model.add(Activation('relu'))
BatchNormalization(axis=-1)
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
# Fully connected layer

BatchNormalization()
model.add(Dense(512))
model.add(Activation('relu'))
BatchNormalization()
model.add(Dropout(0.2))
model.add(Dense(6))

# model.add(Convolution2D(10,3,3, border_mode='same'))
# model.add(GlobalAveragePooling2D())
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
#%%
gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)

test_gen = ImageDataGenerator()

train_generator = gen.flow(X_train, y_train, batch_size=64)
test_generator = test_gen.flow(X_test, y_test, batch_size=64)

# model.fit(X_train, Y_train, batch_size=128, nb_epoch=1, validation_data=(X_test, Y_test))

model.fit_generator(train_generator, steps_per_epoch=60000//64, epochs=5, 
                    validation_data=test_generator, validation_steps=10000//64)