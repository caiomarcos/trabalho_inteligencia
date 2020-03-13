# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:16:07 2019
drive end fault at 1750rpm, 0.021in, 12ksps and 48ksps
@author: caiom
"""
#%%
# Importing modules
import numpy as np
from keras.utils import np_utils
import scipy.io
import scipy.signal
import math
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from sklearn.model_selection import train_test_split
#%% Defining some constants to be used throughout
#number os data points per set
sampling_rate = 120000
#image width
img_w = 14
#image length
img_h = 14
#matrix used to hold final images
A = np.zeros((0, img_w, img_h))
#image length when unidimensional
img_length = img_w*img_h
#length of samples used to build image
N = img_length*2
#images in each class
samples_per_class = (2*sampling_rate)//N
#%% normal baseline at 48ksps
#import matlab file using scipy
normal_48 = scipy.io.loadmat('normal_1750.mat')
#get only the acc data points
normal_48 = normal_48['X099_DE_time']
#undersample by 4 to get 12ksps (resampled)
normal_48_rs = normal_48[::4]

#%% rolling element (ball) at 12ksps
#import matlab file using scipy
ball_12 = scipy.io.loadmat('b021')
#get only the acc data points
ball_12 = ball_12['X224_DE_time']
#resample to 48ksps
ball_12_rs = scipy.signal.resample(ball_12, sampling_rate)

#%% rolling element (ball) at 48ksps
#import matlab file using scipy
ball_48 = scipy.io.loadmat('b021_48')
#get only the acc data points
ball_48 = ball_48['X228_DE_time']
#undersample to 12ksps
ball_48_rs = ball_48[::4]

#%% inner race at 12ksps
#import matlab file using scipy
inner_race_12 = scipy.io.loadmat('ir021')
#get only the acc data points
inner_race_12 = inner_race_12['X211_DE_time']
#resample to 48ksps
inner_race_12_rs = scipy.signal.resample(inner_race_12, sampling_rate)

#%% inner race at 48ksps
#import matlab file using scipy
inner_race_48 = scipy.io.loadmat('ir021_48')
#get only the acc data points
inner_race_48 = inner_race_48['X215_DE_time']
#undersample to 12ksps
inner_race_48_rs = inner_race_48[::4]

#%% outer race at different angles at 12ksps
#import matlab file using scipy
outer_race_at3_12 = scipy.io.loadmat('or021at3')
#get only the acc data points
outer_race_at3_12 = outer_race_at3_12['X248_DE_time']
#resample to 48ksps
outer_race_at3_12_rs = scipy.signal.resample(outer_race_at3_12, sampling_rate)

#import matlab file using scipy
outer_race_at6_12 = scipy.io.loadmat('or021at6')
#get only the acc data points
outer_race_at6_12 = outer_race_at6_12['X236_DE_time']
#resample to 48ksps
outer_race_at6_12_rs = scipy.signal.resample(outer_race_at6_12, sampling_rate)

#import matlab file using scipy
outer_race_at12_12 = scipy.io.loadmat('or021at12')
#get only the acc data points
outer_race_at12_12 = outer_race_at12_12['X260_DE_time']
#resample to 48ksps
outer_race_at12_12_rs = scipy.signal.resample(outer_race_at12_12, sampling_rate)
#%% outer race at different angles at 48ksps
#import matlab file using scipy
outer_race_at3_48 = scipy.io.loadmat('or021at3_48')
#get only the acc data points
outer_race_at3_48 = outer_race_at3_48['X252_DE_time']
#undersample to 12ksps
outer_race_at3_48_rs = outer_race_at3_48[::4]

#import matlab file using scipy
outer_race_at6_48 = scipy.io.loadmat('or021at6_48')
#get only the acc data points
outer_race_at6_48 = outer_race_at6_48['X240_DE_time']
#undersample to 12ksps
outer_race_at6_48_rs = outer_race_at6_48[::4]

#import matlab file using scipy
outer_race_at12_48 = scipy.io.loadmat('or021at12_48')
#get only the acc data points
outer_race_at12_48 = outer_race_at12_48['X264_DE_time']
#undersample to 12ksps
outer_race_at12_48_rs = outer_race_at12_48[::4]
#%% build final data set for each class using both 48ksps and 12ksps (resampled accordingly)
normal = np.append(normal_48_rs, normal_48_rs)
inner_race = np.append(inner_race_48_rs, inner_race_12)
ball = np.append(ball_48_rs, ball_12)
outer_race_at3 = np.append(outer_race_at3_48_rs, outer_race_at3_12)
outer_race_at6 = np.append(outer_race_at6_48_rs, outer_race_at6_12)
outer_race_at12 = np.append(outer_race_at12_48_rs, outer_race_at12_12)
#%% some statistics
#df_48 = pd.DataFrame(inner_race_48)
#df_12 = pd.DataFrame(inner_race_12)
#avg_48 = sum(inner_race_48)/len(inner_race_48)
#avg_12 = sum(inner_race_12)/len(inner_race_12)
#avg_b48 = sum(ball_48)/len(ball_48)
#avg_b12 = sum(ball_12)/len(ball_12)
#ds1[0].plot(figsize = (600,6))
#plt.show()
#ds2[0].plot(figsize = (600,6))
#plt.show()
#rms_48 = np.sqrt(np.mean(df_48**2))
#rms_12 = np.sqrt(np.mean(df_12**2))
#%% build images for normal baseline

# fft for each sample
for i in range(0, samples_per_class):
    y = normal[(i*N):((i+1)*N)]
    yf = fft(y)
    ffty = 2.0/N * np.abs(yf[0:N//2])
    ffty[ffty == 0] = 8.82619e-05

    # B2LS for each sample
    S = []
    for x in range(0, len(ffty)):
        S.append(abs(math.log2(ffty[x]/img_length)))
    
    S = np.asarray(S)
    
    # normalize
    m = max(S)
    I = S/m
    
    # make vector into 2d image
    I.shape = (1, I.size//img_h, img_h)
    A = np.insert(A, 0, I, axis = 0)
    It = I
    It.shape = (It.size//img_h, img_h)
    # plot each image
    #plt.imshow(It, cmap="gray")
    #plt.show()

#plt.figure(figsize = (150,4))
#plt.plot(range(N//2), 2.0/N * np.abs(yf[0:N//2]))
#plt.grid()
#plt.show()
#plt.figure(figsize = (150,4))
#plt.plot(normal[(i*N):((i+1)*N)])
#plt.grid()
#plt.show()

# plot last image
It = I
It.shape = (It.size//img_h, img_h)
plt.imshow(It, cmap="gray")
plt.show()

#%% build images for fault outer race at 3 oclock

# for each sample
for i in range(0, samples_per_class):
    y = outer_race_at3[(i*N):((i+1)*N)]
    yf = fft(y)
    ffty = 2.0/N * np.abs(yf[0:N//2])
    ffty[ffty == 0] = 8.82619e-05

    # B2LS for each sample
    S = []
    for x in range(0, len(ffty)):
        S.append(abs(math.log2(ffty[x]/img_length)))
    S = np.asarray(S)
    
    # normalize
    m = max(S)
    I = S/m
    
    # make vector into 2d image
    I.shape = (1, I.size//img_h, img_h)
    A = np.insert(A, 0, I, axis = 0)
    It = I
    It.shape = (It.size//img_h, img_h)
    # plot each image
    #plt.imshow(It, cmap="gray")
    #plt.show()
    
#plt.figure(figsize = (150,4))
#plt.plot(range(N//2), 2.0/N * np.abs(yf[0:N//2]))
#plt.grid()
#plt.show()
#plt.figure(figsize = (150,4))
#plt.plot(normal[(i*N):((i+1)*N)])
#plt.grid()
#plt.show()
    
# plot last image
It = I
It.shape = (It.size//img_h, img_h)
plt.imshow(It, cmap="gray")
plt.show()
#%% build images for fault outer race at 6 oclock

# for each sample
for i in range(0, samples_per_class):
    y = outer_race_at6[(i*N):((i+1)*N)]
    yf = fft(y)
#plt.figure(figsize = (150,4))
#plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
#plt.grid()
#plt.show()
    ffty = 2.0/N * np.abs(yf[0:N//2])
    ffty[ffty == 0] = 8.82619e-05

    # B2LS for each sample
    S = []
    for x in range(0, len(ffty)):
        S.append(abs(math.log2(ffty[x]/img_length)))
    S = np.asarray(S)
    
    # normalize
    m = max(S)
    I = S/m
    
    # make vector into 2d image
    I.shape = (1, I.size//img_h, img_h)
    A = np.insert(A, 0, I, axis = 0)
    It = I
    It.shape = (It.size//img_h, img_h)
    
    # plot each image
    #plt.imshow(It, cmap="gray")
    #plt.show()
    
#plt.figure(figsize = (150,4))
#plt.plot(range(N//2), 2.0/N * np.abs(yf[0:N//2]))
#plt.grid()
#plt.show()
#plt.figure(figsize = (150,4))
#plt.plot(normal[(i*N):((i+1)*N)])
#plt.grid()
#plt.show()
    
# plot last image
It = I
It.shape = (It.size//img_h, img_h)
plt.imshow(It, cmap="gray")
plt.show()
#%% build images for fault outer race at 12 oclock

# for each sample
for i in range(0, samples_per_class):
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
    I.shape = (1, I.size//img_h, img_h)
    A = np.insert(A, 0, I, axis = 0)
    It = I
    It.shape = (It.size//img_h, img_h)
#    plt.imshow(It, cmap="gray")
#    plt.show()
    
#plt.figure(figsize = (150,4))
#plt.plot(range(N//2), 2.0/N * np.abs(yf[0:N//2]))
#plt.grid()
#plt.show()
#plt.figure(figsize = (150,4))
#plt.plot(normal[(i*N):((i+1)*N)])
#plt.grid()
#plt.show()
It = I
It.shape = (It.size//img_h, img_h)
plt.imshow(It, cmap="gray")
plt.show()
#%% fault at rolling element (ball)

for i in range(0, samples_per_class):
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
    I.shape = (1, I.size//img_h, img_h)
    A = np.insert(A, 0, I, axis = 0)
    It = I
    It.shape = (It.size//img_h, img_h)
#    plt.imshow(It, cmap="gray")
#    plt.show()
    
#plt.figure(figsize = (150,4))
#plt.plot(range(N//2), 2.0/N * np.abs(yf[0:N//2]))
#plt.grid()
#plt.show()
#plt.figure(figsize = (150,4))
#plt.plot(normal[(i*N):((i+1)*N)])
#plt.grid()
#plt.show()
It = I
It.shape = (It.size//img_h, img_h)
plt.imshow(It, cmap="gray")
plt.show()
#%% fault at inner race

for i in range(0, samples_per_class):
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
    I.shape = (1, I.size//img_h, img_h)
    A = np.insert(A, 0, I, axis = 0)
    It = I
    It.shape = (It.size//img_h, img_h)
#    plt.imshow(It, cmap="gray")
#    plt.show()
    
#plt.figure(figsize = (150,4))
#plt.plot(range(N//2), 2.0/N * np.abs(yf[0:N//2]))
#plt.grid()
#plt.show()
#plt.figure(figsize = (150,4))
#plt.plot(normal[(i*N):((i+1)*N)])
#plt.grid()
#plt.show()
It = I
It.shape = (It.size//img_h, img_h)
plt.imshow(It, cmap="gray")
plt.show()
#%%
A = A.reshape(A.shape[0], img_w, img_h, 1)
#%% labels
#normal and not normal
label1 = np.zeros(samples_per_class*6)
label1[0:(samples_per_class*4)] = 1
#normal and each fault
label2 = np.zeros(samples_per_class*6)

label2[0:samples_per_class] = 5
label2[samples_per_class:samples_per_class*2] = 4
label2[samples_per_class*2:samples_per_class*3] = 3
label2[samples_per_class*3:samples_per_class*4] = 2
label2[samples_per_class*4:samples_per_class*5] = 1
label2 = np_utils.to_categorical(label2, 6)

#%%
#X_train, X_test, y_train, y_test = train_test_split(A, label1, test_size=0.25)
X_train, X_test, y_train, y_test = train_test_split(A, label2, test_size=0.25)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#%%
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras import models
from keras.optimizers import Adam

model1 = models.Sequential()
model1.add(Conv2D(16, (3, 3), activation='relu', input_shape=(img_w,img_h,1)))
model1.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model1.add(Conv2D(32, (3, 3), activation='relu'))
model1.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model1.add(Flatten())
model1.add(Dense(128, activation='relu'))
model1.add(Dense(6, activation='softmax'))
#model.add(Dense(units=1, activation='sigmoid'))

model1.summary()

model1.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
#model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['binary_accuracy'])
#%%
model1.fit(X_train, y_train, batch_size=10, nb_epoch=100, validation_data=(X_test, y_test))
#model.fit(X_train, y_train, batch_size=10, nb_epoch=40, validation_data=(X_test, y_test))
#%%
#fazer previsoes com o data set que nao foi utilizado no treinamento
#normaliza conjunto das previsoes
predictions = model1.predict(X_test)
predictions = (predictions > 0.5)
#importa algumas coisas(?) do sklearn, verifica e imprime precisao das previsoes
from sklearn.metrics import confusion_matrix, accuracy_score
accuracy = accuracy_score(y_test, predictions)
print(accuracy)
#matriz de confusao (onde o classificador errou) e imprime
matrix = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
#matrix = confusion_matrix(y_test, predictions)
print(matrix)
#
result = model1.evaluate(X_test, y_test)
print(result)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
#confusion_matrix(y_test, predictions)
#%%
model2 = models.Sequential()
model2.add(Conv2D(2, (2, 2), activation='relu', input_shape=(img_w,img_h,1)))
model2.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model2.add(Flatten())
model2.add(Dense(8, activation='relu'))
model2.add(Dense(6, activation='softmax'))
#model.add(Dense(units=1, activation='sigmoid'))

model2.summary()

model2.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
#%%
model2.fit(X_train, y_train, batch_size=10, nb_epoch=100, validation_data=(X_test, y_test))
#model.fit(X_train, y_train, batch_size=10, nb_epoch=40, validation_data=(X_test, y_test))
#%%
#fazer previsoes com o data set que nao foi utilizado no treinamento
#normaliza conjunto das previsoes
predictions = model2.predict(X_test)
predictions = (predictions > 0.5)
#importa algumas coisas(?) do sklearn, verifica e imprime precisao das previsoes
from sklearn.metrics import confusion_matrix, accuracy_score
accuracy = accuracy_score(y_test, predictions)
print(accuracy)
#matriz de confusao (onde o classificador errou) e imprime
matrix = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
#matrix = confusion_matrix(y_test, predictions)
print(matrix)
#
result = model2.evaluate(X_test, y_test)
print(result)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
#confusion_matrix(y_test, predictions)