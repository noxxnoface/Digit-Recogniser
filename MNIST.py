import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Convolution2D
from keras.utils import np_utils

#Load Data
Xy_train = np.array(pd.read_csv(r'C:\Users\MBL.SRINIVAS\Documents\Data Science\Kaggle\MNIST\train.csv'))
X_test = np.array(pd.read_csv(r'C:\Users\MBL.SRINIVAS\Documents\Data Science\Kaggle\MNIST\test.csv'))
y_train = Xy_train[:,[0]]
X_train = np.delete(Xy_train,0,axis = 1)

#Visualize
"""
plt.subplot(121)
plt.imshow(X_train[0].reshape(28,28), cmap='gray')
plt.subplot(122)
plt.imshow(X_train[1].reshape(28,28))
plt.show()
"""

#Preprocess Input
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
 #Normalize
X_train /= 255
X_test /= 255

#One Hot Encode Outputs
y_train = np_utils.to_categorical(y_train)

#Model
 #Define
model = Sequential()
model.add(Convolution2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (3,3)))
model.add(Flatten())
model.add(Dense(120, activation = 'relu'))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
 #Compile
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
 #Fit
model.fit(X_train, y_train, epochs = 20, batch_size = 64,verbose = 1)
 #Predict
predict = np.argmax(model.predict(X_test), axis = 1)
#Saving the predictions to a csv file.
"""
p = pd.DataFrame()
p['ImageId'] = range(1,28001)
p['Label'] = predict
p.to_csv('predict2.csv', index = False)

"""
