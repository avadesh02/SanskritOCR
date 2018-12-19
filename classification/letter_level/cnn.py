#importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils import plot_model
from keras import optimizers
import matplotlib.pyplot as plt
#from PIL import Image
import numpy as np
#import cv2
import os
np.random.seed(1024)
'''
#code to read images as numpy arrays and save them in files
images_x = []
images_y = []
for filename in os.listdir("path/to/images"):
	images_x.append(cv2.imread(filename))
	images_y.append(int(filename.split("_")[0]))
data_X = np.asarray(images_x)
data_Y = np.asarray(images_y)
np.save("Xdata",data_X)
np.save("Ydata",data_Y)
'''
'''
for i in range(0,10):
	print(data_Y[i])
	Image.fromarray(data_X[i],'RGB').show()
'''	
data_X = np.load("./Data/Xdata.npy")
data_Y = np.load("./Data/Ydata.npy")
data_Y = np.eye(np.max(data_Y)+1)[data_Y]

#initialising the CNN
classifier = Sequential()

# Convolution Layer
classifier.add(Conv2D(4,(3,3),input_shape = (32,32,3), activation='relu',padding='same',kernel_initializer='glorot_normal'))

# Convolution Layer
classifier.add(Conv2D(4,(3,3), activation='relu',padding='same',kernel_initializer='glorot_normal'))

# Convolution Layer
classifier.add(Conv2D(4,(3,3),activation='relu',padding='same',kernel_initializer='glorot_normal'))

#Pooling Layer
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Convolution Layer
classifier.add(Conv2D(8,(3,3), activation='relu',padding='same',kernel_initializer='glorot_normal'))

# Convolution Layer
classifier.add(Conv2D(8,(3,3), activation='relu',padding='same',kernel_initializer='glorot_normal'))

# Convolution Layer
classifier.add(Conv2D(8,(3,3), activation='relu',padding='same',kernel_initializer='glorot_normal'))

#Pooling Layer
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Flatten layer
classifier.add(Flatten())

#Fully Connected Layer
#classifier.add(Dense(units = 2048,activation = 'relu'))
classifier.add(Dense(units = 1024,activation = 'relu'))
classifier.add(Dense(units = 600,activation = 'softmax'))

print(classifier.summary())

#compiling the CNN
opt = optimizers.SGD(lr=0.01)
classifier.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'] )			#need to think about loss function



history = classifier.fit(x=data_X,y=data_Y,batch_size=128,epochs=50,validation_split=0.01)

plot_model(classifier, to_file='./Results/Final_model.png')

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


#to save model to disk
# serialize model to JSON
model_json = classifier.to_json()
with open("./Results/Final_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("./Results/Final_model.h5")
print("Saved model to disk")


'''
#to load model form json file
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data

loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
'''


