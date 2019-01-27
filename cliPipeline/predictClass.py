
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
from keras import optimizers
from keras.preprocessing import image
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import random
import cv2
from keras.utils import to_categorical


# In[2]:


json_file = open('./Data/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("./Data/model.h5")
print("Loaded model from disk")

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'] )


# In[6]:

encoding = np.load("./Data/encodingDict.npy").item()

segmented_words = np.load("letters.npy")
for word in segmented_words:
	for letter in word:
	    img = np.expand_dims(letter,axis=0)
	    #print(img.shape)
	    cl = model.predict_classes(img/255)
	    cv2.imshow(encoding[str(cl[0])],letter)
	    cv2.waitKey(1000)
	    cv2.destroyAllWindows()
	    

