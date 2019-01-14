## Code for importing the trained classifier into the OCR
## Author : Avadesh
## Date : Dec 21 2018

## Note: the imported classifier is by Anupam, please contact him for more querries

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
import pandas as pd

class OCRclassifier:

	def __init__(self, model_dir):
		# load json and create model
		json_file = open(model_dir + 'Results/model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		self.model = model_from_json(loaded_model_json)
		self.model.load_weights(model_dir + "Results/model.h5")
		# load weights into new model
		opt = optimizers.SGD(lr=0.01)
		self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
		self.indextoenglish = pd.read_csv(model_dir + 'dict.csv')['itrans']

	def classify(self, x_test):

		result = self.model.predict_classes(x_test)
		result_itrans = []
		# Converting to english labels using dict	
		for index in result: 
			result_itrans.append(self.indextoenglish[index]) 
	
		return result_itrans
		

	


