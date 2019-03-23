
from flask import *
import cv2
import numpy as np
import os
import sys
import _pickle as pickle
import shutil

app = Flask(__name__,static_url_path = "/words", template_folder='ORC-UI/TemplatesAndCSS')
app.config['SECRET_KEY'] = 'oh_so_secret'

def array_to_files():
    image_array = np.load('letter.npy')
    for i in range(len(image_array)):
        cv2.imwrite('letters/' + str(i) + '.png', image_array[i])

array_to_files()

@app.route('/words/process')
def segment_words():
	current_dir = os.getcwd()
	final_dir = os.path.join(current_dir, r'letters')
	if os.path.exists(final_dir):
		shutil.rmtree(final_dir)
		os.makedirs(final_dir)
	else:
		os.makedirs(final_dir)
	print('letters have been stored...')
	print('uploading letters....')
	session['start'] = 0
	session['flag'] = 0
	session['no_words'] = session['start'] + 250
	return redirect(url_for('dev_home'))

@app.route('/get/nextset',methods=['POST','GET'])
def get_next_set():
	value = 250
	limit = int(len(os.listdir('./letters')))
#	value = request.form['value']
#	print(value)
	if limit -session['start'] < 260 and limit -session['start'] > 0:
		print('if statment worked')
		print(session['flag'])
		session['start'] += 250
		session['no_words'] = limit-1
	elif limit - session['start'] > 250:
		print('elif is working')
		session['start'] += 250
		session['no_words'] = session['start'] + 250
	return redirect(url_for('dev_home'))

@app.route('/get/lastset',methods=['POST','GET'])
def go_back_one_set():

	value = request.form['value']
#	print(value)
	if session['start'] > 0:
		session['start'] -=  250
		session['no_words']  = session['start'] + 250

	return redirect(url_for('dev_home'))

@app.route('/dev_home')
def dev_home():
		return render_template('testing.html',no_words = session['no_words'],start = session['start'])	#start is th point from which words are thrown


@app.route('/get/label_data',methods=['POST','GET'])
def get_data():
	temp_label_array = request.form.getlist('label')
	#print(session['tmp'])
	label_array = []
	for l in range(session['start'],session['no_words']):
		label_array.append(temp_label_array[l])
	#print(label_array)
	#print(no_letter_array)
	session['target_array'] = label_array
	return render_template('temp_testing.html',temp_label_array=temp_label_array,no_words = session['no_words'],label_array = label_array,start = session['start'])

@app.route('/get/confirmation',methods=['GET','POST'])
def get_confirmation():
	dataset = []
	target_array = session['target_array']
	index = 0
	for l in range(session['start'],session['no_words']):
		img = 'letters/' + str(l) + '.png'
		letter_image = cv2.imread(img)
		dataset.append(letter_image)
		dataset.append(target_array[l])

	current_dir = os.getcwd()
	final_dir = os.path.join(current_dir, r'database')
	if not os.path.exists(final_dir):
		os.makedirs(final_dir)
	i = os.listdir('./database')
	print(i)
	name_new_database = final_dir + '/' + str(len(i)) + '.p'

	pickle.dump( dataset ,open(name_new_database,'wb'))
	return redirect(url_for('get_next_set'))
"""
@app.route('/prediction/image/get',methods=['POST','GET'])
def get_image_prediction():
	current_dir = os.getcwd()
	final_dir = os.path.join(current_dir, r'page')
	if os.path.exists(final_dir):
		shutil.rmtree(final_dir)
		os.makedirs(final_dir)
	else:
		os.makedirs(final_dir)
	file = request.files['page']
	file.save('page/image.jpg')
	return redirect(url_for('segment_image_prediction'))

@app.route('/prediction/image/process')
def segment_image_prediction():
	img = 'page/image.jpg'
	current_dir = os.getcwd()
	final_dir = os.path.join(current_dir, r'words')
	if os.path.exists(final_dir):
		shutil.rmtree(final_dir)
		os.makedirs(final_dir)
	else:
		os.makedirs(final_dir)

	image = word_finder(img)
	image.segment_page_into_words()

	return redirect(url_for('segment_words_prediction'))

@app.route('/prediction/words/process')
def segment_words_prediction():
	current_dir = os.getcwd()
	final_dir = os.path.join(current_dir, r'letters')
	if os.path.exists(final_dir):
		shutil.rmtree(final_dir)
		os.makedirs(final_dir)
	else:
		os.makedirs(final_dir)
	word_array = os.listdir('./words')
	print('finding letters and storing.....')
	label_array = store_f(0,len(word_array)-2)
	session['label_array'] = label_array
	print('letters have been stored...')
	print('uploading letters....')
	session['start'] = 0
	session['flag'] = 0
	session['no_words'] = session['start'] + 25
	return redirect(url_for('show_predictions'))

from test_model import make_prediction,predictor

@app.route('/show/predictions',methods = ['GET','POST'])
def show_predictions():
	index_array = []
	prediction_array = []
	letters = os.listdir('./letters')
	print('predicitng output')
	#for i in range(session['start'],session['no_words']):
	for i in range(0,15):
		prediction_array.append([])
		for j in range(session['label_array'][i]):
			let = './letters/' + str(i) + str(j) +  '.png'
			print(let)
			img = cv2.imread(let)
			pred = make_prediction(img)
			pred = predictor(img)
			prediction_array[i].append(pred)
	return render_template('prediction.html',label_array = session['label_array'],prediction_array = prediction_array,start = 0,end = 15)
"""

@app.route('/upload/<filename>')
def upload_letters(filename):
	return send_from_directory('letters',filename)

@app.route('/uploaded/<filename>')
def upload_page(filename):
	return send_from_directory('page',filename)


if __name__ == '__main__':
#	app.run(debug=True)
	app.run()
