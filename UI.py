# This code is only for displaying, labeling and collecting the segmented letters.
# URL - http://127.0.0.1:5000/words/process
from flask import *
import cv2
import numpy as np
import os
import sys
import _pickle as pickle
import shutil

app = Flask(__name__,static_url_path = "/words", template_folder='UI/templates')
app.config['SECRET_KEY'] = 'oh_so_secret'

def array_to_files():
    image_array = np.load('letter.npy')
    for i in range(len(image_array)):
        cv2.imwrite('letters/' + str(i) + '.png', image_array[i])

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
    array_to_files()
    print('uploading letters....')
    session['start'] = 0
    limit = int(len(os.listdir('./letters')))
    if limit > 150:
        session['no_words'] = session['start'] + 150
    else:
        session['no_words'] = session['start'] + limit
    return redirect(url_for('dev_home'))

@app.route('/get/nextset',methods=['POST','GET'])
def get_next_set():
    limit = int(len(os.listdir('./letters')))
    session['start'] += 150
    if limit - session['start'] < 170 and limit - session['start'] > 0:
        session['no_words'] = limit
    elif (limit - session['start']) > 150:
        session['no_words'] = session['start'] + 150
    return redirect(url_for('dev_home'))

@app.route('/get/lastset',methods=['POST','GET'])
def go_back_one_set():

	value = request.form['value']
	if session['start'] > 0 and session['start'] >= 150:
		session['start'] -=  150
		session['no_words']  = session['start'] + 150
	return redirect(url_for('dev_home'))

@app.route('/dev_home')
def dev_home():
		return render_template('testing.html',no_words = session['no_words'],start = session['start'])	#start is th point from which words are thrown


@app.route('/get/label_data',methods=['POST','GET'])
def get_data():
	temp_label_array = request.form.getlist('label')
	label_array = []
	for l in range(len(temp_label_array)):
		label_array.append(temp_label_array[l])
	session['target_array'] = label_array
	return render_template('temp_testing.html',temp_label_array=temp_label_array,no_words = session['no_words'],label_array = label_array,start = session['start'])

@app.route('/get/confirmation',methods=['GET','POST'])
def get_confirmation():
	dataset = []
	target_array = session['target_array']
	index = 0
	for l in range(len(target_array)):
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

@app.route('/upload/<filename>')
def upload_letters(filename):
	return send_from_directory('letters',filename)

@app.route('/uploaded/<filename>')
def upload_page(filename):
	return send_from_directory('page',filename)


if __name__ == '__main__':
#	app.run(debug=True)
	app.run()
