## Code for running the OCR Software
## Author : Avadesh Meduri
## Date : 16th Dec 2018
'''
Note: 
Please do not make changes to this file unless until absolutely neccesary.
All new algorithms developed for segmentation, classification etc.. should work without
having to make any changes to the standardized API used in this file
'''

from flask import *
import os
import sys
import shutil
from segmentation.ipsegmentation.pagesegmenter import pagesegmenter


app = Flask(__name__, template_folder='./UI/templates', static_url_path = "/words")
app.config['SECRET_KEY'] = 'oh_so_secret'


@app.route('/')
def home():
	return render_template('home.html')

@app.route('/image/get',methods=['POST','GET'])
def image_get():
	'''
	This function is to save the image uploaded by the user
	File is saved in tmp/page
	'''
	current_dir = os.getcwd()
	final_dir = os.path.join(current_dir, r'./tmp/page')
	if os.path.exists(final_dir):
		shutil.rmtree(final_dir)
		os.makedirs(final_dir)
	else:
		os.makedirs(final_dir)
	file = request.files['page']
	file.save('./tmp/page/image.jpg')
	return redirect(url_for('image_segment'))

@app.route('/image/segment/letters')
def image_segment():
	try:
		shutil.rmtree('./words')
		shutil.rmtree('./letters')
	except:
		print("Creating Directories: Words, letters")

	os.mkdir('./words')
	os.mkdir('./letters')
	
	img = './tmp/page/image.jpg'
	imagesegmenter = pagesegmenter(img)
	letter_array = imagesegmenter.get_letter_coordinates()
	#letter_array = imagesegmenter.get_word_coordinates()
	
	session['letter_array'] = letter_array
	return redirect(url_for('image_segmented_show'))

@app.route('/image/segment/show')
def image_segmented_show():
	return render_template('image_segmented.html',letter_array = session['letter_array'])




@app.route('/upload/words/<filename>')
def upload_words(filename):
	return send_from_directory('words',filename)

@app.route('/upload/letters/<filename>')
def upload_letters(filename):
	return send_from_directory('letters',filename)

################################################################################################


if __name__ == '__main__':
	app.run(debug=True)
