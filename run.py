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

app = Flask(__name__, template_folder='./UI/templates')

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






#################################################################################################

if __name__ == '__main__':
	app.run(debug=True)
