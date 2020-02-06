import os
import glob
from flask import Flask, request, flash, redirect, url_for
from werkzeug.utils import secure_filename

import preprocess
import line_fitter

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
	
	if request.method == 'GET':

		file_upload_html = \
		'''
		<h2>Upload new File</h2>
		<form method=post enctype=multipart/form-data>
		  <input type=file name=file>
		  <input type=submit value=Upload>
		</form>	
		'''

		if glob.glob('static/image.*'):
			file_upload_html += "<p>There is an image already uploaded.</p> <img src='static/image.jpg' width=150>"
		else:
			file_upload_html += "<p>There is no uploaded image yet.</p>"

		filelist = glob.glob(os.path.join('static', "result*"))
		for f in filelist:
			os.remove(f)

		return '''
		<!doctype html>
		<title>Table recognition</title>
		''' \
		+ file_upload_html + \
		''' 
		<h1>Methods</h1>
		
		<ul> 
			<li><a href='morph-linedetect-contour'>Table-cell detection based on morphology </a>
			</li>
			<li><a href='corners'> Harris vs FAST corner detection </a>
			</li>
			<li><a href='line-fit'> Fit horizontal lines to corner points (RANSAC) </a>
			</li>
		</ul>
		'''
	else:
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)

		file = request.files['file']

		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)

		if file:
			file.save(os.path.join("static/", "image.jpg")) # + file.filename.split('.')[1]))
			return redirect('/')




@app.route('/morph-linedetect-contour')
def morph_linedetect_contour():
	preprocess.recognize_tables()
	return show_all_imgs()


@app.route('/corners')
def corners():
	line_fitter.corner_detector(save_imgs=True)
	return show_all_imgs()


@app.route('/line-fit')
def line_fit():
	line_fitter.fit_line()
	return show_all_imgs()


def show_all_imgs():

	html_results = "<img src='static/image.jpg' width=500>"

	id = 1
	while os.path.isfile("static/result" + str(id) + ".jpg"):
		html_results += "<img src='static/result" + str(id) + ".jpg' width=500>"
		id += 1

	return html_results
