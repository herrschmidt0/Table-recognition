import os
import glob
from flask import Flask, request, flash, redirect, url_for
from werkzeug.utils import secure_filename

import contour

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

		return '''
		<!doctype html>
		<title>Table recognition</title>
		''' \
		+ file_upload_html + \
		''' 
		<h1>Methods</h1>
		
		<ul> 
			<li><a href='morph-linedetect-contour'>Contour finding based table-cell detection 
				(with line filtering using morphology)</a>
			</li>
			<li><a href='simple-contour'>Contour finding based table-cell detection 
				(without line detection)</a>
			</li>
			<li><a href='simple-contour-7p'>Contour finding based table-cell detection 
				(without line detection + heptagon shaped cells)</a>
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
	contour.contour(True, 4)
	return show_all_imgs()


@app.route('/simple-contour')
def simple_contour():
	contour.contour(False, 4)
	return show_all_imgs()


@app.route('/simple-contour-7p')
def simple_contour_7p():
	contour.contour(False, 7)
	return show_all_imgs()


def show_all_imgs():

	html_results = "<img src='static/image.jpg' width=500>"

	id = 1
	while os.path.isfile("static/result" + str(id) + ".png"):
		html_results += "<img src='static/result" + str(id) + ".png' width=500>"
		id += 1

	return html_results
