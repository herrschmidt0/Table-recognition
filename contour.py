import sys
import cv2
import numpy as np
import functools

from PIL import Image
from tesserocr import PyTessBaseAPI

import xlsxwriter

TESSCONFIG_PATH = '/usr/share/tesseract-ocr/4.00/tessdata/'


def contour(use_morph, max_poly_p):

	# Read image
	fname = "static/image.jpg"
	image = cv2.imread(fname)
	result_id = 0

	# Create window
	#cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	#cv2.resizeWindow('image', 400,700)

	# Grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Histogram equalization
	equ = cv2.equalizeHist(gray)
	#gray = equ

	# Blurring + binarization
	kernel_size = 5
	blurred = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

	threshed = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)

	lines_img = threshed
	if use_morph is True:
		# Structuring elements, morphology
		kernel_length = np.array(image).shape[1] // 80
		vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
		horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

		# Morphological operation to detect vertical lines from an image
		img_temp1 = cv2.erode(threshed, vertical_kernel, iterations=3)
		vertical_lines_img = cv2.dilate(img_temp1, vertical_kernel, iterations=6)

		# Morphological operation to detect horizontal lines from an image
		img_temp2 = cv2.erode(threshed, horiz_kernel, iterations=3)
		horizontal_lines_img = cv2.dilate(img_temp2, horiz_kernel, iterations=6)

		lines_img = vertical_lines_img + horizontal_lines_img

	# Find contours
	contours, hierarchy = cv2.findContours(lines_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	img_contours = np.zeros(image.shape)
	#print (hierarchy)

	table_cells = []
	for cnt in contours:

		cnt_len = cv2.arcLength(cnt, True)
		cnt_poly = cv2.approxPolyDP(cnt, epsilon=0.01 * cnt_len, closed=True)
		#cnt = cv2.convexHull(cnt, False)

		area = cv2.contourArea(cnt_poly)
		#print(len(cnt))

		if 4 <= len(cnt_poly) <= max_poly_p and 1000 < area and cv2.isContourConvex(cnt_poly):
			cv2.drawContours(img_contours, [cnt_poly], 0, (0, 255, 0), 1)
			rect = cv2.boundingRect(cnt_poly)
			table_cells.append(rect)
			#print(rect)
		else:
			cv2.drawContours(img_contours, [cnt_poly], 0, (255, 0, 0), 1)

	result_id += 1
	cv2.imwrite("static/result" + str(result_id) + ".png", img_contours)


# Show result
#cv2.imshow("image", img_contours)
#cv2.waitKey(0)

'''
# Sort cells from top-to-bottom, left-to-right
def comp(cell1, cell2):
	tr = 10 # same line difference threshold
	if abs(cell1[1] - cell2[1]) > tr:
		if cell1[1] < cell2[1]:
			return -1
		else:
			return 1
	else:
		if cell1[0] < cell2[0]:
			return -1
		else:
			return 1
		
#table_cells = sorted(table_cells, key=functools.cmp_to_key(comp))
table_cells.sort(key=functools.cmp_to_key(comp))
#print(table_cells)

# Create excel file
workbook = xlsxwriter.Workbook('result.xlsx')
worksheet = workbook.add_worksheet()

# Extract rows and columns 
with PyTessBaseAPI(path=TESSCONFIG_PATH) as api:
	api.SetImage(Image.open(fname))

	row = 0
	col = 0
	y_diff_thresh = 10

	x1, y1, w, h = table_cells[1]
	api.SetRectangle(x1, y1, w, h)
	text = api.GetUTF8Text()

	worksheet.write(0, 0, text)

	for cell in table_cells[2:]:
		x2, y2, w, h = cell
		api.SetRectangle(x2, y2, w, h)
		text = api.GetUTF8Text()

		if abs(y1-y2) < y_diff_thresh:
			col += 1
		else:
			row += 1
			col = 0

		worksheet.write(row, col, text)
		x1, y1 = x2, y2

workbook.close()

'''