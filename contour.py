import sys
import cv2
import numpy as np

from PIL import Image
from tesserocr import PyTessBaseAPI

import xlsxwriter

TESSCONFIG_PATH = '/usr/share/tesseract-ocr/4.00/tessdata/'

# Read image
fname = sys.argv[1]
image = cv2.imread(fname)

# Create window
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 400,700)

# Grayscale, blurring, binarization
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel_size = 5
blurred = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

threshed = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)	

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
cv2.imshow("image", lines_img)
cv2.waitKey(0)


# Find contours
contours, hierarchy = cv2.findContours(lines_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_contours = np.zeros(image.shape)

table_cells = []
for cnt in contours:

	cnt_len = cv2.arcLength(cnt, True)
	cnt_poly = cv2.approxPolyDP(cnt, epsilon=0.01 * cnt_len, closed=True)
	#cnt = cv2.convexHull(cnt, False)

	area = cv2.contourArea(cnt_poly)
	#print(len(cnt))

	if len(cnt_poly)==4 and 1000 < area and cv2.isContourConvex(cnt_poly):
		cv2.drawContours(img_contours, [cnt_poly], 0, (0, 255, 0), 1)
		rect = cv2.boundingRect(cnt_poly)
		table_cells.append(rect)
		#print(rect)

# Show result
cv2.imshow("image", img_contours)
cv2.waitKey(0)

# Sort cells from top-to-bottom, left-to-right
table_cells.sort(key=lambda x: (x[1], x[0]) )

# Create excel file
workbook = xlsxwriter.Workbook('result.xlsx')
worksheet = workbook.add_worksheet()

with PyTessBaseAPI(path=TESSCONFIG_PATH) as api:
	api.SetImage(Image.open(fname))

	for i in range(len(table_cells)):
		x, y, w, h = table_cells[i]
		api.SetRectangle(x, y, w, h)
		text = api.GetUTF8Text()

		worksheet.write(i, 0, text)

workbook.close()