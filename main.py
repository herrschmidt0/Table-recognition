import numpy as np
import cv2
import argparse
from PIL import Image
from pytesseract import pytesseract as pyt
import helper as helper
import matplotlib.pyplot as plt
import skimage.transform as sktrans



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())
# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
original = cv2.imread(args["image"])

imageWidth = original.shape[1]
imageHeight = original.shape[0]


image_to_working_ratio, working_to_window_ratio = helper.resize(imageHeight, imageWidth)


# Resize to working size
image = cv2.resize(original, dsize = (int(imageWidth/image_to_working_ratio), int(imageHeight/image_to_working_ratio)))


'''
# Remove recongized characters
h, w, _ = image.shape

print pyt.image_to_string(Image.fromarray(image))

boxes = pyt.image_to_boxes(Image.fromarray(image))
boxes = boxes.split('\n')

for box in boxes:
	box = map(lambda x: int(x), box.split()[1:])
	if abs(box[0]-box[2])<w/80 and abs(box[1]-box[3])<h/80 and box[1]!=box[3] and box[0]!=box[2] :
		#cv2.rectangle(image,  (box[0], h-box[1]), (box[2], h-box[3]), (160,160,160), cv2.FILLED )
		image[h-box[3] : h-box[1], box[0]:box[2]] = [0,0,0] #cv2.blur(image[h-box[3] : h-box[1], box[0]:box[2]], (27,27))
	#cv2.rectangle(image,  (box[0], h-box[1]), (box[2], h-box[3]), (255,0,0), 1 )
'''



print "Stage 1 - Opencv grayscale + blur + threshold"

## Convert to grayscale
graySrc = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


## Gaussian Blur
kernel_size = 5
blurSrc = cv2.GaussianBlur(graySrc,(kernel_size, kernel_size),0)


## Threshold
#th, threshed = cv2.threshold(blurSrc, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#threshed = cv2.adaptiveThreshold(blurSrc, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3)
#threshed = cv2.adaptiveThreshold(blurSrc, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
threshed = cv2.adaptiveThreshold(blurSrc, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)	


imS = cv2.resize(threshed, (int(threshed.shape[1]/working_to_window_ratio), int(threshed.shape[0]/working_to_window_ratio)) )
#cv2.imshow("im-thresh", imS)


print "Stage 1.5 - Remove text"

#dst = cv2.cornerHarris(graySrc, 3, 5, 0.04)
#image[dst>0.003*dst.max()]=[0,0,255]

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create(threshold=35)

# find and draw the corner points
keypoints = fast.detect(graySrc, None)
image2 = cv2.drawKeypoints(threshed, keypoints, None, color=(255,0,0))


rectangles = helper.clusterCornerPoints([map(int,kp.pt) for kp in keypoints])


for rect in rectangles:
	cv2.drawContours(image2,[rect],0,(0,0,255),2)

imS = cv2.resize(image2, (int(image2.shape[1]/working_to_window_ratio), int(image2.shape[0]/working_to_window_ratio)) )
cv2.imshow("im-text", imS)

notext = threshed

for rect in rectangles:
	cv2.fillConvexPoly(notext, rect, color=(0,0,0))

imS = cv2.resize(notext, (int(notext.shape[1]/working_to_window_ratio), int(notext.shape[0]/working_to_window_ratio)) )
cv2.imshow("im-notext", imS)

print "Stage 1.8 - Detect lines"

	#sobely = cv2.Sobel(threshed, -1, 0, 2, ksize=3)


# Dilation & Erosion
kernel1 = np.ones((7,7), np.uint8)
kernel2 = np.ones((3,3), np.uint8)

dilation = cv2.dilate(notext, kernel1, iterations = 1)
erosion = cv2.erode(dilation, kernel2, iterations = 1)

## Thinning
thinned = cv2.ximgproc.thinning(notext, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
#thinned = cv2.ximgproc.thinning(threshed, thinningType=cv2.ximgproc.THINNING_GUOHALL)

imS = cv2.resize(notext, (int(thinned.shape[1]/working_to_window_ratio), int(thinned.shape[0]/working_to_window_ratio)) )
#cv2.imshow("im-thinned", imS)


# Run Line Detection on  image
line_image = np.copy(image) * 0  # creating a blank to draw lines on

# FastLineDetector
#detector = cv2.ximgproc.createFastLineDetector(_length_threshold=15)
#lines = detector.detect(notext)

#Hough Transform
version = 1

if version == 0:
	lines = cv2.HoughLinesP(notext, rho=1, theta=np.pi / 180 , threshold=70, minLineLength=40, maxLineGap=8)
	vectors = map(lambda x: helper.lineToVector(x[0][0], x[0][1], x[0][2], x[0][3]), lines)

elif version == 1:
	lines = sktrans.probabilistic_hough_line(notext, threshold=70, line_length=40, line_gap=8)
	vectors = map(lambda x: helper.lineToVector(x[0][0], x[0][1], x[1][0], x[1][1]), lines)


for p, v in vectors:
	p = map(int, p)
	v = map(int, v)
	cv2.line(line_image, (p[0],p[1]), (p[0]+v[0], p[1]+v[1]), (255,0,0), 2)

imS = cv2.resize(line_image, dsize = (int(line_image.shape[1]/working_to_window_ratio), int(line_image.shape[0]/working_to_window_ratio)))
#cv2.imshow("im-hough", imS)

print "Stage 2 - Extrapolation"

'''
helper.extrapolate(vectors)

additional_vectors = helper.extrapolate(vectors)

for p, v in additional_vectors:
	cv2.line(line_image, (p[0],p[1]), (p[0]+v[0], p[1]+v[1]), (0,255,0), 2)


vectors = vectors + additional_vectors
'''

#print "Stage 3 - calculating intersection points"

# Intersection points


'''
intersection_points = []
for i,r1 in enumerate(vectors):
	for j,r2 in enumerate(vectors):			
		if i<j:
			x = helper.getIntersectionPoint(r1, r2)
			if x is not None:
				intersection_points.append(x)
				cv2.circle(line_image, center=(int(x[0]),int(x[1])), radius=20, color=(0,1,255))

intersection_points = np.array(intersection_points)
'''

'''
print "Stage 4 - clustering intersection points"

#labels = helper.clusterIntersectionPoints(intersection_points)

colors = ['green', 'red', 'orange', 'yellow', 'black', 'blue', 'gray']

cluster_data = {}
for i, p in enumerate(intersection_points):
	cluster_data[labels[i]] = intersection_points[i]
	plt.plot(p[0], p[1], color=colors[labels[i]%7], marker='o',markersize=4)
	
print cluster_data.keys()

plt.show()
'''

imS = cv2.resize(line_image, dsize = (int(line_image.shape[1]/working_to_window_ratio), int(line_image.shape[0]/working_to_window_ratio)))
cv2.imshow("im-postproc", imS)

cv2.waitKey(0)
cv2.destroyAllWindows()