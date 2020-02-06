import cv2
import numpy as np
import random


def corner_detector(save_imgs = False):

    img = cv2.imread("static/image.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # FAST
    fast = cv2.FastFeatureDetector_create(threshold=35)
    keypoints = fast.detect(gray, None)
    img_fast = img.copy()
    for marker in keypoints:
        img_fast = cv2.drawMarker(img_fast, tuple(int(i) for i in marker.pt), color=(0, 0, 255), markerSize=10)
    if save_imgs:
        cv2.imwrite("static/result2.jpg", img_fast)

    # Harris
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    if save_imgs:
        cv2.imwrite("static/result1.jpg", img)

    return keypoints


def fit_line():

    img = cv2.imread("static/image.jpg")
    height = img.shape[0]
    print(img.shape)

    keypoints = corner_detector()

    dist_thresh = 10
    nr_inliers_min = 20
    best_fit = 999999
    line_y = 0
    for _ in range(1000):

        y = random.randint(1, height)

        loss = 0
        nr_inliers = 0
        for kp in keypoints:
            dist = abs(kp.pt[1] - y)
            # Inliers
            if dist < dist_thresh:
                loss += dist
                nr_inliers += 1

        if loss < best_fit and nr_inliers >= nr_inliers_min:
            best_fit = loss
            line_y = y
            print(loss, y)

    cv2.line(img, (0, line_y), (img.shape[1], line_y), (0, 0, 255))
    cv2.imwrite("static/result1.jpg", img)
