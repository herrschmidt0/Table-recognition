import numpy as np
import random
from sklearn.cluster import DBSCAN
from sklearn import linear_model
import matplotlib.pyplot as plt
import hdbscan
import cv2

def extrapolate(vectors):

    results = []

    for i, r1 in enumerate(vectors):
        for j, r2 in enumerate(vectors):

            if (j > i):


                # Are the line segments on the same line, that need to be connected?
                vec = connectCollinearLines(r1, r2)
              
                if vec is not None:         
                    results.append(vec)

                # Are the line segments perpendicular, that need to be connected?          
                vec = connectPerpendicularLines(r1, r2)

                if vec is not None:
                    results.append(vec)

    return results


#####################################################

def lineToVector(x1, y1, x2, y2):

    if x1 < x2:
        p = np.array([x1, y1])
        v = np.array([x2-x1, y2-y1])
    elif x1 > x2:
        p = np.array([x2, y2])
        v = np.array([x1-x2, y1-y2])
    elif x1 == x2 and y1 < y2:
        p = np.array([x1, y1])
        v = np.array([0, y2-y1])
    elif x1 == x2 and y1 > y2:
        p = np.array([x2, y2])
        v = np.array([0, y1-y2])

    return (p,v)


def connectCollinearLines(r1, r2):
  
    # Are they parallel?
    if not areParallel(r1, r2):
        return None

    # Are they collinear and close to each other?
    v1, v2 = r1[1], r2[1]
    p1, p2 = r1[0], r2[0]

    norm_limit = 200

    if p1[0] <= p2[0]:

        r12 = (p1+v1, p2-(p1+v1))
       

        # Are they too far apart?
        if np.linalg.norm(r12[1]) > norm_limit:
            return None

        # Are they on the same line?
        if areParallel(r1, r12):
            #print r1[1], r12[1]
            return r12


    else:
        
        r12 = (p2+v2, p1-(p2+v2))
        #print r12

        # Are they too far apart?
        if np.linalg.norm(r12[1]) > norm_limit:
            return None

        # Are they on the same line?  
        if areParallel(r1, r12):
            #print r1[1], r12[1]
            return r12

    return None

def areParallel(r1, r2):

    norm1, norm2 = np.linalg.norm(r1[1]), np.linalg.norm(r2[1])
    if norm1 == 0 or norm2 == 0:
        return False

    x = np.dot(r1[1]/norm1, r2[1]/norm2)

    threshold = 0.004

    if abs(abs(x)-1) < threshold:
        #print r1[1], r2[1]
        return True
    else:
        return False

def arePerpendicular(r1, r2):
 
    norm1, norm2 = np.linalg.norm(r1[1]), np.linalg.norm(r2[1])
    if norm1 == 0 or norm2 == 0:
        return False

    threshold = 0.15
    if abs(np.dot(r1[1]/norm1, r2[1]/norm2)) < threshold:
        return True
    else:
        return False


def connectPerpendicularLines(r1, r2):

    #Are the line segments perpendicular?
    if not arePerpendicular(r1, r2):
        return None


    # Solve the LSE
    A = np.array([[r1[1][0], -r2[1][0]], [r1[1][1], -r2[1][1]]])
    if np.linalg.det(A) == 0:
        return None
 
    b = np.array([r2[0][0]-r1[0][0], r2[0][1]-r1[0][1]])

    x = np.linalg.solve(A, b)

    norm_limit = 100
    coef_error = 0.2

    # if one line segment needs to be lengthened to intersect the other
    if 0-coef_error<x[0]<=1+coef_error and (x[1]>1 or x[1]<0):
      
        if x[1]<0:
            p = r2[0]
            v = r2[1]*x[1]
        else:
            p = r2[0] + r2[1]  
            v = r2[1]*(x[1]-1)


    elif 0-coef_error<x[1]<=1+coef_error and (x[0]>1 or x[0]<0):

        if x[0]<1:
            p = r1[0]
            v = r1[1]*x[0]
        else:    
            p = r1[0] + r1[1]
            v = r1[1]*(x[0]-1)
    else: 
        return None 

    p = np.array(p, dtype=np.int32)
    v = np.array(v, dtype=np.int32)

    if np.linalg.norm(v) < norm_limit:
        return (p,v)

    
    return None


# Get intersection point of perpendicular lines
def getIntersectionPoint(r1, r2):

    #
    if not arePerpendicular(r1,r2):
        return None


    # Solve the LSE
    A = np.array([[r1[1][0], -r2[1][0]], [r1[1][1], -r2[1][1]]])
    if np.linalg.det(A) == 0:
        return None
 
    b = np.array([r2[0][0]-r1[0][0], r2[0][1]-r1[0][1]])

    x = np.linalg.solve(A, b)

    error = 0.5
    if 0-error<x[0]<1+error and 0-error<x[1]<1+error:
        #print r1, r2
        return map(int, r1[0]+r1[1]*x[0])
    else:
        return None


def metric(p1,p2):
    return abs(p1[0]-p2[0])>30


def clusterCornerPoints(points):

    #DBSCAN point clustering algorithm
    labels = DBSCAN(eps=20, min_samples=5).fit_predict(points)

    point_labels = zip(points, labels)
    
    rectangles = []
    for label in np.unique(labels):
        if label != -1:
        
            rect = cv2.minAreaRect( np.array([x[0] for x in point_labels if x[1]==label]) )

            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #print box

            rectangles.append(box)


    return rectangles





def clusterIntersectionPoints(points):

    #DBSCAN point clustering algorithm
    labels = DBSCAN(metric = metric ).fit_predict(points)


    # HDBSCAN --||--
    #clusterer = hdbscan.HDBSCAN(metric='manhattan', min_cluster_size=10)
    #clusterer.fit(points)


    '''
    X = points[:,1].reshape(len(points),1)
    y = points[:,0]

    # RANSAC robust linear regression algorithm
    #ransac = linear_model.RANSACRegressor(random_state=15)
    #ransac.fit(X, y)

    # Theil Sen regression
    theilSen = linear_model.TheilSenRegressor(random_state=15)
    theilSen.fit(X, y)

    line_x = np.array([0,2000])

    line_y = theilSen.predict(line_x.reshape(len(line_x), 1))

    plt.plot(line_x, line_y, color='red')  
    plt.scatter(X, y)
    plt.show()
    '''


    '''

    number_of_iteration_x = int(img.shape[0]/100)
    number_of_iteration_y = int(img.shape[1]/200)

    sub_img = np.zeros((img.shape[0], img.shape[1],3), np.uint8)
    sub_points = np.zeros((img.shape[0],img.shape[1]))

    for x in range(number_of_iteration_x):
        for y in range(number_of_iteration_y):
            temp = empty[x * 100-20:+x * 100 + 100,y * 200-20:+y * 200 + 200]
            temp[temp == 255] = 1
            if np.count_nonzero(temp) > 25:
                print(x,y,np.count_nonzero(temp))
                sub_img[x * 100-20:+x * 100 + 100,y * 200-20:+y * 200 + 200] = img[x * 100-20:+x * 100 + 100,y * 200-20:+y * 200 + 200].copy()
                sub_points[x * 100-20:+x * 100 + 100,y * 200-20:+y * 200 + 200] = 255
    '''

    return  None



##################################################################################################

def vectorToVectorDistance(r1, r2):

    v1, v2 = r1[1], r2[1]
    p1, p2 = r1[0], r2[0]

    if p1[0] < p2[0]:
        d = ((p1[0]+v1[0]-p2[0])**2 + (p1[1]+v1[1]-p2[1]))**0.5
    else:
        d = ((p2[0]+v2[0]-p1[0])**2 + (p2[1]+v2[1]-p1[1]))**0.5

    return d

def pointToPointDistance(x1, y1, x2, y2):
    return ((y1-y2)**2 + (x1-x2)**2)**(0.5)

def pointToPointDistanceSq(x1, y1, x2, y2):
    return (y1-y2)**2 + (x1-x2)**2



#####################################################################################################

def resize(image_h, image_w):

    #Constants
    image_to_working_ratio = 4
    
    windows_width = 1366
    windows_height = 768

    working_h = image_h / float(image_to_working_ratio) 
    working_w = image_w / float(image_to_working_ratio)

    working_to_window_ratio = 1

    if working_h > windows_height - 250:
        working_to_window_ratio = max(working_to_window_ratio, (working_h + 250)/ float(windows_height)) #toolbar

    if working_w > windows_width:
        working_to_window_ratio = max(working_to_window_ratio, working_w / float(windows_width))


    return (image_to_working_ratio, working_to_window_ratio)

