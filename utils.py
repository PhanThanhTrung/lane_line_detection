import numpy as np 
import cv2 
import matplotlib.image as pimg
import os

def grayscale(img):
    """
        input: image
        output: grayscale image
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian_blur(img, kernel_size=3):
    """
        input: image and kernel size
        output: image after adding gauusian blur
    """
    return cv2.GaussianBlur(img, ksize=(kernel_size,kernel_size),sigmaX=0)

def canny(img, low_threshold=200, high_threshold=255):
    """
        get the edges with canny algorithms
        input: img, low_threshold, high_theshold
    """
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
    '''
        extract the Region proposal 
        only use for rectangle ROI
    '''
    x_min,y_min=vertices[1][0],vertices[1][1]
    x_max,y_max=vertices[3][0],vertices[3][1]
    m,n= img.shape
    for i in range(m):
        for j in range(n):
            if i<y_min or i >y_max or j<x_min or j>x_max:
                img[i][j]=0
    
    return img

def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    
    right_slopes = []
    right_intercepts = []
    left_slopes = []
    left_intercepts = []
    left_points_x = []
    left_points_y = []
    right_points_x = []
    right_points_y = []
    
    y_max = img.shape[0]
    y_min = img.shape[0]
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            if x2-x1!=0.:
                slope = (y2-y1)/(x2-x1)
            else:
                slope=999.
            if slope < 0.0:
                left_slopes.append(slope) # left line
                left_points_x.append(x1)
                left_points_x.append(x2)
                left_points_y.append(y1)
                left_points_y.append(y2)
                left_intercepts.append(y1 - slope*x1)
            
            if slope > 0.0:
                right_slopes.append(slope) # right line
                right_points_x.append(x1)
                right_points_x.append(x2)
                right_points_y.append(y1)
                right_points_y.append(y2)
                right_intercepts.append(y1 - slope*x1)
            
            y_min = min(y1,y2,y_min)
            

    left_line=[]
    right_line=[]    
    if len(left_slopes) > 0:
        left_slope = np.mean(left_slopes)
        left_intercept = np.mean(left_intercepts)
        x_min_left = int((y_min - left_intercept)/left_slope) 
        x_max_left = int((y_max - left_intercept)/left_slope)
        cv2.line(img, (x_min_left, y_min), (x_max_left, y_max), color, thickness)
        left_line.append((x_min_left, y_min))
        left_line.append((x_max_left, y_max))
    
    if len(right_slopes) > 0:
        right_slope = np.mean(right_slopes)
        right_intercept = np.mean(right_intercepts)
        x_min_right = int((y_min - right_intercept)/right_slope) 
        x_max_right = int((y_max - right_intercept)/right_slope)
        cv2.line(img, (x_min_right, y_min), (x_max_right, y_max),  color, thickness)
        right_line.append((x_min_left, y_min))
        right_line.append((x_max_left, y_max))
    return left_line,right_line

def hough_lines(img, rho=2, theta=np.pi/180, threshold=40, min_line_len=10, max_line_gap=25):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    try:
        left_line,right_line = draw_lines(line_img, lines)
        return line_img, left_line,right_line
    except:
        return line_img,[],[]

def weighted_img(img, initial_img, a=0.8, b=1., y=0.):
    
    
    return cv2.addWeighted(initial_img, a, img, b, y)

