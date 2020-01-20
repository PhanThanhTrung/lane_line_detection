import numpy as np 
import matplotlib.image as pimg
import cv2
import os 
from utils import * 
import matplotlib.pyplot as plt
image_path='./sample image/'
predict_image_path='./predict image/'

def process(img):
    gray_img= grayscale(img)
    gauss_img=gaussian_blur(gray_img,kernel_size=3)
    cann=canny(gauss_img)
    h,w=cann.shape
    vectices=np.array([[0,h],[0,100],[w,100],[w,h]])
    ROI=region_of_interest(cann,vectices)
    line_img,left_line,right_line = hough_lines(ROI)
    predict=weighted_img(line_img, img)
    return predict

def main():
    for image_name in sorted(os.listdir(image_path)):
        print(image_name)
        file_path=image_path+image_name
        img=pimg.imread(file_path)
        predict=process(img)
        name=predict_image_path+image_name
        plt.imsave(name,predict)

if __name__=="__main__":
    main()