import cv2
import numpy as np
import pandas as pd
import matplotlib as plt

PATH = '/groups/CS156b/data/'

fil = pd.read_csv("train.csv")
img = cv2.imread(PATH + str(fil['Path'][0]))
height = img.shape[0] 
width = img.shape[1] 
channels = img.shape[2]
print(height,width,channels)
new_img = cv2.resize(img, (50,50))

height = new_img.shape[0] 
width = new_img.shape[1] 
channels = new_img.shape[2]
print(height,width,channels)
cv2.imwrite("test_resize.png", new_img)
