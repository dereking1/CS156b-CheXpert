# Test file to process the images
import cv2
import numpy as np
import pandas as pd
import matplotlib as plt

PATH = '/groups/CS156b/data/'

fil = pd.read_csv("train.csv")
print(len(fil['Path']))
cnt = 0
for path in fil['Path']:
    if cnt % 1000 == 0:
        print(cnt)
    img = cv2.imread(PATH + path)
    new_img = cv2.resize(img, (50,50))
    pth = path.replace("/","_")
    cv2.imwrite(f"resized_images/{pth}", new_img)
    cnt += 1
