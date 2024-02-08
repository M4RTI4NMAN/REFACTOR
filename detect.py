import selenium
import pyautogui
import torch 
import numpy as np
import cv2
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService 
from selenium.webdriver.chrome import ChromeDriverManager 
#from bs4 import beautifulsoup
#import tensorflow as tf
#import tensoprflow_datasets as tfds
#import matplotlib.pyplot as plt
#import numpy as np
#from PIL import Image, ImageDraw n 
#import pynport
# from pynput.mouse import Button, Controller
import os
import webbrowser

#check for object || OBJECT DETECTION
     
ZYMM_DETECTION = cv2.CascadeClassifier(cv2.data.zymmCascades + '')

cap = cv2.VideoCapture(0)

object_detection = False

while 1:
    ret, img = cap.read()
    grey = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)
    objects = ZYMM_DETECTION.detectMultiScale(grey, 1.3, 5)

    if len(ZYMM_DETECTION) > 0:
        object_detection = True
    else: 
        object_detection = False
    print(object_detection)  
    if len() > 0
    
for (x,y,w,h) in faces:
 
    ##REPEAT FOR EACH CLASS TYPE INSTANCE [[BUTTONS, HYPERLINK, INPUT FIELDS,]]
    
    
while 1:
    ret, img = cap.read()
    grey = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)
    objects = ZYMM_DETECTION.detectMultiScale(grey, 1.3, 5)

    if len(ZYMM_DETECTION) > 0:
        object_detection = True
    else: 
        object_detection = False
    print(object_detection)  
    if len() > 0
     
    ##REPEAT FOR EACH CLASS TYPE INSTANCE [[BUTTONS, HYPERLINK, INPUT FIELDS,]]
    
    
while 1:
    ret, img = cap.read()
    grey = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)
    objects = ZYMM_DETECTION.detectMultiScale(grey, 1.3, 5)

    if len(ZYMM_DETECTION) > 0:
        object_detection = True
    else: 
        object_detection = False
    print(object_detection)  
    if len() > 0
    
    ##REPEAT FOR EACH CLASS TYPE INSTANCE [[BUTTONS, HYPERLINK, INPUT FIELDS,]]
    
    
    ##FIND OBJECT ON SCREEN AND CLICK ON SCREEN 




    
##check for detected item(s)

import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

face_detect = False

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        face_detect = True
    else:
        face_detect = False
    print(face_detect)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()






# Load image - work in greyscale as 1/3 as many pixels
obj = cv2.imread('.png', cv2.IMREAD_GRAYSCALE)
# Overwrite "Current Best" with white - these numbers will vary depending on what you capture   
obj[134:400,447:714] = 255
# Overwrite menu and "Close" button at top-right with white - these numbers will vary depending on what you capture
obj[3:107,1494:1726] = 255
# Negate image so whites become black   
obj=255-obj

# Find anything not black, i.e. the object 
nz = cv2.findNonZero(im)

# Find top, bottom, left and right edge of object 
a = nz[:,0,0].min()
b = nz[:,0,0].max()    
c = nz[:,0,1].min()
d = nz[:,0,1].max()

print( 'a:{}, b:{}, c:{}, d:{}'.format(a,b,c,d))

# Average top and bottom edges, left and right edges, to give centre
c0 = (a+b)/2
c1 = (c+d)/2
print('Object Location: {}, {}'.format(c0,c1))
##EST 840 miliseconds to identify 


#implement images into model/work Zymms - computerized vision in model  (recognition)

OUTLINE = (0,255,0) 

builder = tfds.builder('voc2007')
builder.download_and_prepare()
datasetss = builder.as_dataset()
train_data, test_data = datasets['train'], datasets['test']
iterator = train_data.repeat(1).batch(1).make_one_shot_iterator()
next_batch = iterator.get_next()

with tf.Session() as sess:
    for _ in range(1):
        batch = sess.run(next_batch)
        image = batch['image']
        bboxes = batch['objects']['bbox']
        bboxes, image = np.squeeze(bboxes), np.squeeze(image)
        pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
        draw = ImageDraw.Draw(pil_image)
        height, width = image.shape[:2]
        try:
            if (isinstance(bboxes[0], np.float32)
                or isinstance(bboxes[0], np.float64)):
                bboxes = [bboxes]

            for bbox in bboxes:
                ymin, xmin, ymax, xmax = bbox
                xmin *= width
                xmax *= width
                ymin *= height
                ymax *= height
                c1 = (xmin, ymin)
                c2 = (xmax, ymin)
                c3 = (xmax, ymax)
                c4 = (xmin, ymax)
                draw.line([c1, c2, c3, c4, c1],
                          fill=OUTLINE,
                          width=3)
            asnumpy = np.array(pil_image)
            figure = plt.figure(figsize=tuple(x/50 for x in image.shape[:2]))
            plt.imshow(asnumpy)
        except TypeError:
            pass

#identify and perform action on event with Pynout. Click on specific point foward to next web element with tab
#buttons and input boxes 
# define a function to display the coordinates of        







#algorithmic data structure that takes coordinates from zymm's vision and input here - result put into INTERACT function
    