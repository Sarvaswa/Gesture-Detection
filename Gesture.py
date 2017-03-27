'''
MIT License

Copyright (c) 2017 Sarvaswa Tandon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import cv2
import numpy as np

cam = cv2.VideoCapture(0)       #Initialising Camera

while True:
    #Reading Image and Converting to HSV
    stat, img = cam.read()
    imgh = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    #Setting Detection Parameters and Detecting Skin
    low1 = np.array([0,0,0])
    high1 = np.array([30,200,255])
    img_bin1 = cv2.inRange(imgh,low1,high1)

    low2 = np.array([150,0,0])
    high2 = np.array([180,200,255])
    img_bin2 = cv2.inRange(imgh,low2,high2)

    img_bin = img_bin1 | img_bin2

    #Smoothing Image
    img_bin = cv2.medianBlur(img_bin,3)

    #Performing Morphological Processing - Errosion Followed by Dilation
    kernel = np.ones((7,7),np.uint8)                        #Kernel Creation (Square Kernel)
    img_bin = cv2.erode(img_bin, kernel, iterations = 1)    #Erosin
    img_bin = cv2.dilate(img_bin, kernel, iterations = 1)   #Dilation

    #Contour Finding
    imcpy = img_bin | img_bin
    _, contours, hierarchy = cv2.findContours(imcpy,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    #Filtering Out insignificant contours on the basis of area
    hand = []
    for i in contours:
        if cv2.contourArea(i)>10000:
            hand.append(i)

    #Finding convex hulls
    chull = []
    for contour in hand:
        hull = cv2.convexHull(contour)
        chull.append(hull)

    if len(hand)>0 and len(chull)>0:
        hull_points = []
        for item in chull[0]:
            point = [item[0][0],item[0][1]]
            hull_points.append(point)
            
        contour_points = []
        for item in hand[0]:
            point = [item[0][0],item[0][1]]
            contour_points.append(point)

        indices = []
        for item in hull_points:
            indices.append(contour_points.index(item))

        indices = np.array(indices)
    
        defect = cv2.convexityDefects(hand[0],indices)

        cv2.drawContours(img, chull, -1, (0,255,255), 3)
        
        marks = []
        for i in defect:
            if i[0][3]>8000:
                x = hand[0][i[0][0]][0][0]
                y = hand[0][i[0][0]][0][1]
                cv2.circle(img, (x,y), 4, (0,0,255), -1)
                marks.append(i)
                
        if len(marks)>0:
            x = hand[0][marks[-1][0][1]][0][0]
            y = hand[0][marks[-1][0][1]][0][1]
            cv2.circle(img, (x,y), 4, (0,0,255), -1)
        
    #Drawing the contours in Yellow
##    cv2.drawContours(img, chull, -1, (0,255,255), 3)
    cv2.imshow('Image',img)         #Displaying Image
    
    #Exit Code
    if cv2.waitKey(1) == 32:
        break

cv2.destroyAllWindows()
cam.release()
