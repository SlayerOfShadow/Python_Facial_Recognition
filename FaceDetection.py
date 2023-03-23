# -*- coding: utf-8 -*-
"""
Created on Sat May 22 14:37:03 2021

@author: Antoine
"""

#-------------------- DATABASE CONNECTION --------------------

import mysql.connector
from mysql.connector import Error

try:
    connection = mysql.connector.connect(host='localhost',
                                         database='face_detection_test',
                                         user='root',
                                         password='jesuisuneracinesuperpuissante')
    if connection.is_connected():
        db_Info = connection.get_server_info()
        print("Connected to MySQL Server version ", db_Info)
        cursor = connection.cursor()
        cursor.execute("select database();")
        record = cursor.fetchone()
        print("You're connected to database: ", record)

except Error as e:
    print("Error while connecting to MySQL", e)

#-------------------- FACE DETECTION --------------------

import os
import cv2
import numpy as np
import json

with open('data_python.json') as json_file:
    data = json.load(json_file)
    
    ImagePath = data['image_path']

#path
path = ImagePath

image = cv2.imread(path)
#gray level needed to use facial recognition function
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

faceCascade = {"name" : "faceCascade", "xml" : cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")}
eyeCascade = {"name" : "eyeCascade", "xml" : cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")}
noseCascade = {"name" : "noseCascade", "xml" : cv2.CascadeClassifier('E:\Logiciels\OpenCV\opencv\sources\data\haarcascades\haarcascade_mcs_nose.xml')}
mouthCascade = {"name" : "mouthCascade", "xml" : cv2.CascadeClassifier('E:\Logiciels\OpenCV\opencv\sources\data\haarcascades\haarcascade_mcs_mouth.xml')}
leftearCascade = {"name" : "leftearCascade", "xml" : cv2.CascadeClassifier('E:\Logiciels\OpenCV\opencv\sources\data\haarcascades\haarcascade_mcs_leftear.xml')}
rightearCascade = {"name" : "rightearCascade", "xml" : cv2.CascadeClassifier('E:\Logiciels\OpenCV\opencv\sources\data\haarcascades\haarcascade_mcs_rightear.xml')}

cascades = []
cascades.extend((faceCascade, eyeCascade, noseCascade, mouthCascade, leftearCascade, rightearCascade))
    
for i in range(len(cascades)):
    if cascades[i]["xml"].empty():
        raise IOError('Unable to load ' + cascades[i]["name"] + ' classifier xml file')

faces = faceCascade["xml"].detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(30, 30)
)
print("Found {0} face(s).".format(len(faces)))

if len(faces) == 0:
    isValid = False
else:
    isValid = True

if (isValid == True):
    
    print("Starting facial recognition...")
    
    #-------------------- FACE --------------------
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_color = image[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]
        cv2.imwrite("Results/DetectedFace.jpg", roi_color)
        
        #-------------------- EYES --------------------
        
        eyes = eyeCascade["xml"].detectMultiScale(
            roi_gray,
            scaleFactor=1.15,
            minNeighbors=3,
            minSize=(30, 30)
        )
        
        num = 0
        for (x, y, w, h) in eyes:
            cv2.rectangle(roi_color, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imwrite('Results/DetectedEye' +str(num)+ '.jpg', roi_color[y:y+h, x:x+w])
            num = num + 1
            
        #-------------------- NOSE --------------------
            
        nose = noseCascade["xml"].detectMultiScale(
            roi_gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(45, 45)
        )
        for (x, y, w, h) in nose:
            cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.imwrite("Results/DetectedNose.jpg", roi_color[y:y+h, x:x+w])
        
        #-------------------- MOUTH --------------------
        
        mouth = mouthCascade["xml"].detectMultiScale(
            roi_gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(60, 60)
        )
        for (x, y, w, h) in mouth:
            cv2.rectangle(roi_color, (x, y), (x+w, y+h), (0, 165, 255), 2)
            cv2.imwrite("Results/DetectedMouth.jpg", roi_color[y:y+h, x:x+w])
            
        #-------------------- EARS --------------------
            
        leftear = leftearCascade["xml"].detectMultiScale(
            roi_gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
        )
        for (x, y, w, h) in leftear:
            cv2.rectangle(roi_color, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
        rightear = rightearCascade["xml"].detectMultiScale(
            roi_gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
        )
        for (x, y, w, h) in rightear:
            cv2.rectangle(roi_color, (x, y), (x+w, y+h), (0, 0, 255), 2)
            

    window_name = "Image"
    cv2.imshow(window_name, image)
    
    min_YCrCb = np.array([0,133,77],np.uint8)
    max_YCrCb = np.array([235,173,127],np.uint8)
    
    # Get pointer to video frames from primary device
    faceImage = cv2.imread("Results/DetectedFace.jpg")
    
    imageYCrCb = cv2.cvtColor(faceImage,cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
    
    skinYCrCb = cv2.bitwise_and(faceImage, faceImage, mask = skinRegionYCrCb)
    
    cv2.imwrite("Results/DetectedSkin.jpg", np.hstack([skinYCrCb]))
    
    print("Facial recognition done. Check the results in the folder 'Results'.")
    
else:
    print("No face detected, facial recognition impossible. Please try again.")

#-------------------- INSERT RESULTS INTO DATABASE --------------------

directory = "Results"

with connection.cursor() as cursor:
    for filename in os.listdir(directory):
        insert_result_query = "INSERT INTO images (name) VALUES ('" + filename + "')"
        cursor.execute(insert_result_query)
        connection.commit()
        
#-------------------- TODO --------------------
        #Detect hair color

#avoid Python kernel crash
cv2.waitKey(0) 
cv2.destroyAllWindows() 