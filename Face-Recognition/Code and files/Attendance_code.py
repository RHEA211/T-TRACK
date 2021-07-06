
import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime

path = (r'C:\Users\KIIT\Desktop\Matchr\T-TRACK\data')
images = []
#List that has all the images

className = []
# List the has all the class names

myList = os.listdir(path)
print("Total Students Detected:",len(myList))
print(myList)
#This prints the no.of images

for x,c in enumerate(myList):

        currentImg = cv2.imread(f'{path}/{c}')
        images.append(currentImg)
        className.append(os.path.splitext(c)[0])
print(className)
# here we extract the name from eg- Selena.jpg gives Selena

#Face encoding- noting down important measurements of the face
def Encode(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

#  Function to mark the attendance
def Attendance(name):
    with open('Class.csv','r+') as f:
        myDataList = f.readlines()
        nameList =[]
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dt_string = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dt_string}')
#If the name is already present , it wont mark it again


encodingKnown = Encode(images)
print('Done with Encoding')
print(len(encodingKnown))

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), fx=0.20, fy=0.20)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)


    facesinFrame = face_recognition.face_locations(imgS)
    encodesFrame = face_recognition.face_encodings(imgS, facesinFrame)

    for encodeFace,faceLoc in zip(encodesFrame,facesinFrame):
        matches = face_recognition.compare_faces(encodingKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodingKnown, encodeFace)

        #print(faceDis)- will show the values of dist
        #the lowest one means the face resembles the person haveing the lowest value of dst
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = className[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 5, x2 * 5, y2 * 5, x1 * 5
           #since we had resized the image above as :fx = 0.20, fy = 0.20
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 139), 2)
            cv2.rectangle(img, (x1, y2 -40), (x2, y2), (0, 0, 139))
            cv2.putText(img, name, (x1 + 10, y2 - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
            Attendance(name)




    cv2.imshow('FaceRecognition', img)
    key = cv2.waitKey(10)
    if key == 27: # that is escape button
         break


