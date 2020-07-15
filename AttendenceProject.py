import cv2
import numpy as np
import face_recognition
import  os
from  datetime import datetime

path = 'pics'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in  myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttedance(name):
    with open('Attendence.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in  myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
           now = datetime.now()
           dtString = now.strftime('%H:%M:%S')
           f.writelines(f'\n{name},{dtString}')






encodeListKnown = findEncoding(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgs = cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgs)
    encodeCurFrame = face_recognition.face_encodings(imgs,facesCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttedance(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)




# facLoc = face_recognition.face_locations(imgSri)[0]
# encodeSri = face_recognition.face_encodings(imgSri)[0]
# cv2.rectangle(imgSri,(facLoc[3],facLoc[0]),(facLoc[1],facLoc[2]),(255,0,255),2)
#
# facArj = face_recognition.face_locations(imgArj)[0]
# encodeArj = face_recognition.face_encodings(imgArj)[0]
# cv2.rectangle(imgArj,(facArj[3],facArj[0]),(facArj[1],facArj[2]),(255,0,255),2)
#
#
# results = face_recognition.compare_faces([encodeSri],encodeArj)
# faceDis = face_recognition.face_distance([encodeSri],encodeArj)

