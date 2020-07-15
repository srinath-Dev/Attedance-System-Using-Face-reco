import cv2
import numpy as np
import face_recognition

imgSri = face_recognition.load_image_file('pics/Srinath.jpg')
imgSri = cv2.cvtColor(imgSri,cv2.COLOR_BGR2RGB)
imgArj = face_recognition.load_image_file('pics/arjpic.png')
imgArj = cv2.cvtColor(imgArj,cv2.COLOR_BGR2RGB)

facLoc = face_recognition.face_locations(imgSri)[0]
encodeSri = face_recognition.face_encodings(imgSri)[0]
cv2.rectangle(imgSri,(facLoc[3],facLoc[0]),(facLoc[1],facLoc[2]),(255,0,255),2)

facArj = face_recognition.face_locations(imgArj)[0]
encodeArj = face_recognition.face_encodings(imgArj)[0]
cv2.rectangle(imgArj,(facArj[3],facArj[0]),(facArj[1],facArj[2]),(255,0,255),2)


results = face_recognition.compare_faces([encodeSri],encodeArj)
faceDis = face_recognition.face_distance([encodeSri],encodeArj)
print(results,faceDis)
cv2.putText(imgArj,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Srinath',imgSri)
cv2.imshow('Arjun',imgArj)
cv2.waitKey(0)