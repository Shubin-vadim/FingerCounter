import os
import random
import cv2
import HandTrackingModule as htm
import time
import numpy as np
import cv2

# настройки камеры
wCam, hCam = 720, 640
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
print(os.getcwd())
# получение списка изображений
folderPath = 'images'
mylist = os.listdir(folderPath)
overlayList = []
detector = htm.MpHandTracking(min_threshold=0.75)
tipIds = [4, 8, 12, 16, 20]
print(mylist)
for imgPath in mylist:
    img = cv2.imread(f"{folderPath}/{imgPath}")
    img = cv2.resize(img, (200, 200))
    overlayList.append(img)

while True:
    sccuess, frame = cap.read()  # чтение записи камеры
    # frame[0:200, 200:400] = overlayList[random.randint(0, 5)]
    frame = detector.find_hands(frame)  # нахождение контуров пальцев
    lmdist = detector.find_position(frame, draw=False)  # нахождение позиции точек пальцев, без их рисовки

    if len(lmdist) > 0:  # проверка на наличии найденных точек
        fingers = []
        if lmdist[tipIds[0]][1] > lmdist[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)  # проверка на закрытость или большого пальца

        for id in range(1, 5):
            if lmdist[tipIds[id]][2] < lmdist[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)  # проверка на закрытость или открытость пальцев руки
        print(fingers)
        totalFingers = fingers.count(1)
        frame[0:200, 0:200] = overlayList[totalFingers]
    #cnt = 0

    # h,w,c = overlayList[0].shape
    # cv2.rectangle(frame, (0,0), (100,100), (132, 43, 184), cv2.FILLED)
        cv2.putText(frame, 'Score: ' + str(totalFingers), (410, 55), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
    cv2.imshow('Rez', frame)
    cv2.waitKey(1)