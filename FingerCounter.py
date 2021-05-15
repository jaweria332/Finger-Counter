# Importing necessary libraries
import cv2
import time
import os
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
# Defining parameters
wcam, hcam = 640, 480
# For width
cap.set(3, wcam)
# For height
cap.set(4, hcam)

# Storing images
folderpath = "FingerImages"
myList = os.listdir(folderpath)
print(myList)

# Creating an overlay list
overlayList = []
for imgPath in myList:
    # Here: imgPath = 1.jgp etc
    image = cv2.imread(f'{folderpath}/{imgPath}')
    # print(f'{folderpath}/{imgPath}')
    overlayList.append(image)

# Checking - everything is working correctly - Got the image length correctly
print(len(overlayList))

cTime = 0
pTime = 0

# Creating a detector
detector = htm.HandDetector(detectionCon=0.75)
# Defining landmarks id
tipIds = [4, 8, 12, 16, 20]

while True:
    ret, img = cap.read()

    # Defining detector parameter
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)

    if len(lmList) != 0:
        # We need to get landmark 4,8,12,16,20 - can refer to the mediapipe website
        fingers = []


        # For thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # For Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        print(fingers)

    # Getting height and width
    h, w, c = overlayList[0].shape
    img[0:h, 0:w] = overlayList[0]

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (200, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    cv2.imshow("Camera Stream", img)
    cv2.waitKey(1)
