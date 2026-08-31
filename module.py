import cv2 as cv
import mediapipe as mp
import math
import time
from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2:CLASS THAT DETECTS AND TRACKS HANDS IN VIDEO USING GOOGLE's MEDIAPIPE

class HandDetector():

    def __init__(self, modePath="hand_landmarker.task", maxHands=2):

        #STORING THE SETTING FOR HAND DETECTION
        self.maxHands=maxHands
        self.lmList=[]
        # CREATE THE HandLandmaker options OBJECT
        # This tells MediaPipe:
        # 1. Which model file to load
        # 2. Maximum number of hands to detect
        # BaseOptions= mp.tasks.BaseOptions(): this creates object not class
        BaseOptions=python.BaseOptions
        #CREATE A HAND LANDMARK INSTANCE :

        module_dir = Path(__file__).resolve().parent
        candidate_paths = [
            module_dir / modePath,
            module_dir.parent / "Hand Tracking" / "hand_landmarker (4).task",
            module_dir / "hand_landmarker.task",
        ]
        model_path = next((p for p in candidate_paths if p.exists()), None)
        if model_path is None:
            raise FileNotFoundError(
                "Could not find a hand landmarker model file. "
                "Expected one of: "
                + ", ".join(str(p) for p in candidate_paths)
            )

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(model_path),
                delegate=python.BaseOptions.Delegate.CPU
            ),
            num_hands=maxHands,
            running_mode=vision.RunningMode.IMAGE
        )

        #CREATE HAND DETECTOR OBJECT
        self.handDetector=vision.HandLandmarker.create_from_options(options=options)
        #VARIABLE USED LATER TO STORE DETECTION RESULTS
        self.results =None

    def findHands(self,img,draw=True):

        #CHANGE THE IMAGE COLOR FROM BGR TO RGB
        imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)


        #CONVERT NUMPY IMAGE INTO MEDIAPIPE IMAGE
        mpImage =mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=imgRGB

        )
        #DETECT HANDS IN CURRENT FRAME
        self.results=self.handDetector.detect(mpImage)

        #DRAW ALL LANDMARKS IF HANDS ARE DETECTED
        if self.results.hand_landmarks:
            h,w,c=img.shape
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (5, 9), (9, 10), (10, 11), (11, 12),
                (9, 13), (13, 14), (14, 15), (15, 16),
                (13, 17), (17, 18), (18, 19), (19, 20),
                (0, 17)
            ]
            #LOOP THROUGH EVERY DETECTED HAND
            for hand in self.results.hand_landmarks:
                points = []
                #LOOP THROUGH ALL 21 LANDMARKS
                for id, lm in enumerate(hand):
                    cx=int (lm.x*w)
                    cy=int (lm.y*h)
                    points.append((cx, cy))

                    if draw :
                        cv.circle(img, (cx, cy), 10, (0, 0, 255), cv.FILLED)

                if draw:
                    for start, end in connections:
                        cv.line(img, points[start], points[end], (255, 255, 255), 2)

        return img

    def findPosition(self,img,handNo=0,draw=True):
        #LIST THAT STORES LANDMARK POSITION
        self.lmList=[]

        #CONTINUE ONLY IF HANDS ARE DETECTED
        if self.results and self.results.hand_landmarks:
            #SELECT REQUESTED HANDS
            myHands=self.results.hand_landmarks[handNo]
            h,w,c=img.shape

            #LOOP THROUGH ALL LANDMARKS.
            for id ,lm in enumerate(myHands):

                #CONVERT NORMALISED COORDINATE INTO PIXEL COORDINATES
                cx=int(lm.x*w)
                cy=int(lm.y*h)

                #SAVE LANDMARK ID COORDINATES
                self.lmList.append([id,cx,cy])

        return self.lmList

    def findDistance(self, p1, p2, img, draw=True):

        # Get coordinates of the two landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]

        # Midpoint
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if draw:
            # Draw line
            cv.line(img, (x1, y1), (x2, y2), ( 0,255,0 ), 3)

            # Draw circles
            cv.circle(img, (x1, y1), 10, ( 255, 0,255), cv.FILLED)
            cv.circle(img, (x2, y2), 10, (255,0,255), cv.FILLED)
            cv.circle(img, (cx, cy), 10, (255,0, 0), cv.FILLED)

        # Euclidean distance
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]




def main():

    #VARAIABLES USED FOR FPS CALCULATIONS
    prevTime=0
    currentTime=0

    #CREATE DETECTOR OBJECT
    detector=HandDetector()

    #OPEN WEBCAM
    cap=cv.VideoCapture(0)

    while True :

        #READ FRAME FROM WEBCAM
        success,img=cap.read()

        #SKIP FRAME IF CAMERA FAILS
        if not success :
            continue

        #DETECT HANDS
        img =detector.findHands(img)

        #EXTRACT LANDMARK COORDINATES
        lmList=detector.findPosition(img)

        #PRINT  THUMB TIP POSITION
        if len(lmList)!=0:
            print(lmList[4])

        #FPS CALCUTIONS
        currentTime=time.time()

        fps = 0
        if currentTime-prevTime!=0:
            fps=1/(currentTime-prevTime)

        prevTime=currentTime

        #DISPLAY FPS
        cv.putText(img,str(int(fps)), (50,70),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        #SHOW IMAGE
        cv.imshow("Image",img)
        cv.waitKey(1)

if __name__ == "__main__":
    main()





