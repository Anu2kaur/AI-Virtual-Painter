import cv2 as cv
import module as htm
import time
import numpy as np

###########################################
# Settings
###########################################

brush_thickness = 15
eraser_thickness = 100
frameR=100
smoothening= 5

xp, yp = 0, 0
drawColor = (0, 255, 0)  # Green default

# White canvas (easier erasing with white color)
imageCanvas = np.ones((720, 1280, 3), np.uint8) * 255

############################################
# Load Toolbar
############################################

folderPath = "."
mylist=['yellow.jpg', 'blue.jpg', 'erase.jpg', 'front.jpg', 'red.jpg']

overlay = []
for imgPath in mylist:
    image = cv.imread(f"{folderPath}/{imgPath}")
    overlay.append(image)

header = overlay[3]

# Resize toolbar if needed
header = cv.resize(header, (142, 720))

toolbar_width = header.shape[1]
############################################
# Camera
############################################

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

ptime = 0

detector = htm.HandDetector()

############################################
# Main Loop
############################################

while True:

    success, img = cap.read()


    if not success:
        break

    img = cv.flip(img, 1)
    h, w = img.shape[:2]

    img = detector.findHands(img, False)
    # Draw active drawing area
    cv.rectangle(
        img,
        (toolbar_width + frameR, frameR),
        (w - frameR, h - frameR),
        (180, 180, 180),
        2
    )
    lmList = detector.findPosition(img)

    if len(lmList) >= 21:

        tips = [8, 12, 16, 20]
        fingers = []

        for tip in tips:
            if lmList[tip][2] < lmList[tip - 2][2]:
                fingers.append(tip)

        ####################################
        # Selection Mode
        ####################################

        if len(fingers) == 2 and 8 in fingers and 12 in fingers:

            xp, yp = 0, 0

            x1, y1 = lmList[8][1], lmList[8][2]
            x2, y2 = lmList[12][1], lmList[12][2]

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            cv.circle(img, (cx, cy), 15, (0, 0, 0), cv.FILLED)

            if 43 < cx < 250:

                if cy < 125:
                    print("Red")
                    header= overlay[4]
                    drawColor = (0, 0, 255)

                elif 125 < cy < 285:
                    print("Yellow")
                    header = overlay[0]
                    drawColor = (0, 255, 255)

                elif 285 < cy < 445:
                    print("Blue")
                    header = overlay[1]
                    drawColor = (255, 0, 0)

                elif 445 < cy < 612:
                    print("Eraser")
                    header = overlay[2]
                    drawColor = (255, 255, 255)

        ####################################
        # Drawing Mode
        ####################################

        elif (len(fingers) == 1 and fingers[0] == 8)and (drawColor==(0, 0, 255) or drawColor==(0, 255, 255) or drawColor==(255, 0, 0) or drawColor == (255, 255, 255)
):

            cx = lmList[8][1]
            cy = lmList[8][2]

            # Draw only inside active region
            if not (
                    toolbar_width + frameR < cx < w - frameR
                    and
                    frameR < cy < h - frameR
            ):
                xp, yp = 0, 0
                continue

            # Smooth the movement
            if xp == 0 and yp == 0:
                xp, yp = cx, cy

            cx = xp + (cx - xp) // smoothening
            cy = yp + (cy - yp) // smoothening

            cv.circle(img, (cx, cy), 10, drawColor, cv.FILLED)

            if drawColor == (255, 255, 255):

                cv.line(
                    imageCanvas,
                    (xp, yp),
                    (cx, cy),
                    drawColor,
                    eraser_thickness
                )

            else:

                cv.line(
                    imageCanvas,
                    (xp, yp),
                    (cx, cy),
                    drawColor,
                    brush_thickness
                )

            xp, yp = cx, cy

        else:
            xp, yp = 0, 0

    ####################################
    # Merge Canvas with Webcam
    ####################################

    imgGray=cv.cvtColor(
        imageCanvas,
        cv.COLOR_BGR2GRAY
    )

    _,mask=cv.threshold(imgGray,250,255,cv.THRESH_BINARY_INV)
    mask_inv=cv.bitwise_not(mask)
    mask=cv.cvtColor(mask,cv.COLOR_GRAY2BGR)
    mask_inv = cv.cvtColor(mask_inv, cv.COLOR_GRAY2BGR)

    img = cv.bitwise_and(img,mask_inv)
    drawing=cv.bitwise_and(imageCanvas,mask)
    img =cv.add(img,drawing)

    ####################################
    # Toolbar
    ####################################

    img[0:720, 0:142] = header

    ####################################
    # FPS
    ####################################

    ctime = time.time()
    fps = 1 / (ctime - ptime) if ctime != ptime else 0
    ptime = ctime

    cv.putText(
        img,
        f"FPS: {int(fps)}",
        (1050, 50),
        cv.FONT_HERSHEY_PLAIN,
        2,
        (255, 0, 0),
        2
    )

    ####################################
    # Display
    ####################################

    cv.imshow("Virtual Painter", img)

    key = cv.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('c'):
        imageCanvas[:] = 255

############################################
# Cleanup
############################################

cap.release()
cv.destroyAllWindows()