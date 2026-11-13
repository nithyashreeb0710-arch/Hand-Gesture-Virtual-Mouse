
import cv2
import mediapipe as mp
import pyautogui
import time
import math


class HandDetector:
    def __init__(self, maxHands=2, detectionCon=0.7, trackCon=0.7):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            max_num_hands=maxHands,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findPosition(self, img):1
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        allHands = []

        if results.multi_hand_landmarks:
            for i, handLms in enumerate(results.multi_hand_landmarks):
                self.mpDraw.draw_landmarks(
                    img, handLms, self.mpHands.HAND_CONNECTIONS
                )

                lmList = []
                h, w, c = img.shape

                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])

                handType = results.multi_handedness[i].classification[0].label

                allHands.append({
                    "landmarks": lmList,
                    "type": handType
                })

        return allHands


def countFingers(lmList, handType):
    fingers = []

    if handType == "Right":
        fingers.append(1 if lmList[4][1] < lmList[3][1] else 0)
    else:
        fingers.append(1 if lmList[4][1] > lmList[3][1] else 0)

    tips = [8, 12, 16, 20]

    for tip in tips:
        fingers.append(1 if lmList[tip][2] < lmList[tip - 2][2] else 0)

    return fingers


def isPinching(lmList):
    thumb = lmList[4]
    index = lmList[8]
    wrist = lmList[0]
    middleMcp = lmList[9]

    distance = math.hypot(
        thumb[1] - index[1],
        thumb[2] - index[2]
    )

    palmSize = math.hypot(
        wrist[1] - middleMcp[1],
        wrist[2] - middleMcp[2]
    )

    if palmSize == 0:
        return False

    return distance / palmSize < 0.35


def recognizeGesture(lmList, fingers, pinch):
    if pinch:
        return "PINCH"

    if fingers == [1, 1, 1, 1, 1]:
        return "OPEN PALM"

    if fingers == [0, 0, 0, 0, 0]:
        return "FIST"

    if fingers == [0, 1, 1, 0, 0]:
        return "PEACE SIGN"

    if fingers == [0, 1, 0, 0, 0]:
        return "POINTING"

    if fingers == [0, 1, 1, 1, 0]:
        return "THREE FINGERS"

    if fingers[1:] == [0, 0, 0, 0]:
        thumbTipY = lmList[4][2]
        thumbBaseY = lmList[2][2]

        if thumbTipY < thumbBaseY - 20:
            return "THUMBS UP"

        if thumbTipY > thumbBaseY + 20:
            return "THUMBS DOWN"

    return "UNKNOWN"


def drawHandInfo(img, hand, totalFingers, gesture):
    lmList = hand["landmarks"]
    h, w, c = img.shape

    palmX = sum(lmList[i][1] for i in [0, 5, 9, 13, 17]) // 5
    palmY = sum(lmList[i][2] for i in [0, 5, 9, 13, 17]) // 5

    boxWidth = 180
    boxHeight = 65

    x = max(5, min(palmX - boxWidth // 2, w - boxWidth - 5))
    y = min(palmY + 65, h - boxHeight - 5)

    cv2.rectangle(
        img,
        (x, y),
        (x + boxWidth, y + boxHeight),
        (30, 30, 30),
        cv2.FILLED
    )

    cv2.putText(
        img,
        f"{hand['type']} HAND",
        (x + 8, y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        2
    )

    cv2.putText(
        img,
        f"Fingers: {totalFingers}",
        (x + 8, y + 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )

    cv2.putText(
        img,
        gesture,
        (x + 8, y + 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=2)

    pyautogui.FAILSAFE = True

    screenWidth, screenHeight = pyautogui.size()

    mouseMode = False
    clickPressed = False

    smoothening = 5
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0

    pTime = 0

    windowName = "Hand Gesture Control"
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

    while True:
        success, img = cap.read()

        if not success:
            break

        img = cv2.flip(img, 1)
        h, w, c = img.shape

        allHands = detector.findPosition(img)

        cv2.putText(
            img,
            f"Hands Detected: {len(allHands)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            img,
            f"Mouse Mode: {'ON' if mouseMode else 'OFF'}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if mouseMode else (0, 0, 255),
            2
        )

        mouseHand = None

        for hand in allHands:
            lmList = hand["landmarks"]
            handType = hand["type"]

            fingers = countFingers(lmList, handType)
            totalFingers = sum(fingers)

            pinch = isPinching(lmList)
            gesture = recognizeGesture(lmList, fingers, pinch)

            drawHandInfo(img, hand, totalFingers, gesture)

            for tip in [4, 8, 12, 16, 20]:
                cv2.circle(
                    img,
                    (lmList[tip][1], lmList[tip][2]),
                    7,
                    (255, 0, 255),
                    cv2.FILLED
                )

            if handType == "Right":
                mouseHand = hand

        if mouseHand is None and len(allHands) > 0:
            mouseHand = allHands[0]

        if mouseMode and mouseHand is not None:
            lmList = mouseHand["landmarks"]

            indexX = lmList[8][1]
            indexY = lmList[8][2]

            screenX = int(indexX * screenWidth / w)
            screenY = int(indexY * screenHeight / h)

            clocX = plocX + (screenX - plocX) / smoothening
            clocY = plocY + (screenY - plocY) / smoothening

            pyautogui.moveTo(
                screenWidth - clocX,
                clocY
            )

            plocX, plocY = clocX, clocY

            pinch = isPinching(lmList)

            if pinch and not clickPressed:
                pyautogui.click()
                clickPressed = True

            elif not pinch:
                clickPressed = False

        else:
            clickPressed = False

        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime != pTime else 0
        pTime = cTime

        cv2.putText(
            img,
            f"FPS: {int(fps)}",
            (w - 130, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        cv2.imshow(windowName, img)
        cv2.resizeWindow(windowName, 1000, 700)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('m'):
            mouseMode = not mouseMode
            clickPressed = False

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
