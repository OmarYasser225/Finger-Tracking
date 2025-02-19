import cv2
from cvzone.HandTrackingModule import HandDetector  

# Open Camera
cap = cv2.VideoCapture(0)

# Initialize the hand detector
detector = HandDetector(detectionCon=0.5, maxHands=1)

while True:
    success, img = cap.read()
    if not success:
        continue  # Skip frame if not read properly

    hands , img = detector.findHands(img)  # Detect hands   

    if hands:
        # Extract first detected hand
        hand = hands[0]

        # Get the list of landmarks
        lmList = hand["lmList"]  # List of 21 hand landmark points

        if lmList:
            fingers = detector.fingersUp(hand)  # Check which fingers are up
            print(fingers)
        

    cv2.imshow("Image", img)