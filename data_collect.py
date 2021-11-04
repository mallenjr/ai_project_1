import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils


thumbs = open('fist.dat', 'wb')

count = 0

# Initialize the webcam
cap = cv2.VideoCapture(1)

while True:
    # Read each frame from the webcam
    _, image = cap.read()

    x, y, c = image.shape

    # Flip the frame vertically
    image = cv2.flip(image, 1)
    imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cv2.imshow("raw", imagergb)

    # Get hand landmark prediction
    result = hands.process(imagergb)
    
    className = ''

    landmarks = []

    # post process the result
    if result.multi_hand_landmarks:
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(image, handslms, mpHands.HAND_CONNECTIONS)

    cv2.putText(image, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", image) 

    key = cv2.waitKey(1)
    
    if key == 48 and len(landmarks) > 0:
      print('save fist - count ', count)
      count += 1
    elif key == 49 and len(landmarks) > 0:
      print('save stop - count ', count)
      count += 1
    elif key == 50 and len(landmarks) > 0:
      print('save peace - count ', count)
      count += 1
    elif key == 51 and len(landmarks) > 0:
      print('save thumbs - count ', count)
      count += 1
    elif key == ord('q'):
      break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()