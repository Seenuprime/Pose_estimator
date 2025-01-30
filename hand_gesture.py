import mediapipe as mp
import cv2 as cv
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

## Finding Euclidean distance between two points 
def distance(pt1, pt2):
    """
        Finds the Euclidean distance between two 2D points.

        Args:
            pt1: Landmark object (contains `x` and `y` attributes representing the coordinates of point 1).
            pt2: Landmark object (contains `x` and `y` attributes representing the coordinates of point 2).

        Returns:
            float: The Euclidean distance between pt1 and pt2.
    """
    return np.linalg.norm(np.array([pt1.x, pt1.y]) - np.array([pt2.x, pt2.y]))

def is_thums_up(landmarks):
    global thumb_landmark
    thumb_landmark = landmarks[4]
    index_landmark = landmarks[8]
    middle_landmark = landmarks[12]
    ring_landmark = landmarks[16]
    pinky_landmark = landmarks[20]

    index_proximal = landmarks[5]
    middle_proximal = landmarks[9]
    ring_proximal = landmarks[13]
    pinky_proximal = landmarks[17]

    thumb_up = thumb_landmark.y < index_landmark.y

    min_finger_distance = 0.1

    fingers_folded = (distance(index_landmark, index_proximal) < min_finger_distance and
                      distance(middle_landmark, middle_proximal) < min_finger_distance and
                      distance(ring_landmark, ring_proximal) < min_finger_distance and 
                      distance(pinky_landmark, pinky_proximal) < min_finger_distance)
    
    return thumb_up, fingers_folded


def is_pease(landmarks):
    index_tip_landmark = landmarks[8]
    middle_tip_landmark = landmarks[12]
    ring_tip_landmark = landmarks[16]
    pinky_tip_landmark = landmarks[20]
    thumb_tip_landmark = landmarks[4]

    index_pip_landmark = landmarks[7]
    middle_pip_landmark = landmarks[11]

    other_fingers = (ring_tip_landmark.y > thumb_tip_landmark.y and pinky_tip_landmark.y > thumb_tip_landmark.y)
    pease_fingers =  index_tip_landmark.y < index_pip_landmark.y and middle_tip_landmark.y < middle_pip_landmark.y

    return other_fingers, pease_fingers
    

cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    image = cv.resize(frame, (700, 500))
    image = cv.flip(image, 1)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            ## if you want to draw landmarks on the image or frame
            # mp_drawing.draw_landmarks(image, hand_landmarks)

            landmarks = hand_landmarks.landmark
            
            thums_up, fingers_folded = is_thums_up(landmarks)
            other_fingers, pease_fingers = is_pease(landmarks) 

            if thums_up and fingers_folded:
                cv.putText(image, "Thumbs Up", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (197, 66, 245), 1)
            
            elif other_fingers and pease_fingers:
                cv.putText(image, "Pease Sign", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (197, 66, 245), 1)

    cv.imshow("Got the Gesture", cv.cvtColor(image, cv.COLOR_RGB2BGR))
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()