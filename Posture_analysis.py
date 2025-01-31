import mediapipe as mp
import cv2 as cv
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv.VideoCapture("data/posture_video.mp4")
# cap = cv.VideoCapture(0)
width, height = 800, 700

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    image = cv.resize(frame, (width, height))
    image = cv.flip(image, 1)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    results = pose.process(image)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = results.pose_landmarks.landmark
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

        shoulder_mid = (
            int((left_shoulder.x + right_shoulder.x) / 2 * width),
            int((left_shoulder.y + right_shoulder.y) / 2 * height)
        )

        hip_mid = (
            int((left_hip.x + right_hip.x) / 2 * width),
            int((left_hip.y + right_hip.y) / 2 * height)
        )

        shoulder_to_hip_distance = np.linalg.norm(np.array(shoulder_mid) - np.array(hip_mid))
        print(shoulder_to_hip_distance)

        ## Compute dynamic threshold based on hip width
        hip_width = np.linalg.norm(np.array([left_hip.x * width, left_hip.y * height]) - np.array([right_hip.x * width, right_hip.y * height]))
        print(hip_width)
        threshold = hip_width + 0.6

        
        cv.circle(image, tuple(shoulder_mid), 5, (255, 0, 0), -1)
        cv.circle(image, tuple(hip_mid), 5, (255, 0, 0), -1)
        cv.line(image, shoulder_mid, hip_mid, (255, 255, 255), 2)

        if shoulder_to_hip_distance < threshold:
            cv.putText(image, "Poor Posture", (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        else:
            cv.putText(image, "Good Posture", (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            

    cv.imshow("Posture Check", cv.cvtColor(image, cv.COLOR_RGB2BGR))
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()        
cv.destroyAllWindows()
