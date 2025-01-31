import mediapipe as mp
import cv2 as cv

mp_pose = mp.solutions.pose
pose = mp_pose.Pose() 
mp_drawing = mp.solutions.drawing_utils

cap = cv.VideoCapture('data/squat_video.mp4')

squat_count = 0
squat = 'up'

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (800, 700))

    results = pose.process(image)

    landmarks = results.pose_landmarks
    if landmarks:
        mp_drawing.draw_landmarks(image, landmarks, mp_pose.POSE_CONNECTIONS)

        right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value]

        hip_mid = (right_hip.y + left_hip.y) / 2
        knee_mid = (right_knee.y + left_knee.y) / 2

        if hip_mid > knee_mid:
            if squat == 'up':
                squat_count += 1
                squat = 'down'
        else:
            squat = 'up'

        cv.putText(image, f"Squat Count {str(squat_count)}", (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 244), 2)

    cv.imshow("Frames", cv.cvtColor(image, cv.COLOR_RGB2BGR))
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()