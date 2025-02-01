import mediapipe as mp
import cv2 as cv

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawings = mp.solutions.drawing_utils

cap = cv.VideoCapture('data/push_up.mp4')

pushup_count = 0
push = 'up'

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv.resize(frame, (800, 600))
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    results = pose.process(image)

    if results.pose_landmarks:
        mp_drawings.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = results.pose_landmarks.landmark
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]

        shoulder_mid = (right_shoulder.y + left_shoulder.y) / 2
        elbow_mid = (right_elbow.y + left_elbow.y) / 2

        height, width, _ = image.shape

        threshold = shoulder_mid + 0.13
        # cv.circle(image, (int(width/2), int(threshold *height)), 5, (255, 0, 0), -1)
        if threshold > elbow_mid :
            if push == 'up':
                pushup_count += 1
                push = 'down'
        else:
            push = 'up'

        cv.putText(image, f"Push up count: {pushup_count}.", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv.imshow("Push up detectors", cv.cvtColor(image, cv.COLOR_RGB2BGR))
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()