import mediapipe as mp
import cv2 as cv

mp_posture = mp.solutions.pose
posture = mp_posture.Pose()
mp_drawing = mp.solutions.drawing_utils 

# cap = cv.VideoCapture(0)
cap = cv.VideoCapture("data/dance_video.mp4")

width, height = 800, 600

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    image = cv.resize(frame, (width, height))
    image = cv.flip(image, 1)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    results = posture.process(image)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_posture.POSE_CONNECTIONS)
        # print(results.pose_landmarks)

    ## To access each landmarks
    # if results.pose_landmarks:
    #     for idx, pose_landmarks in enumerate(results.pose_landmarks.landmark):
    #         cx, cy = int(pose_landmarks.x * width), int(pose_landmarks.y * height)
    #         cv.circle(image, (cx, cy), 5, (255, 255, 0), -1)
    #         cv.putText(image, str(idx), (cx, cy), cv.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
    #         print(f"{idx}: {cx}, {cy}")

    cv.imshow("Real-time pose estimator", cv.cvtColor(image, cv.COLOR_RGB2BGR))
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()