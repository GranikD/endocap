import cv2

cap = cv2.VideoCapture("capsule_video.mp4")
fps = 5
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if count % fps == 0:
        cv2.imwrite(f"frames/frame_{count}.jpg", frame)

    count += 1

cap.release()
