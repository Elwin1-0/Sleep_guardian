import cv2
from tensorflow.keras.models import load_model
import winsound
import threading
import numpy as np


model = load_model(r"C:\Users\elwin\OneDrive\Desktop\Project datasets\eye_sleep.h5")
face_cascade=cv2.CascadeClassifier(r"C:\Users\elwin\OneDrive\Desktop\Project datasets\Cascades\haarcascade_frontalface_default (1).xml")
eye_cascade=cv2.CascadeClassifier(r"C:\Users\elwin\OneDrive\Desktop\Project datasets\Cascades\haarcascade_eye (1).xml")


def eye_process(eye):
    gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    resized = cv2.resize(equalized, (128, 128))
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    normalized = rgb / 255.0
    return np.expand_dims(normalized, axis=0)


alarm_on = False


def play_alarm():
    global alarm_on
    while alarm_on:
        winsound.Beep(4000, 500)


close_counter = 0
alarm_threshold = 20
capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    # Default status for this frame
    current_status = "eyes closed"

    for (x, y, w, h) in faces:
        face_roi_gray = gray[y:y + h, x:x + w]
        face_roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.1, 10)  # Higher minNeighbors reduces noise


        if len(eyes) > 0:
            for (ex, ey, ew, eh) in eyes:
                eye_img = face_roi_color[ey:ey + eh, ex:ex + ew]
                ready = eye_process(eye_img)
                prediction = model.predict(ready, verbose=0)

                if prediction[0][0] > 0.5:
                    current_status = "eyes open"
                    break


        if current_status == "eyes open":
            close_counter = 0
            alarm_on = False
        else:
            close_counter += 1


        color = (0, 255, 0) if current_status == "eyes open" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"Status :{current_status}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Alarm Trigger
        if close_counter > alarm_threshold:
            cv2.putText(frame, 'WAKE UP !!', (x, y - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 187, 255), 2)
            if not alarm_on:
                alarm_on = True
                threading.Thread(target=play_alarm, daemon=True).start()

    cv2.imshow('Sleep Guardian', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        alarm_on = False
        break

capture.release()
cv2.destroyAllWindows()