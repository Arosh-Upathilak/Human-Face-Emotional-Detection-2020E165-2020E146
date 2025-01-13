import numpy as np
import cv2
import tensorflow as tf

def capture_and_predict_emotions(model, emotion_dict):
    cap = cv2.VideoCapture(0)
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)

            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

            prediction = model.predict(cropped_img, verbose=0)
            maxindex = int(np.argmax(prediction))
            confidence_score = float(np.max(prediction)) * 100

            emotion_text = f"{emotion_dict[maxindex]}: {confidence_score:.1f}%"

            text_size = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            cv2.rectangle(frame,
                         (x + 20, y - 75),
                         (x + 20 + text_size[0], y - 35),
                         (0, 0, 0),
                         -1)

            cv2.putText(frame, emotion_text,
                       (x + 20, y - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1,
                       (255, 255, 255), 2,
                       cv2.LINE_AA)

        resized_frame = cv2.resize(frame, (1200, 800), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Emotion Detection', resized_frame)

        if cv2.waitKey(1) & 0xFF == 13:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = tf.keras.models.load_model('emotion_detection_model.h5')

    emotion_dict = {
        0: "Angry",
        1: "Disgusted",
        2: "Fearful",
        3: "Happy",
        4: "Neutral",
        5: "Sad",
        6: "Surprised"
    }

    capture_and_predict_emotions(model, emotion_dict)