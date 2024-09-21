import cv2
import numpy as np
from utils.media_pipe import mediapipe_detection, draw_styled_landmarks
from utils.data_processing import extract_keypoints
from utils.visualization import prob_viz
from models.lstm_models import build_lstm_model, load_model
import mediapipe as mp

# Set actions, sequence, and threshold
actions = np.array(['hello', 'thanks', 'goodbye', 'yes', 'no'])
sequence = []
sentence = []
threshold = 0.8

# Load or build the model
model = load_model('data/action.h5')  # Load pre-trained model
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

# Initialize camera and mediapipe
cap = cv2.VideoCapture(0)
with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        image, results = mediapipe_detection(frame, holistic)

        draw_styled_landmarks(image, results)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            image = prob_viz(res, actions, image, colors)

        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
