import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
from gtts import gTTS

model_dict = pickle.load(open('./sign_model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
    7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N',
    13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T',
    19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y', 24: 'SPACE'
}

engine = pyttsx3.init()
engine.setProperty('rate', 150)  
engine.setProperty('volume', 1.0)

string = ""  
current_gesture = ""  
stable_gesture = ""  
gesture_stable_count = 0  
THRESHOLD = 30
hand_present = False

text_file_path = "sign_to_text.txt"
audio_file_path = "sign_to_audio.mp3"

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_present = True
        
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            if x_ and y_:
                x_min, y_min = min(x_), min(y_)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - x_min)
                    data_aux.append(y - y_min)

                if len(data_aux) < 42:
                    data_aux.extend([0] * (42 - len(data_aux)))  
                elif len(data_aux) > 42:
                    data_aux = data_aux[:42]  

                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                if predicted_character == current_gesture:
                    gesture_stable_count += 1
                else:
                    current_gesture = predicted_character
                    gesture_stable_count = 0

                if gesture_stable_count >= THRESHOLD:
                    if stable_gesture != current_gesture:  
                        stable_gesture = current_gesture
                        if stable_gesture == 'SPACE':
                            string += " "
                        else:
                            string += stable_gesture
                        gesture_stable_count = 0

                x1 = int(x_min * W) - 10
                y1 = int(y_min * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10
                x1, y1 = max(0, x1), max(0, y1)  # Ensure within bounds
                x2, y2 = min(W, x2), min(H, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    else:
        if hand_present:
            hand_present = False
            current_gesture = ""
            stable_gesture = ""
            gesture_stable_count = 0

    cv2.putText(frame, f"String: {string}", (10, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('frame', frame)

    with open(text_file_path, "w") as text_file:
        text_file.write(string)
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
        tts = gTTS(string)
        tts.save(audio_file_path)
        print(f"Audio saved as {audio_file_path}")
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
