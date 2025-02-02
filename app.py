from flask import Flask, render_template, Response, jsonify, send_file
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from gtts import gTTS

model_dict = pickle.load(open('./sign_model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
    7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N',
    13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T',
    19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y', 24: 'SPACE',
}

app = Flask(__name__, 
            static_folder="static", 
            template_folder="templates")

string = ""
current_gesture = ""
stable_gesture = ""
gesture_stable_count = 0
THRESHOLD = 30
hand_present = False

cap = None

def generate_frames():
    global string, current_gesture, stable_gesture, gesture_stable_count, hand_present


    while cap and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_present = True
            
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmarks
                data_aux = []
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]

                x_min, y_min = min(x_), min(y_)
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - x_min)
                    data_aux.append(lm.y - y_min)

                if len(data_aux) < 42:
                    data_aux.extend([0] * (42 - len(data_aux)))

                # Predict gesture
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                # Gesture confirmation
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

        else:
            if hand_present:
                hand_present = False
                current_gesture = ""
                stable_gesture = ""
                gesture_stable_count = 0
        
        cv2.putText(frame, f"Output: {string}", (10, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    global string
    string = ""
    return render_template('index.html')


@app.route('/start_camera', methods=['POST'])
def start_camera():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({'success': False, 'message': 'Could not open webcam'})
        
    return jsonify({'success': True})


@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global cap
    if cap and cap.isOpened():
        cap.release()
        
    return jsonify({'success': True})


@app.route('/video_feed')
def video_feed():
    global cap
    if not cap or not cap.isOpened():
        return "Camera not started", 400
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_text', methods=['GET'])
def get_text():
    return jsonify({'recognized_text': string})


@app.route('/download_text', methods=['GET'])
def download_text():
    global string
    text_file_path = "sign_to_text.txt"
    with open(text_file_path, "w") as text_file:
        text_file.write(string)
    return send_file(text_file_path, as_attachment=True)


@app.route('/download_audio', methods=['GET'])
def download_audio():
    global string
    language = 'en'
    audio_file_path = "sign_to_audio.mp3"
    tts = gTTS(text=string, lang=language, slow=False)
    tts.save(audio_file_path)
    return send_file(audio_file_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
