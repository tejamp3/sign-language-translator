from flask import Flask, render_template, Response
import pickle
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__, template_folder='/Users/bandipurushowtham/working/sign language translator/templates')

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'K', 10: 'L', 11:'M', 
               12:'N', 13:'O', 14:'P', 15:'Q', 16:'R', 17:'S', 18:'T', 19:'W', 20:'X', 21:'Y', 22:'Hello', 
               23:'Thank you', 24:'Z'
}

def preprocess_data(data_aux, expected_length):
    if len(data_aux) < expected_length:
        return np.pad(data_aux, (0, expected_length - len(data_aux)), 'constant')
    return data_aux[:expected_length]

expected_feature_length = model.n_features_in_

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Initialize fresh data for each hand
                data_aux = []
                x_coords = []
                y_coords = []

                # Collect landmarks
                for landmark in hand_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y
                    x_coords.append(x)
                    y_coords.append(y)

                # Normalize coordinates
                min_x = min(x_coords)
                min_y = min(y_coords)
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min_x)
                    data_aux.append(landmark.y - min_y)

                # Process and predict
                data_aux = preprocess_data(data_aux, expected_feature_length)
                try:
                    prediction = model.predict([data_aux])[0]
                    predicted_char = labels_dict[int(prediction)]
                except (KeyError, ValueError) as e:
                    print(f"Prediction error: {e}")
                    predicted_char = "Unknown"

                # Draw annotations
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Calculate bounding box
                x1 = int(min(x_coords) * W) - 10
                y1 = int(min(y_coords) * H) - 10
                x2 = int(max(x_coords) * W) + 10
                y2 = int(max(y_coords) * H) + 10

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_char, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
