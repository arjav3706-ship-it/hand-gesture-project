from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

app = Flask(__name__)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def fingers_up(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    fingers.append(hand_landmarks.landmark[tips[0]].x <
                   hand_landmarks.landmark[tips[0]-1].x)

    for i in range(1, 5):
        fingers.append(hand_landmarks.landmark[tips[i]].y <
                       hand_landmarks.landmark[tips[i]-2].y)

    return fingers

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture = "Detecting..."

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                f = fingers_up(hand_landmarks)

                if f == [True, False, False, False, False]:
                    gesture = "👍 GOOD LUCK / LIKE"
                elif f == [False, False, False, False, False]:
                    gesture = "✊ POWER / NO"
                elif f == [False, True, False, False, False]:
                    gesture = "☝️ NUMBER ONE"
                elif f == [False, True, True, False, False]:
                    gesture = "✌️ VICTORY / PEACE"
                elif f == [True, True, True, False, False]:
                    gesture = "🤟 I LOVE YOU"
                elif f == [False, True, True, True, True]:
                    gesture = "✋ HELLO / STOP"
                elif f == [False, False, False, False, True]:
                    gesture = "👎 DISLIKE / NO"

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, gesture, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)