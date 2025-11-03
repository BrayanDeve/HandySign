#pip install opencv-python mediapipe

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Carrega imagens dos sinais (PNG com fundo transparente)
sign_images = {
    "Like": cv2.imread("icons/like.png", cv2.IMREAD_UNCHANGED),
    "Dislike": cv2.imread("icons/dislike.png", cv2.IMREAD_UNCHANGED),
    "OK": cv2.imread("icons/ok.png", cv2.IMREAD_UNCHANGED),
    "Stop": cv2.imread("icons/stop.png", cv2.IMREAD_UNCHANGED)
}

ICON_SIZE = 80
icon_positions = {
    "Like": (20, 20),
    "Dislike": (120, 20),
    "OK": (220, 20),
    "Stop": (320, 20)
}

GESTURE_THRESHOLD = 5  # Frames consecutivos para validar

# --- Variáveis de controle ---
current_gesture = None       # Gesto que está sendo exibido
gesture_buffer = None        # Gesto detectado no frame atual
gesture_counter = 0          # Contador de estabilidade

def overlay_image(frame, img, x, y):
    """Sobrepõe PNG com transparência sobre o frame"""
    h, w, _ = img.shape
    for i in range(h):
        for j in range(w):
            if img[i, j, 3] != 0:
                if 0 <= y+i < frame.shape[0] and 0 <= x+j < frame.shape[1]:
                    frame[y+i, x+j] = img[i,j,:3]
    return frame

def distance(a, b):
    return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def detect_gesture(landmarks):
    thumb = [landmarks[i] for i in range(1,5)]
    fingers = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
    wrist = landmarks[0]

    # Verifica se todos os dedos exceto polegar estão fechados
    fingers_closed = all([f.y > landmarks[i].y for f, i in zip(fingers, [6,10,14,18])])

    if fingers_closed:
        # Polegar para cima
        if thumb[3].y < thumb[0].y:
            return "Like"
        # Polegar para baixo
        elif thumb[3].y > thumb[0].y:
            return "Dislike"

    # OK: polegar e indicador se tocam
    if distance(landmarks[4], landmarks[8]) < 0.05:
        return "OK"

    # Stop: todos os dedos abertos
    all_fingers_up = all([
        landmarks[8].y < landmarks[6].y,
        landmarks[12].y < landmarks[10].y,
        landmarks[16].y < landmarks[14].y,
        landmarks[20].y < landmarks[18].y
    ])
    if all_fingers_up:
        return "Stop"

    return None

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gesture_buffer = None  # reset do buffer a cada frame

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Mostra Left ou Right
                if results.multi_handedness:
                    label = results.multi_handedness[idx].classification[0].label
                    h_f, w_f, _ = frame.shape
                    x = int(hand_landmarks.landmark[0].x * w_f)
                    y = int(hand_landmarks.landmark[0].y * h_f)
                    cv2.putText(frame, label, (x - 20, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                # Detecta gesto
                gesture_detected = detect_gesture(hand_landmarks.landmark)
                if gesture_detected:
                    gesture_buffer = gesture_detected

        # --- Lógica de estabilidade ---
        if gesture_buffer == current_gesture:
            gesture_counter = 0  # já está ativo, mantém
        elif gesture_buffer is not None:
            gesture_counter += 1
            if gesture_counter >= GESTURE_THRESHOLD:
                current_gesture = gesture_buffer
                gesture_counter = 0
        else:
            gesture_counter = 0
            # current_gesture = None  # opcional: desativa se nenhum gesto

        # Desenha os ícones em caixinha fixa
        for gesture_name, pos in icon_positions.items():
            x, y = pos
            bg_color = (0,200,0) if gesture_name == current_gesture else (50,50,50)
            cv2.rectangle(frame, (x-5, y-5), (x+ICON_SIZE+5, y+ICON_SIZE+5), bg_color, -1)
            icon = cv2.resize(sign_images[gesture_name], (ICON_SIZE, ICON_SIZE), interpolation=cv2.INTER_AREA)
            frame = overlay_image(frame, icon, x, y)

        cv2.imshow("Hand & Gesture Tracker", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()