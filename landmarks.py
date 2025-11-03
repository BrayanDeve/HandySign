import cv2
import mediapipe as mp

# Inicializa MediaPipe Hands e Drawing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Captura da webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=2,                # Detecta até duas mãos
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Espelha a imagem para efeito de selfie
        frame = cv2.flip(frame, 1)
        # Converte para RGB (MediaPipe trabalha com RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processa o frame
        results = hands.process(rgb_frame)

        # Verifica se há mãos detectadas
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Desenha os landmarks e conexões
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Pega dimensões da imagem
                h, w, _ = frame.shape

                # Desenha e numera cada ponto
                for i, lm in enumerate(hand_landmarks.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), cv2.FILLED)  # ponto vermelho
                    cv2.putText(frame, str(i), (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # Identifica se é mão esquerda ou direita
                if results.multi_handedness:
                    hand_label = results.multi_handedness[idx].classification[0].label
                    # Pega a posição do punho (landmark 0) para colocar o texto
                    x0 = int(hand_landmarks.landmark[0].x * w)
                    y0 = int(hand_landmarks.landmark[0].y * h)
                    cv2.putText(frame, hand_label, (x0 - 20, y0 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostra o resultado
        cv2.imshow("Dual Hand Tracker", frame)

        # Sai com ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()