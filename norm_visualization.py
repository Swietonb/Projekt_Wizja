import cv2
import mediapipe as mp
from preprocessing import Preprocessor

# Inicjalizacja MediaPipe dla rozpoznawania dłoni
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Ścieżka do obrazu dłoni
image_path = 'data/closed_hand/A_P_hgr2B_id08_1.jpg'
image = cv2.imread(image_path)
width = 800
height = 1050
original_image_resized = cv2.resize(image, (width, height))  # Zmniejszony obraz dla wygodniejszego wyświetlania
image_rgb = cv2.cvtColor(original_image_resized, cv2.COLOR_BGR2RGB)

# Przetwarzanie obrazu z użyciem MediaPipe
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
results = hands.process(image_rgb)

if results.multi_hand_landmarks:
    hand_landmarks = results.multi_hand_landmarks[0]
    landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

    # Wyświetlanie oryginalnego szkieletu dłoni
    mp_drawing.draw_landmarks(original_image_resized, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Normalizacja punktów dłoni
    preprocessor = Preprocessor()
    normalized_landmarks = preprocessor.normalize_hand(landmarks)

    # Rysowanie znormalizowanego szkieletu dłoni
    for start_idx, end_idx in mp_hands.HAND_CONNECTIONS:
        start_point = normalized_landmarks[start_idx * 3:start_idx * 3 + 2]
        end_point = normalized_landmarks[end_idx * 3:end_idx * 3 + 2]

        # Przekształcenie punktów na koordynaty względem szerokości i wysokości
        cv2.line(original_image_resized,
                 (int((start_point[0] + 1) * width / 2), int((start_point[1] + 1) * height / 2)),
                 (int((end_point[0] + 1) * width / 2), int((end_point[1] + 1) * height / 2)),
                 (0, 0, 255), 2)

    # Rysowanie punktów
    for i in range(0, len(normalized_landmarks), 3):
        x, y = (normalized_landmarks[i] + 1) * width / 2, (normalized_landmarks[i + 1] + 1) * height / 2
        cv2.circle(original_image_resized, (int(x), int(y)), 5, (0, 255, 0), -1)

    # Rysowanie linii na osi y = 0
    y_axis_center = height // 2
    cv2.line(original_image_resized, (0, y_axis_center), (width, y_axis_center), (255, 255, 255), 1)  # Linia na niebiesko

# Wyświetlanie obrazu z oryginalnym i znormalizowanym szkieletem
cv2.imshow('Oryginalny i znormalizowany szkielet dłoni', original_image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
