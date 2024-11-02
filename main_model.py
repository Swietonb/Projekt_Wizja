import cv2
import pickle
from typing import Tuple, List
from classifier import HandClassifier
import mediapipe as mp
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

#---ZMIENNA GLOBLALNA DO UZYCIA W INNYCH PLIKACH---
gesture = None
#--- użycie --- from main_model import gesture

def load_training_data():
    """
    Wczytuje dane uczące z pliku 'training_data.pkl'. Jeśli plik nie istnieje, wywołuje funkcję
    prepare_data() z modułu prepare_data do wygenerowania i zapisania danych.

    Returns:
        Tuple[List[List[float]], List[str]]: Lista znormalizowanych danych uczących i lista etykiet.

    Raises:
        FileNotFoundError: Jeśli plik nie istnieje i nie udało się go utworzyć.
        IOError: Jeśli wystąpił błąd we wczytywaniu pliku.
    """
    try:
        with open("training_data.pkl", "rb") as f:
            data, labels = pickle.load(f)
        print("Dane uczące wczytane z pliku 'training_data.pkl'.")
    except (FileNotFoundError, IOError):
        print("Brak pliku 'training_data.pkl'. Generowanie danych uczących...")
        from prepare_data import prepare_data  # Import funkcji przygotowującej dane
        prepare_data()
        # Po wygenerowaniu danych uczących wczytaj je ponownie
        with open("training_data.pkl", "rb") as f:
            data, labels = pickle.load(f)
    return data, labels

def run_recognition() -> None:
    global gesture
    """
    Główna funkcja aplikacji, która wczytuje dane uczące, inicjalizuje klasyfikator,
    uruchamia kamerę, przetwarza obrazy oraz wyświetla wyniki w czasie rzeczywistym.
    """
    # Wczytanie danych uczących lub wygenerowanie ich, jeśli plik nie istnieje
    data, labels = load_training_data()

    # Inicjalizacja klasyfikatora z danymi uczącymi
    classifier = HandClassifier(n_neighbors=5, vote_threshold=0.6, distance_threshold=0.9)
    classifier.load_data(data, labels)
    classifier.train()

    # Inicjalizacja MediaPipe i kamery
    mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Nie udało się otworzyć kamery.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Nie udało się pobrać klatki z kamery.")
            break

        # Przetwarzanie obrazu
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            normalized_landmarks = classifier.preprocessor.normalize_hand(landmarks)
            predicted_label = classifier.predict(normalized_landmarks)


        #----------WYŚWIETLANIE OKNA Z OBRAZEM Z KAMERY I WYNIKAMI-----------
        #---Zakomentować ten fragment aby nie wyświetlać wyników-------------

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            cv2.putText(frame, f'Gest: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'Nie wykryto dloni', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            predicted_label = "Nie wykryto dloni"

        cv2.imshow('Rozpoznawanie gestow', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # 'ESC', aby wyjść
            break
        #---------------------------------------------------------------------

        #---ZAKOMENTOWAC PRZY WYSWIETLANIU WYNIKOW
        # else:
        #     predicted_label = "Nie wykryto dloni"

        #---Wypisywanie wyniku w konsoli---
        # print(predicted_label)

        gesture = predicted_label

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_recognition()
