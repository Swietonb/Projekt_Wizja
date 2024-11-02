# --------------------------------------------------------------------------------------------
# --- PLIK DO ODCZYTU PUNKTÓW KLUCZOWYCH DŁONI Z OBRAZÓW DANYCH UCZĄCYCH ---------------------
# --------------------------------------------------------------------------------------------
# --- Uruchomić w przypadku zmiany/dodania danych uczących raz przed wywołaniem main_model.py -------
# --------------------------------------------------------------------------------------------

# W przypadku dodania nowego gestu dodaj nazwę gestu i ścieżkę do danych uczących (.jpg lub .png)
GESTURES = {
    "play": "data/open_hand",
    "stop": "data/closed_hand",
    "skip": "data/two_fingers",
    "backwards": "data/thumb_pinky",
}

import cv2
import os
import pickle
import mediapipe as mp
from preprocessing import Preprocessor
from typing import List, Tuple


def load_images(directory: str, label: str) -> List[Tuple]:
    """
    Wczytuje obrazy z katalogu i przypisuje im etykietę (odpowiednią ze zmiennej GESTURES).

    Args:
        directory (str): Ścieżka do katalogu z obrazami.
        label (str): Etykieta dla danego zestawu obrazów.

    Returns:
        List[Tuple]: Lista krotek zawierających obraz i etykietę w każdej.
    """
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append((img, label))
    return images


def prepare_data() -> None:
    """
    Przetwarza obrazy z danymi uczącymi i zapisuje znormalizowane punkty kluczowe dłoni do pliku.

    Funkcja korzysta z MediaPipe do identyfikacji punktów kluczowych dłoni, a następnie normalizuje
    je za pomocą klasy Preprocessor. Znormalizowane dane i etykiety są zapisywane w pliku 'training_data.pkl'.
    """
    mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    preprocessor = Preprocessor()

    data = []
    labels = []

    # Przetwarzanie obrazów dla każdego gestu zdefiniowanego w GESTURES
    for label, directory in GESTURES.items():
        images = load_images(directory, label)
        for img, _ in images:
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(image_rgb)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                normalized_landmarks = preprocessor.normalize_hand(landmarks)
                data.append(normalized_landmarks)
                labels.append(label)

    # Zapisanie przygotowanych danych do pliku
    with open("training_data.pkl", "wb") as f:
        pickle.dump((data, labels), f)
    print("Dane uczące zostały zapisane do pliku 'training_data.pkl'.")


if __name__ == "__main__":
    prepare_data()
