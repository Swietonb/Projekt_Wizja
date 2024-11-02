from sklearn.neighbors import KNeighborsClassifier
from preprocessing import Preprocessor
import mediapipe as mp
from collections import deque
from typing import List


class HandClassifier:
    def __init__(self, n_neighbors: int = 3, vote_threshold: float = 0.6, distance_threshold: float = 0.6) -> None:
        """
        Inicjalizuje klasyfikator gestów dłoni oparty na KNN, buforze predykcji i MediaPipe do ekstrakcji punktów charakterystycznych.

        Args:
            n_neighbors (int): Liczba sąsiadów do rozważenia w modelu KNN.
            vote_threshold (float): Próg głosów wymagany do przyjęcia predykcji.
            distance_threshold (float): Maksymalna odległość dla najbliższego sąsiada, poniżej której predykcja jest uznawana za wiarygodną.
        """
        self.model = self.model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='kd_tree')
        self.vote_threshold = vote_threshold
        self.distance_threshold = distance_threshold
        self.data = []
        self.labels = []
        self.preprocessor = Preprocessor()

        # Bufor predykcji
        self.prediction_history = deque(maxlen=10)

        # Inicjalizacja MediaPipe dla wyciągania punktów charakterystycznych dłoni
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    def load_data(self, data: List[List[float]], labels: List[str]) -> None:
        """
        Ładuje dane uczące bezpośrednio do klasyfikatora.

        Args:
            data (List[List[float]]): Dane uczące, znormalizowane punkty charakterystyczne dłoni.
            labels (List[str]): Etykiety odpowiadające gestom.
        """
        self.data = data
        self.labels = labels

    def train(self) -> None:
        """
        Trenuje model KNN na załadowanych danych uczących.
        """
        self.model.fit(self.data, self.labels)

    def predict(self, features: List[float]) -> str:
        """
        Dokonuje predykcji na podstawie dostarczonych cech (punktów charakterystycznych dłoni).

        Args:
            features (List[float]): Znormalizowane cechy dłoni do predykcji gestu.

        Returns:
            str: Etykieta przewidywanego gestu lub 'brak gestu', jeśli predykcja nie spełnia wymagań.
        """
        distances, neighbors = self.model.kneighbors([features], return_distance=True)
        min_distance = distances[0][0]

        if min_distance > self.distance_threshold:
            # Jeśli najbliższy sąsiad jest zbyt daleko, przyjmijmy, że nie rozpoznano gestu
            self.prediction_history.append("brak gestu")
            final_prediction = max(set(self.prediction_history), key=self.prediction_history.count)
            return final_prediction

        # Głosowanie KNN na podstawie sąsiadów
        neighbor_labels = [self.labels[i] for i in neighbors[0]]
        label_counts = {label: neighbor_labels.count(label) for label in set(neighbor_labels)}
        predicted_label, vote_count = max(label_counts.items(), key=lambda item: item[1])
        vote_percentage = vote_count / len(neighbors[0])

        # Jeśli wynik spełnia próg głosów, dodaj do historii, w przeciwnym razie brak gestu
        if vote_percentage >= self.vote_threshold:
            self.prediction_history.append(predicted_label)
        else:
            self.prediction_history.append("brak gestu")

        # Ostateczna predykcja na podstawie historii
        final_prediction = max(set(self.prediction_history), key=self.prediction_history.count)
        return final_prediction
