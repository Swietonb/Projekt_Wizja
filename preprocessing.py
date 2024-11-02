import numpy as np
from typing import List


class Preprocessor:
    def normalize_hand(self, landmarks) -> np.ndarray:
        """
        Normalizuje współrzędne punktów charakterystycznych dłoni do jednorodnego układu współrzędnych [-1,1].

        Proces obejmuje przesunięcie punktów względem pierwszego punktu (bazy dłoni) oraz skalowanie do jednostki
        w celu ujednolicenia rozmiaru dłoni. Następnie współrzędne są spłaszczane do jednowymiarowej tablicy.

        Args:
            landmarks (List[List[float]]): Lista punktów charakterystycznych dłoni (współrzędne x, y, z).

        Returns:
            np.ndarray: Znormalizowane i spłaszczone współrzędne punktów charakterystycznych dłoni.
        """
        # Przesunięcie dłoni do początku układu współrzędnych
        base_x, base_y, base_z = landmarks[0][:3]
        for point in landmarks:
            point[0] -= base_x
            point[1] -= base_y
            point[2] -= base_z

        # Normalizacja długości (skalowanie do jednostki)
        max_distance = max(np.linalg.norm(np.array(point) - np.array(landmarks[0])) for point in landmarks[1:])
        normalized_landmarks = [[coord / max_distance for coord in point] for point in landmarks]

        #----------------------PĘTLA TESTOWA----------------------------
        #---Do sprawdzenia poprawności punktów po normalizacji----------
        #---Oblicza odległość punktu od środka układu współrzędnych-----
        #---min distance = 0 - Początek układu współrzędnych------------
        #---max distance = 1 - Punkt oddalony najbardziej ma wartość 1--
        #---------------------------------------------------------------

        # for i in normalized_landmarks:
        #     point = i
        #     # Obliczenie odległości euklidesowej od początku układu współrzędnych
        #     distance = np.sqrt(point[0] ** 2 + point[1] ** 2 + point[2] ** 2)
        #     print("Odległość od początku układu współrzędnych:", distance)

        #----------------------------------------------------------------

        # Spłaszczenie do jednowymiarowej tablicy
        flattened_landmarks = np.array(normalized_landmarks).flatten()
        return flattened_landmarks
