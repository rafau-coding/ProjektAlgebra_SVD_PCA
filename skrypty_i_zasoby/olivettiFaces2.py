import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_olivetti_faces
import os
import sys
import cv2

class FaceDetectorPCA:
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.mean_face = None
        self.threshold = None

    def train(self, face_images):
        print("Trening PCA na 400 zdjęciach...")
        self.mean_face = np.mean(face_images, axis=0)
        centered_faces = face_images - self.mean_face
        self.pca.fit(centered_faces)

        reconstructed_faces = self.pca.inverse_transform(self.pca.transform(centered_faces))
        errors = np.mean((centered_faces - reconstructed_faces) ** 2, axis=1)

        self.threshold = np.max(errors) * 1.2
        print(f"Trening zakończony. Próg błędu DFFS: {self.threshold:.4f}\n")

    def detect(self, new_image):
        centered_image = (new_image - self.mean_face).reshape(1, -1)
        projected = self.pca.transform(centered_image)
        reconstructed = self.pca.inverse_transform(projected)

        mse_error = np.mean((centered_image - reconstructed) ** 2)
        is_face = mse_error <= self.threshold

        final_reconstruction = reconstructed.flatten() + self.mean_face
        return is_face, mse_error, final_reconstruction


def przygotuj_twarz_z_okularami(twarz_1d):
    """Rysuje czarny pasek na oczach na podanym zdjęciu"""
    obraz_2d = twarz_1d.copy().reshape(64, 64)
    obraz_2d[20:30, 15:50] = 0.0  # Czarny pasek
    return obraz_2d.flatten()


def przygotuj_twarz_do_gory_nogami(twarz_1d):
    """Obraca zdjęcie do góry nogami"""
    obraz_2d = twarz_1d.reshape(64, 64)
    return np.flipud(obraz_2d).flatten()

def wczytaj_i_skaluj_zdjecie(sciezka):
    """
    Wczytuje zdjęcie, zamienia na skalę szarości,
    skaluje do 64x64 i normalizuje (0-1).
    """
    if not os.path.exists(sciezka):
        print(f"BŁĄD KRYTYCZNY: Nie znaleziono pliku '{sciezka}' w tym folderze.")
        print("Pobierz zdjęcia z linków podanych w odpowiedzi i zapisz je tutaj.")
        sys.exit(1)

    try:
        # Wczytujemy w skali szarości
        img = cv2.imread(sciezka, cv2.IMREAD_GRAYSCALE)
        # Skalujemy do rozmiaru Olivetti (64x64 piksele)
        img_resized = cv2.resize(img, (64, 64))
        # Normalizujemy (0-1) i spłaszczamy do 1D
        return (img_resized.astype('float32') / 255.0).flatten()
    except Exception as e:
        print(f"Błąd podczas przetwarzania pliku {sciezka}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    dataset = fetch_olivetti_faces(shuffle=True, random_state=42)
    prawdziwe_twarze = dataset.data

    twarze_treningowe = prawdziwe_twarze

    detector = FaceDetectorPCA(n_components=50)
    detector.train(twarze_treningowe)

    sciezka_do_zdjecia = 'Human.jpg'

    # Przygotowanie 4 eksperymentów
    testy = [
        ("1. Twarz w okularach (Czarny pasek)", przygotuj_twarz_z_okularami(twarze_treningowe[300])),
        ("2. Twarz do góry nogami (Zmiana pozycji)", przygotuj_twarz_do_gory_nogami(twarze_treningowe[301])),
        ("3. Nowa twarz", wczytaj_i_skaluj_zdjecie(sciezka_do_zdjecia))
    ]

    # Wizualizacja wszystkich testów na jednej planszy
    fig, axes = plt.subplots(len(testy), 2, figsize=(8, 12))
    plt.subplots_adjust(hspace=0.4)

    for i, (nazwa_testu, obraz_testowy) in enumerate(testy):
        is_face, error, recon_face = detector.detect(obraz_testowy) # zdjecie * V * VT

        werdykt = "TWARZ WYKRYTA" if is_face else "ODRZUCONO (Błąd krytyczny)"
        kolor_tytulu = "green" if is_face else "red"

        # Oryginał (Lewa kolumna)
        axes[i, 0].imshow(obraz_testowy.reshape(64, 64), cmap='gray')
        axes[i, 0].set_title(nazwa_testu, fontsize=10)
        axes[i, 0].axis('off')

        # Rekonstrukcja (Prawa kolumna)
        axes[i, 1].imshow(recon_face.reshape(64, 64), cmap='gray')
        axes[i, 1].set_title(f"{werdykt}\nBłąd: {error:.4f} (Próg: {detector.threshold:.4f})",
                             fontsize=10, color=kolor_tytulu)
        axes[i, 1].axis('off')

    plt.suptitle("Analiza Błędu Rekonstrukcji (Słabości PCA)", fontsize=14, fontweight='bold')
    plt.show()