import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_olivetti_faces

# Wymuszenie backendu (przydatne na Windowsie, by okna się nie zawieszały)
try:
    matplotlib.use('TkAgg')
except Exception as e:
    print(f"Uwaga: {e}")


class FaceDetectorPCA:
    def __init__(self, n_components=50):
        # Zachowujemy 50 najważniejszych 'Twarzy Własnych' (Eigenfaces)
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=42)
        self.mean_face = None
        self.threshold = None

    def train(self, face_images):
        print("Trening PCA na 300 zdjęciach...")
        # 1. Wyliczenie Twarzy Średniej
        self.mean_face = np.mean(face_images, axis=0)

        # 2. Centrowanie (odjęcie średniej)
        centered_faces = face_images - self.mean_face

        # 3. Trening PCA (Budowa macierzy V)
        self.pca.fit(centered_faces)

        # 4. Rzutowanie i Rekonstrukcja danych treningowych
        reconstructed_faces = self.pca.inverse_transform(self.pca.transform(centered_faces))

        # 5. Ustalenie progu błędu (największy błąd na danych treningowych + 20% marginesu)
        errors = np.mean((centered_faces - reconstructed_faces) ** 2, axis=1)
        self.threshold = np.max(errors) * 1.2
        print(f"Trening zakończony! Próg błędu DFFS: {self.threshold:.4f}\n")

    def detect(self, new_image):
        # Centrowanie nowego obrazu i zmiana kształtu dla sklearn
        centered_image = (new_image - self.mean_face).reshape(1, -1)

        # Kompresja (Z) i Rekonstrukcja
        projected = self.pca.transform(centered_image)
        reconstructed = self.pca.inverse_transform(projected)

        # Obliczenie błędu
        mse_error = np.mean((centered_image - reconstructed) ** 2)
        is_face = mse_error <= self.threshold

        # Zwracamy wynik, błąd i obraz zrekonstruowany (do wizualizacji)
        # Dodajemy z powrotem twarz średnią, żeby obraz wyglądał naturalnie
        final_reconstruction = reconstructed.flatten() + self.mean_face
        return is_face, mse_error, final_reconstruction

    def visualize(self, original, reconstructed, title_text):
        """Pomocnicza funkcja do rysowania wyników na wykresie"""
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        # Rysujemy oryginał (wymaga zwinięcia wektora 4096 z powrotem w kwadrat 64x64)
        axes[0].imshow(original.reshape(64, 64), cmap='gray')
        axes[0].set_title("Obraz wejściowy")
        axes[0].axis('off')

        # Rysujemy rekonstrukcję algorytmu
        axes[1].imshow(reconstructed.reshape(64, 64), cmap='gray')
        axes[1].set_title("Rekonstrukcja PCA (Eigenfaces)")
        axes[1].axis('off')

        plt.suptitle(title_text, color="red", fontweight="bold")

        plt.subplots_adjust(top=0.75)

        # plt.show() blokuje działanie skryptu do momentu zamknięcia okna
        plt.show()


# ==========================================
# GŁÓWNA CZĘŚĆ PROGRAMU
# ==========================================
if __name__ == "__main__":
    print("Pobieranie bazy zdjęć Olivetti Faces (400 zdjęć, 64x64 piksele)...")
    dataset = fetch_olivetti_faces(shuffle=True, random_state=42)
    prawdziwe_twarze = dataset.data

    # Dzielimy dane: 300 zdjęć uczy model, 100 zostawiamy do sprawdzania
    twarze_treningowe = prawdziwe_twarze[:300]
    twarze_testowe = prawdziwe_twarze[300:]

    # Tworzymy i trenujemy detektor
    detector = FaceDetectorPCA(n_components=50)
    detector.train(twarze_treningowe)

    # ---------------------------------------------------------
    # TEST MASOWY: Sprawdzamy wszystkie 100 odłożonych zdjęć
    # ---------------------------------------------------------
    liczba_testow = len(twarze_testowe)
    liczba_bledow = 0

    print(f"Rozpoczynanie sprawdzania {liczba_testow} zdjęć testowych...")
    print("------------------------------------------------------------------")

    # Przechodzimy w pętli przez wszystkie 100 zdjęć
    for i, test_face in enumerate(twarze_testowe):
        is_face, error, recon_face = detector.detect(test_face)

        # Jeśli model uznał, że to NIE jest twarz (czyli popełnił błąd, bo wiemy, że to twarze)
        if not is_face:
            liczba_bledow += 1

            # Prawdziwy indeks w oryginalnej bazie to 300 + i
            rzeczywisty_indeks = 300 + i
            tytul = f"BŁĄD DETEKCJI (Indeks: {rzeczywisty_indeks})\nBłąd: {error:.4f} > Próg: {detector.threshold:.4f}"

            print(f"Zanotowano błąd dla zdjęcia nr {rzeczywisty_indeks}. Wyświetlanie zdjęcia i rekonstrukcji...")

            # Wyświetlamy okno. Skrypt zatrzyma się w tym miejscu, dopóki go nie zamkniesz!
            detector.visualize(test_face, recon_face, tytul)

    # ---------------------------------------------------------
    # PODSUMOWANIE W KONSOLI
    # ---------------------------------------------------------
    print("\n================ PODSUMOWANIE ================")
    print(f"Liczba przetestowanych zdjęć: {liczba_testow}")
    print(f"Poprawnie wykryte twarze:     {liczba_testow - liczba_bledow}")
    print(f"Błędy (odrzucone twarze):     {liczba_bledow}")

    if liczba_bledow == 0:
        print("Wynik: 100% skuteczności na danych testowych!")
    else:
        skutecznosc = ((liczba_testow - liczba_bledow) / liczba_testow) * 100
        print(f"Skuteczność detekcji:         {skutecznosc:.1f}%")
    print("==============================================")