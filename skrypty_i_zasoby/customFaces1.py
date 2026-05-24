import numpy as np
import matplotlib
import os
import sys
import cv2
import csv
import re
from sklearn.decomposition import PCA

# !!! TECHNICZNA POPRAWKA DLA WINDOWS !!!
try:
    matplotlib.use('TkAgg')
except Exception as e:
    print(f"Uwaga: Nie udało się wymusić backendu TkAgg: {e}")
import matplotlib.pyplot as plt


class FaceDetectorPCA:
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=42)  # Dodano random_state dla powtarzalności
        self.mean_face = None
        self.threshold = None

    def train(self, face_images):
        print(f"Trening PCA na {len(face_images)} własnych zdjęciach...")
        self.mean_face = np.mean(face_images, axis=0)
        centered_faces = face_images - self.mean_face
        self.pca.fit(centered_faces)

        # Obliczenie progu detekcji (DFFS / MSE)
        reconstructed_faces = self.pca.inverse_transform(self.pca.transform(centered_faces))
        errors = np.mean((centered_faces - reconstructed_faces) ** 2, axis=1)
        self.threshold = np.max(errors) * 1.2
        print(f"Trening zakończony! Próg błędu MSE: {self.threshold:.4f}\n")

    def detect(self, new_image):
        # Centrowanie i rzutowanie
        centered_image = (new_image - self.mean_face).reshape(1, -1)
        projected = self.pca.transform(centered_image)
        reconstructed = self.pca.inverse_transform(projected)

        # Błąd (im mniejszy, tym większe podobieństwo do wzorca twarzy)
        mse_error = np.mean((centered_image - reconstructed) ** 2)
        is_face = mse_error <= self.threshold
        final_reconstruction = reconstructed.flatten() + self.mean_face
        return is_face, mse_error, final_reconstruction


# ==========================================
# FUNKCJE DO OBSŁUGI PLIKÓW I KATALOGÓW
# ==========================================

def wczytaj_i_skaluj_zdjecie(sciezka):
    """
    Wczytuje pojedyncze zdjęcie testowe, zamienia na skalę szarości,
    skaluje do 64x64 i normalizuje (0-1). Zwraca None, jeśli wystąpi błąd.
    """
    if not os.path.exists(sciezka):
        print(f"BŁĄD: Nie znaleziono pliku '{sciezka}'.")
        return None

    try:
        img = cv2.imread(sciezka, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img_resized = cv2.resize(img, (64, 64))
        return (img_resized.astype('float32') / 255.0).flatten()
    except Exception as e:
        print(f"Błąd podczas przetwarzania pliku {sciezka}: {e}")
        return None


def wczytaj_katalog_do_treningu(sciezka_katalogu):
    """
    Przeszukuje katalog ze zdjęciami, każde przerabia na skalę szarości
    i standaryzuje do 64x64. Zwraca macierz gotową do włożenia w PCA.
    """
    if not os.path.exists(sciezka_katalogu):
        print(f"BŁĄD KRYTYCZNY: Katalog '{sciezka_katalogu}' nie istnieje.")
        print("Utwórz folder o tej nazwie i wrzuć do niego zdjęcia treningowe.")
        sys.exit(1)

    dane = []
    print(f"Przeszukiwanie katalogu '{sciezka_katalogu}'...")

    for plik in os.listdir(sciezka_katalogu):
        if plik.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            sciezka_pliku = os.path.join(sciezka_katalogu, plik)
            try:
                img = cv2.imread(sciezka_pliku, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, (64, 64))
                    dane.append((img_resized.astype('float32') / 255.0).flatten())
            except Exception as e:
                print(f"Pominięto plik {plik} z powodu błędu: {e}")

    dane_np = np.array(dane)
    print(f"Załadowano {len(dane_np)} zdjęć treningowych.\n")

    if len(dane_np) < 5:
        print("BŁĄD: Masz za mało zdjęć!")
        sys.exit(1)

    return dane_np


# ==========================================
# NOWA FUNKCJA - MASOWA ANALIZA ZDJĘĆ
# ==========================================

def analizuj_moje_zdjecia_i_zapisz_csv(folder_testowy, detektor, nazwa_csv="raport_rozpoznawania.csv"):
    """
    Przechodzi przez podkatalogi w 'folder_testowy', analizuje zdjęcia
    (oryginał i po rozkładzie SVD), i wypluwa statystyki do pliku .csv.
    """
    print(f"\n--- ROZPOCZYNAM MASOWĄ ANALIZĘ FOLDERU: '{folder_testowy}' ---")
    if not os.path.exists(folder_testowy):
        print(f"BŁĄD: Folder '{folder_testowy}' nie istnieje! Zostanie pominięty.")
        return

    # Słownik do przechowywania statystyk. Klucz to np. "original", "10", "20"
    # Wartość to słownik: {'suma_bledow': float, 'wykryto': int, 'wszystkie': int}
    statystyki = {}

    for podfolder in os.listdir(folder_testowy):
        sciezka_podfolderu = os.path.join(folder_testowy, podfolder)

        if not os.path.isdir(sciezka_podfolderu):
            continue

        for plik in os.listdir(sciezka_podfolderu):
            if not plik.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            # Rozpoznawanie nazwy pliku i wyciąganie typu/poziomu SVD
            nazwa_lower = plik.lower()
            poziom = "nieznany"

            if nazwa_lower.startswith('original'):
                poziom = "original"
            else:
                # Szukamy cyfr na samym początku nazwy pliku (np. "10_zdjecie.jpg" -> "10")
                match = re.match(r'^(\d+)', nazwa_lower)
                if match:
                    poziom = match.group(1)

            # Inicjalizacja wpisu w słowniku
            if poziom not in statystyki:
                statystyki[poziom] = {'suma_bledow': 0.0, 'wykryto': 0, 'wszystkie': 0}

            sciezka_pliku = os.path.join(sciezka_podfolderu, plik)

            # Wczytanie i detekcja
            obraz_testowy = wczytaj_i_skaluj_zdjecie(sciezka_pliku)
            if obraz_testowy is not None:
                is_face, error, _ = detektor.detect(obraz_testowy)

                statystyki[poziom]['wszystkie'] += 1
                statystyki[poziom]['suma_bledow'] += error
                if is_face:
                    statystyki[poziom]['wykryto'] += 1

    # Zapis statystyk do CSV
    print(f"Zapisywanie wyników do pliku: {nazwa_csv}...")
    try:
        with open(nazwa_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Nagłówki tabeli
            writer.writerow(['Poziom SVD / Typ', 'Rozpoznane jako twarz', 'Laczna liczba zdjec', 'Skutecznosc (%)',
                             'Sredni blad MSE'])

            # Sortowanie kluczy (najpierw "original", potem liczbowo poziomy SVD)
            def klucz_sortowania(k):
                if k == "original": return 999
                if k == "nieznany": return 1000
                return int(k)

            posortowane_poziomy = sorted(statystyki.keys(), key=klucz_sortowania)

            for poziom in posortowane_poziomy:
                dane = statystyki[poziom]
                wszystkie = dane['wszystkie']
                wykryto = dane['wykryto']
                suma_bledow = dane['suma_bledow']

                if wszystkie > 0:
                    skutecznosc = (wykryto / wszystkie) * 100
                    sredni_blad = suma_bledow / wszystkie

                    writer.writerow([poziom, wykryto, wszystkie, f"{skutecznosc:.2f}%", f"{sredni_blad:.4f}"])

        print("Wygenerowano plik CSV.\n")
    except Exception as e:
        print(f"Błąd podczas zapisywania pliku CSV: {e}")


# ==========================================
# GŁÓWNA CZĘŚĆ PROGRAMU
# ==========================================
if __name__ == "__main__":
    # 1. Zdefiniowanie ścieżek
    folder_treningowy = 'dataset_treningowy_1/AI-Generated Images'
    folder_testowy = 'dataset_testowy'

    # Zmienne ze ścieżkami do konkretnych zdjęć testowych
    sciezka_do_zdjecia_1 = 'Animal - monkey.jpg'
    sciezka_do_zdjecia_2 = 'Animal - dog.png'
    sciezka_do_zdjecia_3 = 'Animal - cat.png'

    # 2. Wczytanie i przygotowanie danych do treningu
    twarze_treningowe = wczytaj_katalog_do_treningu(folder_treningowy)
    docelowe_komponenty = min(50, len(twarze_treningowe) - 1)

    # 3. Tworzymy i trenujemy model
    detector = FaceDetectorPCA(n_components=docelowe_komponenty)
    detector.train(twarze_treningowe)

    # 4. Przetworzenie folderu i generowanie CSV
    analizuj_moje_zdjecia_i_zapisz_csv(folder_testowy, detector,
                                       nazwa_csv="statystyki_rozpoznawania_twarzy_4630_ai.csv")

    # -----------------------------------------------------------
    # 5. WIZUALIZACJA DLA WSKAZANYCH ZDJĘĆ
    # -----------------------------------------------------------
    print("\n--- Przygotowuję weryfikację wizualną modeli ---")

    testy_do_wyswietlenia = []

    for sciezka in [sciezka_do_zdjecia_1, sciezka_do_zdjecia_2, sciezka_do_zdjecia_3]:
        if os.path.exists(sciezka):
            zdjecie = wczytaj_i_skaluj_zdjecie(sciezka)
            if zdjecie is not None:
                testy_do_wyswietlenia.append((sciezka, zdjecie))
        else:
            print(f"Pominięto wizualizację: Brak pliku '{sciezka}'.")

    if testy_do_wyswietlenia:
        # Tworzymy wykres (liczba wierszy zależy od tego, ile plików znaleziono)
        liczba_wierszy = len(testy_do_wyswietlenia)
        fig, axes = plt.subplots(liczba_wierszy, 2, figsize=(8, 4 * liczba_wierszy))

        # Zabezpieczenie nakładających się napisów
        plt.subplots_adjust(hspace=0.5, wspace=0.3, top=0.85 if liczba_wierszy == 1 else 0.9)

        # Standaryzacja 'axes' do tablicy 2D, jeśli jest tylko jedno zdjęcie
        if liczba_wierszy == 1:
            axes = np.array([axes])

        for i, (nazwa_pliku, obraz_testowy) in enumerate(testy_do_wyswietlenia):
            is_face, error, recon_face = detector.detect(obraz_testowy)

            werdykt = "WYKRYTO TWARZ" if is_face else "ODRZUCONO (To nie twarz)"
            kolor_tytulu = "green" if is_face else "red"

            # Lewa kolumna: Oryginał
            axes[i, 0].imshow(obraz_testowy.reshape(64, 64), cmap='gray')
            axes[i, 0].set_title(f"Wejście:\n{os.path.basename(nazwa_pliku)}", fontsize=10)
            axes[i, 0].axis('off')

            # Prawa kolumna: Rekonstrukcja
            axes[i, 1].imshow(recon_face.reshape(64, 64), cmap='gray')
            axes[i, 1].set_title(f"PCA Werdykt: {werdykt}\nBłąd MSE: {error:.4f} (Próg: {detector.threshold:.4f})",
                                 fontsize=10, color=kolor_tytulu)
            axes[i, 1].axis('off')

        plt.suptitle("System Detekcji Twarzy: Sprawdzanie wskazanych zdjęć", fontsize=14, fontweight='bold')
        print("Uruchamiam okno wykresu...")
        plt.show()
    else:
        print("Brak obrazów do wyświetlenia (nie znaleziono plików na dysku).")