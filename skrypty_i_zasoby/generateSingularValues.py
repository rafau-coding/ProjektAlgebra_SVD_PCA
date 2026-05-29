import cv2
import numpy as np
import os
import csv

sciezka_do_zdjecia = "pattern - gorszy.jpg"
folder_wynikowy = "pattern - gorszy - maski svd"

def generuj_kolorowe_obrazy_osobliwe(sciezka_wejsciowa, liczba_komponentow=10):
    # 1. Sprawdzenie czy plik istnieje
    if not os.path.exists(sciezka_wejsciowa):
        print(f"BŁĄD: Plik nie istnieje pod ścieżką: {sciezka_wejsciowa}")
        return

    # 2. Wczytanie kolorowego obrazu (format BGR)
    img = cv2.imread(sciezka_wejsciowa)

    if img is None:
        print(f"BŁĄD: Nie można wczytać obrazu. Sprawdź format pliku.")
        return

    print(f"Pomyślnie wczytano obraz kolorowy o wymiarach: {img.shape}")

    # Konwersja na float64 dla precyzji obliczeń matematycznych
    A = img.astype(np.float64)

    # 3. Przygotowanie list na wyniki SVD dla poszczególnych kanałów
    U_channels = []
    S_channels = []
    Vt_channels = []

    print("Obliczanie SVD dla kanałów B, G i R...")

    # Przechodzimy przez 3 kanały (0 = Blue, 1 = Green, 2 = Red)
    for c in range(3):
        U, S, Vt = np.linalg.svd(A[:, :, c], full_matrices=False)
        U_channels.append(U)
        S_channels.append(S)
        Vt_channels.append(Vt)

    print("Obliczenia SVD zakończone.")

    # 4. Przygotowanie folderu na wyniki
    if not os.path.exists(folder_wynikowy):
        os.makedirs(folder_wynikowy)
        print(f"Utworzono folder: {folder_wynikowy}")

    nazwa_bazowa = os.path.splitext(os.path.basename(sciezka_wejsciowa))[0]
    max_k = min(liczba_komponentow, len(S_channels[0]))

    # Ścieżka do pliku CSV
    sciezka_csv = os.path.join(folder_wynikowy, "wartosci_osobliwe.csv")

    # 5. Generowanie obrazów i zapis do CSV
    print(f"Generowanie {max_k} kolorowych obrazów składowych oraz pliku CSV...")

    # Otwieramy plik CSV do zapisu
    with open(sciezka_csv, mode='w', newline='', encoding='utf-8') as plik_csv:
        writer = csv.writer(plik_csv)
        # Nagłówki kolumn w pliku CSV
        writer.writerow(
            ['Numer_skladowej', 'Wartosc_Niebieski', 'Wartosc_Zielony', 'Wartosc_Czerwony', 'Srednia_Waznosc'])

        for i in range(max_k):
            warstwy_BGR = []

            # Pobieramy wartości osobliwe dla danego kroku
            sigma_B = S_channels[0][i]
            sigma_G = S_channels[1][i]
            sigma_R = S_channels[2][i]
            srednia_sigma = (sigma_B + sigma_G + sigma_R) / 3

            # Zapisujemy wiersz z danymi do CSV (zaokrąglone do 2 miejsc po przecinku dla czytelności)
            writer.writerow([
                i + 1,
                round(sigma_B, 2),
                round(sigma_G, 2),
                round(sigma_R, 2),
                round(srednia_sigma, 2)
            ])

            # Rekonstruujemy i-tą warstwę osobno dla niebieskiego, zielonego i czerwonego
            for c in range(3):
                sigma_i = S_channels[c][i]
                u_i = U_channels[c][:, i:i + 1]
                v_i_t = Vt_channels[c][i:i + 1, :]

                # Wzór: u_i * sigma_i * v_i_t
                warstwa = np.dot(u_i, v_i_t) * sigma_i
                warstwy_BGR.append(warstwa)

            # 6. Złożenie kanałów z powrotem w jeden kolorowy obraz (3D)
            obraz_3d = np.dstack(warstwy_BGR)

            # 7. Normalizacja obrazu
            zminimalizowana = obraz_3d - np.min(obraz_3d)
            zakres = np.max(zminimalizowana)

            if zakres > 0:
                obraz_wizualny = (zminimalizowana / zakres) * 255
            else:
                obraz_wizualny = zminimalizowana

            obraz_finalny = obraz_wizualny.astype(np.uint8)

            # 8. Zapisanie pliku ze zdjęciem
            numer = f"{i + 1:02d}"
            pelna_sciezka_zapisu = os.path.join(folder_wynikowy, f"{numer}.png")

            cv2.imwrite(pelna_sciezka_zapisu, obraz_finalny)

            print(f"[-] Zapisano: {pelna_sciezka_zapisu} (Średnia ważność: {srednia_sigma:.2f})")


# --- Uruchomienie ---
if __name__ == "__main__":
    generuj_kolorowe_obrazy_osobliwe(sciezka_do_zdjecia)