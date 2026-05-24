import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from sklearn.datasets import fetch_olivetti_faces

# Wymuszenie backendu (aby okno poprawnie reagowało na kliknięcia w Windows)
try:
    matplotlib.use('TkAgg')
except Exception as e:
    print(f"Uwaga: {e}")

# 1. Pobieranie bazy danych
print("Pobieranie bazy Olivetti Faces...")
dataset = fetch_olivetti_faces(shuffle=False, random_state=42)
twarze = dataset.images
calkowita_liczba_zdjec = len(twarze)

# 2. Zmienne do kontrolowania "stron"
zdjec_na_strone = 10
aktualny_indeks = 0

# 3. Przygotowanie okna i siatki (2 wiersze po 5 zdjęć = 10 zdjęć)
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
plt.subplots_adjust(bottom=0.2, wspace=0.3, hspace=0.3)


# 4. Funkcja rysująca aktualną paczkę 10 zdjęć
def rysuj_galerie():
    for i, ax in enumerate(axes.flat):
        ax.clear()
        ax.axis('off')

        indeks_zdjecia = aktualny_indeks + i

        if indeks_zdjecia < calkowita_liczba_zdjec:
            ax.imshow(twarze[indeks_zdjecia], cmap='gray')

            numer_osoby = (indeks_zdjecia // 10) + 1
            numer_zdjecia = (indeks_zdjecia % 10) + 1
            ax.set_title(f"Osoba {numer_osoby}\n(Zdj {numer_zdjecia}/10)", fontsize=9)

    strona = (aktualny_indeks // zdjec_na_strone) + 1
    max_stron = calkowita_liczba_zdjec // zdjec_na_strone
    fig.suptitle(f"Baza Olivetti: Strona {strona} z {max_stron}", fontsize=14, fontweight='bold')

    fig.canvas.draw_idle()


# 5. Funkcje obsługujące kliknięcia przycisków
def nastepna_strona(event):
    global aktualny_indeks
    if aktualny_indeks + zdjec_na_strone < calkowita_liczba_zdjec:
        aktualny_indeks += zdjec_na_strone
        rysuj_galerie()


def poprzednia_strona(event):
    global aktualny_indeks
    if aktualny_indeks - zdjec_na_strone >= 0:
        aktualny_indeks -= zdjec_na_strone
        rysuj_galerie()


# 6. Tworzenie przycisków na dole ekranu
# Wymiary: [pozycja_x, pozycja_y, szerokosc, wysokosc] w skali 0-1
ax_poprzedni = plt.axes([0.3, 0.05, 0.15, 0.075])
ax_nastepny = plt.axes([0.55, 0.05, 0.15, 0.075])

przycisk_poprzedni = Button(ax_poprzedni, '<- Poprzednia')
przycisk_nastepny = Button(ax_nastepny, 'Następna ->')

# Przypisanie akcji do kliknięcia
przycisk_poprzedni.on_clicked(poprzednia_strona)
przycisk_nastepny.on_clicked(nastepna_strona)

# 7. Pierwsze narysowanie i wyświetlenie okna
rysuj_galerie()
plt.show()