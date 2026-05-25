import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def robust_smart_crop(image_path, target_size=(1024, 1024)):
    """
    Kuloodporna funkcja do wczytywania i kadrowania zdjęć.
    Używa Pillow dla bezpieczeństwa ścieżek i luźnych filtrów OpenCV do twarzy.
    """
    try:
        pil_img = Image.open(image_path).convert('L')
        gray_matrix = np.array(pil_img)
    except Exception as e:
        raise ValueError(f"Pillow nie potrafi otworzyć pliku: {e}")

    H, W = gray_matrix.shape

    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(gray_matrix, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30))

    if len(faces) > 0:
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]
        cx = x + w // 2
        cy = y + h // 2
    else:
        cx = W // 2
        cy = H // 2

    crop_size = min(W, H)

    x1 = max(0, min(cx - crop_size // 2, W - crop_size))
    y1 = max(0, min(cy - crop_size // 2, H - crop_size))
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    cropped_gray = gray_matrix[y1:y2, x1:x2]

    final_pil = Image.fromarray(cropped_gray)
    final_resized = final_pil.resize(target_size, Image.Resampling.LANCZOS)

    return np.array(final_resized)


def analyze_and_save_svd(folder_path, output_base_folder="wyniki_svd", target_size=(1024, 1024),
                         k_values=[3, 7, 10, 15, 20, 25, 30, 50]):
    """
    Skanuje folder, robi smart-crop, wyświetla wyniki na ekranie
    oraz zapisuje wygenerowane obrazy do zorganizowanych folderów na dysku.
    """
    if not os.path.exists(folder_path):
        print(f"Błąd: Folder '{folder_path}' nie istnieje!")
        return

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print("Nie znaleziono obsługiwanych zdjęć w folderze.")
        return

    os.makedirs(output_base_folder, exist_ok=True)
    print(f"Rozpoczynam analizę {len(image_files)} zdjęć. Wyniki trafią do: ./{output_base_folder}/\n")

    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        print(f"Przetwarzam: {filename}...")

        base_name = os.path.splitext(filename)[0]
        img_output_folder = os.path.join(output_base_folder, base_name)
        os.makedirs(img_output_folder, exist_ok=True)

        try:
            img_matrix = robust_smart_crop(image_path, target_size=target_size)
        except Exception as e:
            print(f"  [X] Pomijam plik. Błąd: {e}")
            continue

        # Zapisujemy ORYGINAŁ (wykadrowany)
        original_output_path = os.path.join(img_output_folder, f"original_{filename}")
        Image.fromarray(img_matrix).save(original_output_path)

        # ROZKŁAD SVD na ogromnej macierzy 1024x1024
        U, S, Vt = np.linalg.svd(img_matrix, full_matrices=False)

        # Przygotowanie wykresu
        fig, axes = plt.subplots(3, 3, figsize=(16, 16))
        fig.suptitle(f"SVD (1024x1024) | Plik: {filename}", fontsize=18, fontweight='bold')
        axes = axes.flatten()

        axes[0].imshow(img_matrix, cmap='gray')
        axes[0].set_title("Oryginał\n(1 048 576 pikseli)", fontsize=12)
        axes[0].axis('off')

        for i, k in enumerate(k_values):
            reconstructed = np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))
            reconstructed = np.clip(reconstructed, 0, 255)

            reconstructed_uint8 = reconstructed.astype(np.uint8)
            k_output_path = os.path.join(img_output_folder, f"{k}_{filename}")
            Image.fromarray(reconstructed_uint8).save(k_output_path)

            ax = axes[i + 1]
            ax.imshow(reconstructed_uint8, cmap='gray')
            ax.set_title(f"k = {k}", fontsize=12)
            ax.axis('off')

        plt.tight_layout()
        print(f"  -> Zapisano oryginalny obraz i {len(k_values)} kompresji w: {img_output_folder}/")

    print("\nGotowe! Obrazy zostały zapisane na dysku. Zamknij obecne okno z wykresem, aby wyświetlić kolejne.")
    #plt.show()


# =====================================================================
# URUCHOMIENIE
# =====================================================================
folder_ze_zdjeciami = "moje_zdjecia"  # <-- Twój folder ze zdjęciami wejściowymi
analyze_and_save_svd(folder_ze_zdjeciami)
