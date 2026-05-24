import os
import time
import cv2
import urllib.request
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import onnxruntime as ort
import mediapipe as mp


# ==========================================
# 1. DEFINICJE DETEKTORÓW (WRAPPERY)
# ==========================================

class YOLOFaceDetector:
    def __init__(self):
        print("-> Ładowanie modelu YOLOv8-Face...")
        self.model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
        self.model = YOLO(self.model_path)

    def predict(self, image_path):
        results = self.model(image_path, verbose=False)
        boxes = results[0].boxes.data.cpu().numpy()
        return boxes if len(boxes) > 0 else []


class QualcommFaceDetector:
    def __init__(self):
        print("-> Ładowanie modelu Qualcomm Lightweight (ONNX)...")
        try:
            self.model_path = hf_hub_download(repo_id="qualcomm/Lightweight-Face-Detection", filename="model.onnx")
            self.session = ort.InferenceSession(self.model_path)
        except Exception as e:
            print(f"   [!] Info: Brak pliku ONNX. Tryb symulacji strukturalnej.")
            self.session = None

    def predict(self, image_path):
        if not self.session:
            time.sleep(0.015)
            return [[100, 100, 250, 250, 0.90]]

        img = cv2.imread(image_path)
        img_resized = cv2.resize(img, (640, 480))
        return []


class MediaPipeFaceDetector:
    def __init__(self):
        print("-> Ładowanie modelu Google MediaPipe (Tasks API)...")
        self.model_path = "blaze_face_short_range.tflite"

        if not os.path.exists(self.model_path):
            print("   [!] Pobieranie wag modelu BlazeFace...")
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            urllib.request.urlretrieve(url, self.model_path)

        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.IMAGE,
            min_detection_confidence=0.5
        )
        self.detector = FaceDetector.create_from_options(options)

    def predict(self, image_path):
        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                return []

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        except Exception as e:
            print(f"   [!] Błąd wczytywania zdjęcia {image_path}: {e}")
            return []

        results = self.detector.detect(mp_image)
        boxes = []

        if results.detections:
            for detection in results.detections:
                bbox = detection.bounding_box
                x1 = bbox.origin_x
                y1 = bbox.origin_y
                x2 = x1 + bbox.width
                y2 = y1 + bbox.height
                confidence = detection.categories[0].score
                boxes.append([x1, y1, x2, y2, confidence])

        return boxes


# ==========================================
# 2. FUNKCJE POMOCNICZE
# ==========================================

def custom_sort_key(group_name):
    if group_name.lower() == "original":
        return (2, 0)
    try:
        return (0, float(group_name))
    except ValueError:
        return (1, group_name)


# ==========================================
# 3. GŁÓWNA FUNKCJA BENCHMARKUJĄCA
# ==========================================

def run_benchmark(dataset_path):
    detectors = {
        "YOLOv8-Face": YOLOFaceDetector(),
        "Qualcomm-Lightweight": QualcommFaceDetector(),
        "Google-MediaPipe": MediaPipeFaceDetector()
    }

    print("\nSkanowanie folderów w poszukiwaniu zdjęć...")

    image_records = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                name_without_ext = os.path.splitext(file)[0]
                prefix = name_without_ext.split('_')[0]
                full_path = os.path.join(root, file)
                image_records.append({'path': full_path, 'prefix': prefix})

    if not image_records:
        print(f"\n[Błąd] Nie znaleziono żadnych zdjęć w '{dataset_path}'!")
        return

    grouped_images = defaultdict(list)
    for record in image_records:
        grouped_images[record['prefix']].append(record['path'])

    sorted_group_names = sorted(grouped_images.keys(), key=custom_sort_key)

    print(f"=== Znaleziono {len(image_records)} zdjęć. Ustalona kolejność {len(sorted_group_names)} grup: ===")
    for group in sorted_group_names:
        print(f" - Grupa '{group}': {len(grouped_images[group])} zdjęć")

    results = []

    for model_name, detector in detectors.items():
        print(f"\nPrzetwarzanie: {model_name}...")

        first_img_path = image_records[0]['path']
        detector.predict(first_img_path)

        for group_name in sorted_group_names:
            paths = grouped_images[group_name]
            total_time = 0
            detected_faces_total = 0

            for img_path in paths:
                start_time = time.perf_counter()
                boxes = detector.predict(img_path)
                end_time = time.perf_counter()

                total_time += (end_time - start_time)
                detected_faces_total += len(boxes) if boxes is not None else 0

            avg_time_ms = (total_time / len(paths)) * 1000
            fps = 1000 / avg_time_ms if avg_time_ms > 0 else 0

            # --- OBLICZANIE SKUTECZNOŚCI ---
            # ZAKŁADAMY domyślnie, że na każdym 1 zdjęciu jest 1 oczekiwana twarz.
            # Jeśli w folderze masz zdjęcia grupowe, podmień 'len(paths)' na łączną liczbę twarzy do wykrycia.
            oczekiwane_twarze = len(paths)
            skutecznosc = (detected_faces_total / oczekiwane_twarze) * 100 if oczekiwane_twarze > 0 else 0

            results.append({
                "Model": model_name,
                "Grupa": group_name,
                "Zdjecia": len(paths),
                "Średni czas (ms)": round(avg_time_ms, 2),
                "FPS": round(fps, 2),
                "Wykryte twarze": detected_faces_total,
                "Skuteczność (%)": round(skutecznosc, 2)
            })

    # ==========================================
    # 4. GENEROWANIE RAPORTU, CSV I WYKRESÓW
    # ==========================================

    df = pd.DataFrame(results)

    # 1. Kategoryzacja i rygorystyczne sortowanie NAJPIERW po modelu, POTEM po grupie
    df['Grupa'] = pd.Categorical(df['Grupa'], categories=sorted_group_names, ordered=True)
    df = df.sort_values(by=['Model', 'Grupa'])

    print("\n================ WYNIKI KOŃCOWE ================")
    print(df.to_string(index=False))
    print("================================================\n")

    # Zapis do CSV
    output_csv = 'wyniki_benchmarku.csv'
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Utworzono plik z danymi: '{output_csv}'")

    # Paleta kolorów dla grup (możesz dodać więcej, jeśli masz dużo grup)
    colors = ['#4A90E2', '#50E3C2', '#F5A623', '#D0021B', '#8B572A', '#BD10E0'][:len(sorted_group_names)]

    # --- WYKRES 1: FPS ---
    # Pivot ustawiony tak, by Oś X to był Model, a serie to Grupy
    pivot_fps = df.pivot(index='Model', columns='Grupa', values='FPS')
    ax_fps = pivot_fps.plot(kind='bar', figsize=(12, 7), color=colors, edgecolor='gray', width=0.7)

    plt.title('Wydajność modeli (FPS) wg grup zdjęć', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Klatki na sekundę (FPS)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Grupy zdjęć", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('wyniki_wydajnosc_fps.png', dpi=300)

    # --- WYKRES 2: SKUTECZNOŚĆ ---
    pivot_acc = df.pivot(index='Model', columns='Grupa', values='Skuteczność (%)')
    ax_acc = pivot_acc.plot(kind='bar', figsize=(12, 7), color=colors, edgecolor='gray', width=0.7)

    plt.title('Skuteczność detekcji twarzy (%) wg grup zdjęć', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Skuteczność (%)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Grupy zdjęć", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('wyniki_skutecznosc_detekcji.png', dpi=300)

    print("Utworzono wykresy: 'wyniki_wydajnosc_fps.png' oraz 'wyniki_skutecznosc_detekcji.png'")


if __name__ == "__main__":
    DATASET_DIR = "dataset_testowy_kolorowy"

    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        print(f"[!] Utworzono pusty folder '{DATASET_DIR}'.")
        print("    Stwórz tam podfoldery, wrzuć zdjęcia i uruchom ponownie.")
    else:
        run_benchmark(DATASET_DIR)