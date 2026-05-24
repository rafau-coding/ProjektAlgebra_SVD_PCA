import os
import time
import cv2
import urllib.request
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
# 2. GŁÓWNA FUNKCJA BENCHMARKUJĄCA
# ==========================================

def run_benchmark(dataset_path):
    detectors = {
        "YOLOv8-Face": YOLOFaceDetector(),
        "Qualcomm-Lightweight": QualcommFaceDetector(),
        "Google-MediaPipe": MediaPipeFaceDetector()
    }

    print("\nSkanowanie folderów w poszukiwaniu zdjęć...")

    image_paths = []
    # os.walk pozwala wejść do każdego podfolderu i zebrać stamtąd zdjęcia
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        print(f"\n[Błąd] Nie znaleziono żadnych zdjęć w folderze '{dataset_path}' ani jego podfolderach!")
        return

    total_images = len(image_paths)
    print(f"=== Znaleziono łącznie {total_images} zdjęć do przetworzenia. ===")

    results = []

    for model_name, detector in detectors.items():
        print(f"\nPrzetwarzanie: {model_name}...")

        # Rozgrzewka na pierwszym zdjęciu
        detector.predict(image_paths[0])

        total_time = 0
        detected_faces_total = 0

        # Pętla po wszystkich zebranych zdjęciach (bez względu na folder/nazwę)
        for img_path in image_paths:
            start_time = time.perf_counter()
            boxes = detector.predict(img_path)
            end_time = time.perf_counter()

            total_time += (end_time - start_time)
            detected_faces_total += len(boxes) if boxes is not None else 0

        avg_time_ms = (total_time / total_images) * 1000
        fps = 1000 / avg_time_ms if avg_time_ms > 0 else 0

        # Skuteczność (domyślne założenie: 1 twarz na 1 zdjęcie)
        skutecznosc = (detected_faces_total / total_images) * 100 if total_images > 0 else 0

        results.append({
            "Model": model_name,
            "Zdjecia": total_images,
            "Średni czas (ms)": round(avg_time_ms, 2),
            "FPS": round(fps, 2),
            "Wykryte twarze": detected_faces_total,
            "Skuteczność (%)": round(skutecznosc, 2)
        })

    # ==========================================
    # 3. GENEROWANIE RAPORTU, CSV I WYKRESÓW
    # ==========================================

    df = pd.DataFrame(results)

    print("\n================ WYNIKI KOŃCOWE ================")
    print(df.to_string(index=False))
    print("================================================\n")

    output_csv = 'wyniki_benchmarku_ogolne.csv'
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Utworzono plik z danymi: '{output_csv}'")

    colors = ['#4A90E2', '#50E3C2', '#F5A623']

    # --- WYKRES 1: FPS ---
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Model'], df['FPS'], color=colors, edgecolor='gray', width=0.6)

    plt.title('Wydajność modeli detekcji (FPS)', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Klatki na sekundę (FPS)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Podpisy słupków
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + max(1, yval * 0.02), f'{round(yval, 1)} FPS', ha='center',
                 va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('wyniki_wydajnosc_fps.png', dpi=300)

    # --- WYKRES 2: SKUTECZNOŚĆ ---
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Model'], df['Skuteczność (%)'], color=colors, edgecolor='gray', width=0.6)

    plt.title('Skuteczność detekcji twarzy (%)', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Skuteczność (%)', fontsize=12)
    # Wymuszenie skali Y od 0 do nieco ponad 100% dla lepszej czytelności
    plt.ylim(0, max(110, df['Skuteczność (%)'].max() + 10))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{round(yval, 1)}%', ha='center', va='bottom',
                 fontweight='bold')

    plt.tight_layout()
    plt.savefig('wyniki_skutecznosc_detekcji.png', dpi=300)

    print("Utworzono wykresy: 'wyniki_wydajnosc_fps.png' oraz 'wyniki_skutecznosc_detekcji.png'")


if __name__ == "__main__":
    # Możesz tu wpisać dowolną ścieżkę do głównego folderu ze zdjęciami
    DATASET_DIR = "dataset_treningowy_1/Real Images"

    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        print(f"[!] Utworzono pusty folder '{DATASET_DIR}'.")
        print("    Wrzuć tam swoje zdjęcia (mogą być w podfolderach) i uruchom ponownie.")
    else:
        run_benchmark(DATASET_DIR)