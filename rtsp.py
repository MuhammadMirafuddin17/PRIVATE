# Library
import os
import cv2
import numpy as np
from ultralytics import YOLO
import insightface
from sklearn.metrics.pairwise import cosine_similarity

# Inisialisasi model InsightFace
face_model = insightface.app.FaceAnalysis(name='buffalo_l')
face_model.prepare(ctx_id=0, det_size=(640, 640))

# Fungsi pengenal identitas dengan cosine similarity
def recognize_identity_cosine(embedding, known_faces, known_names, threshold=0.5):
    if not known_faces:
        return "Tidak Dikenali"
    
    similarities = cosine_similarity([embedding], known_faces)[0]
    best_idx = np.argmax(similarities)
    if similarities[best_idx] >= threshold:
        return known_names[best_idx]
    else:
        return "Tidak Dikenali"

# Fungsi untuk menambahkan padding ke bounding box
def expand_bbox(x1, y1, x2, y2, img_shape, scale=0.3):
    h, w = img_shape[:2]
    dx = int((x2 - x1) * scale)
    dy = int((y2 - y1) * scale)
    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(w, x2 + dx)
    y2 = min(h, y2 + dy)
    return x1, y1, x2, y2

# Fungsi untuk menghasilkan warna yang berbeda
def get_color_from_name(name):
    np.random.seed(abs(hash(name)) % (2**32))
    color = np.random.randint(0, 255, 3).tolist()
    return tuple(map(int, color))  

# Fungsi untuk memuat wajah dari folder
def load_known_faces(folder_path):
    known_faces = []
    known_names = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                img = cv2.imread(image_path)

                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    faces = face_model.get(img_rgb)

                    if faces:
                        embedding = faces[0].embedding
                        label = os.path.basename(root)
                        known_faces.append(embedding)
                        known_names.append(label)
                        print(f"[✓] Memuat data wajah '{label}' dari {file}")
                    else:
                        print(f"[!] Tidak ditemukan wajah pada {file}")
                else:
                    print(f"[✗] Gagal memuat {file}")
    return known_faces, known_names

# Fungsi utama webcam + YOLO + InsightFace
def recognize_faces_from_webcam(known_faces, known_names, model_path, width=1280, height=720):
    model = YOLO(model_path)
    rtsp_url = "rtsp://admin:admin123@192.168.100.14:554/Streaming/Channels/202"
    cap = cv2.VideoCapture(rtsp_url)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[!] Gagal membuka webcam.")
        return

    print(f"[INFO] Resolusi kamera disetel ke {width}x{height}")
    print("[INFO] Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        faces_detected_count = 0

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()

            if conf < 0.5:
                continue

            x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, frame.shape)
            face_crop = frame[y1:y2, x1:x2]
            rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

            # Tampilkan crop wajah
            cv2.imshow("Cropped Wajah", face_crop)

            faces = face_model.get(rgb_crop.copy())
            name = "Tidak Dikenali"

            if faces:
                embedding = faces[0].embedding
                name = recognize_identity_cosine(embedding, known_faces, known_names, threshold=0.5)
                faces_detected_count += 1
            else:
                print("[!] InsightFace gagal mendeteksi wajah dari crop.")

            color = get_color_from_name(name)
            label_text = f"{name} ({conf:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Tampilkan jumlah wajah terdeteksi
        cv2.putText(frame, f"Jumlah Wajah: {faces_detected_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Deteksi & Pengenalan Wajah", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ========== MAIN ==========
if __name__ == "__main__":
    folder_path = "DATASET"  # Folder tempat menyimpan gambar wajah
    model_path = "yolov8m-face.pt"

    print("[INFO] Memuat wajah dari folder...")
    known_faces, known_names = load_known_faces(folder_path)
    print(f"[INFO] Total wajah yang dikenali: {len(known_faces)}")

    print("[INFO] Menyalakan kamera dan mulai deteksi...")

    #Resolusi 
    recognize_faces_from_webcam(known_faces, known_names, model_path, width=1920, height=1080)
