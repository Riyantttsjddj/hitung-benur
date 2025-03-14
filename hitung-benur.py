import cv2

# Buka kamera (0 untuk kamera utama)
cap = cv2.VideoCapture("http://127.16.0.1:8080/video")

while True:
    # Ambil frame dari kamera
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Terapkan thresholding
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Deteksi kontur (benur)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hitung jumlah benur
    jumlah_benur = len(contours)

    # Gambar kontur di frame
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Tampilkan jumlah benur di layar
    cv2.putText(frame, f"Benur: {jumlah_benur}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imwrite("hasil.jpg", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup kamera
cap.release()
cv2.destroyAllWindows()
