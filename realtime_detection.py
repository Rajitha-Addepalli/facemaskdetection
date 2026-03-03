import cv2
import numpy as np
import tensorflow as tf

# Load Trained Model
model = tf.keras.models.load_model("mask_detector.keras")

# Load Face Detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

IMG_SIZE = 128

# Start Webcam
cap = cv2.VideoCapture(0)

print("🎥 Starting Real-Time Face Mask Detection... Press Q to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to Gray for Face Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extract Face ROI
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = np.expand_dims(face, axis=0) / 255.0

        # Prediction
        prediction = model.predict(face)[0][0]

        if prediction < 0.5:
            label = "Mask 😷"
            color = (0, 255, 0)
        else:
            label = "No Mask ❌"
            color = (0, 0, 255)

        # Draw Rectangle + Label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show Output
    cv2.imshow("Face Mask Detector", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Detection stopped.")