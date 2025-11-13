import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- Load your trained model ---
model = load_model("asl_gray_cnn_model.h5")

# Class labels (use the same order as in train_generator.class_indices)
class_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

# --- Start webcam ---
cap = cv2.VideoCapture(0)  # 0 = default camera

print("📸 Starting camera... Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Camera not found.")
        break

    # Draw a region of interest (ROI) box on screen
    x, y, w, h = 300, 300, 600, 600
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi = frame[y:y + h, x:x + w]

    # --- Preprocess the ROI ---
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    gray = (gray - gray.mean()) / (gray.std() + 1e-7)  # normalize
    gray = np.expand_dims(gray, axis=(0, -1))  # (1, 64, 64, 1)

    # --- Predict ---
    prediction = model.predict(gray, verbose=0)
    pred_class = np.argmax(prediction)
    label = class_labels[pred_class]

    # --- Display prediction ---
    cv2.putText(frame, f"Prediction: {label}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow("ASL Live Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Clean up ---
cap.release()
cv2.destroyAllWindows()
print("👋 Camera closed.")
