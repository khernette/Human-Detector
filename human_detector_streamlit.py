import os
import time
import csv
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from email.message import EmailMessage
import smtplib
from ultralytics import YOLO

# ===============================
# CONFIG – EDIT THESE
# ===============================

# Email settings
SENDER_EMAIL = "ken.ai.quilang@gmail.com"
SENDER_NAME = "Human Detector"
EMAIL_PASSWORD = "vewm usqj rpkf olbm"  # Gmail App Password
RECIPIENT_EMAIL = "ken.ai.quilang@gmail.com"   # Can be same as sender

EMAIL_SUBJECT = "Human Detected! Video Clip Attached"
EMAIL_BODY = "A human was detected by the camera. See attached video clip."

# Detection / recording logic
CONFIDENCE_THRESHOLD = 0.5         # YOLO confidence threshold
CLIP_DURATION_SECONDS = 5.0        # Fixed video length when human detected

CAMERA_INDEX = 0                   # Laptop camera index (0 is default)
MODEL_PATH = "yolov8n.pt"          # YOLO model weights (COCO)

# Paths (YOUR requested folders)
LOG_DIR = r"C:\Users\Ken\OneDrive\Desktop\AI\Human Detector\Logs"
ERROR_DIR = r"C:\Users\Ken\OneDrive\Desktop\AI\Human Detector\Error"

DETECTION_LOG_FILE = os.path.join(LOG_DIR, "detections_log.csv")
ERROR_LOG_FILE = os.path.join(ERROR_DIR, "error_log.csv")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ERROR_DIR, exist_ok=True)


# ===============================
# HELPER FUNCTIONS – LOGGING
# ===============================

def init_detection_log_file():
    """Create detection CSV log with header if it doesn't exist."""
    if not os.path.exists(DETECTION_LOG_FILE):
        with open(DETECTION_LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "class_name",
                "confidence",
                "is_human",
                "recording",
                "video_file"
            ])


def log_detection(class_name, confidence, is_human, recording, video_file):
    """Append one detection event to the CSV log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(DETECTION_LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            class_name,
            f"{confidence:.4f}",
            is_human,
            recording,
            video_file
        ])


def init_error_log_file():
    """Create error CSV log with header if it doesn't exist."""
    if not os.path.exists(ERROR_LOG_FILE):
        with open(ERROR_LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "event", "error_message", "extra_info"])


def log_error(event, error_message, extra_info=""):
    """Append an error event (e.g., email failure) to the error log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ERROR_LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, event, str(error_message), extra_info])


# ===============================
# EMAIL SENDING FUNCTION
# ===============================

def send_email_with_attachment(
    sender_email,
    sender_name,
    password,
    to_email,
    subject,
    body,
    attachment_path
):
    msg = EmailMessage()
    msg["From"] = f"{sender_name} <{sender_email}>"
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    # Attach the video file
    with open(attachment_path, "rb") as f:
        file_data = f.read()
        filename = os.path.basename(attachment_path)
        msg.add_attachment(
            file_data,
            maintype="video",
            subtype="mp4",
            filename=filename,
        )

    # Gmail SMTP SSL
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender_email, password)
        smtp.send_message(msg)


# ===============================
# HUMAN DETECTOR CORE
# ===============================

def start_human_detector():
    st.write("📹 Initializing camera and YOLO model...")

    init_detection_log_file()
    init_error_log_file()

    # Load YOLO model
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        log_error("model_load_failure", e, f"MODEL_PATH={MODEL_PATH}")
        return

    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        st.error("Could not open camera. Check your webcam and try again.")
        log_error("camera_open_failure", "Could not open camera", f"CAMERA_INDEX={CAMERA_INDEX}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 20.0  # default fallback

    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    recording = False
    out = None
    current_clip_path = None
    recording_start_time = None

    status_placeholder.info("Running... Stop Streamlit (Ctrl+C in terminal) to end the app.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                status_placeholder.warning("Failed to grab frame. Stopping.")
                log_error("frame_grab_failure", "Failed to grab frame from camera")
                break

            # Run YOLO detection
            try:
                results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            except Exception as e:
                status_placeholder.error(f"YOLO inference failed: {e}")
                log_error("inference_failure", e)
                break

            human_detected = False

            # Draw boxes and log ALL detections
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = model.names[cls_id]
                    conf = float(box.conf[0].item())

                    # Determine if this is a human (YOLO COCO uses "person")
                    is_human = cls_name.lower() == "person"

                    # Draw bounding box for every object
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        frame,
                        f"{cls_name} {conf:.2f}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                    # Log this detection
                    log_detection(
                        class_name=cls_name,
                        confidence=conf,
                        is_human=is_human,
                        recording=recording,
                        video_file=current_clip_path if current_clip_path else ""
                    )

                    # If it's a human and confidence is high enough, flag it
                    if is_human and conf >= CONFIDENCE_THRESHOLD:
                        human_detected = True

            # Start recording if a human is detected and we're not already recording
            if human_detected and not recording:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                current_clip_path = os.path.join(LOG_DIR, f"human_{timestamp}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                h, w = frame.shape[:2]
                out = cv2.VideoWriter(current_clip_path, fourcc, fps, (w, h))
                recording = True
                recording_start_time = time.time()
                status_placeholder.success(f"Human detected! Recording 3-second clip: {current_clip_path}")

            # If currently recording, write frames and stop after fixed duration
            if recording and out is not None:
                out.write(frame)
                elapsed = time.time() - recording_start_time
                if elapsed >= CLIP_DURATION_SECONDS:
                    status_placeholder.info("3 seconds elapsed. Stopping recording...")
                    recording = False
                    out.release()
                    out = None

                    # Try to send email with the recorded clip
                    try:
                        status_placeholder.info("Sending email with recorded clip...")
                        send_email_with_attachment(
                            SENDER_EMAIL,
                            SENDER_NAME,
                            EMAIL_PASSWORD,
                            RECIPIENT_EMAIL,
                            EMAIL_SUBJECT,
                            EMAIL_BODY,
                            current_clip_path
                        )
                        status_placeholder.success(f"Email sent with attachment: {current_clip_path}")
                    except Exception as e:
                        status_placeholder.error(f"Failed to send email: {e}")
                        log_error(
                            event="email_send_failure",
                            error_message=e,
                            extra_info=f"clip={current_clip_path}"
                        )

            # Show the frame in the Streamlit UI
            frame_placeholder.image(
                frame,
                channels="BGR",
                caption="Human Detector – stop the Streamlit process to end",
                use_column_width=True
            )

            # Optional small delay
            # time.sleep(0.01)

    except Exception as e:
        status_placeholder.error(f"Unexpected error: {e}")
        log_error("unexpected_exception", e)
    finally:
        cap.release()
        status_placeholder.info("Camera released. App stopped.")
        log_error("app_stopped", "Main loop exited / app stopped")


# ===============================
# STREAMLIT UI
# ===============================

def main():
    st.title("🧍 Human Detector with YOLO")
    st.write("""
    This app:
    - Uses your **laptop camera**  
    - Detects **all objects** using YOLO  
    - When a **human (person)** is detected:
      - Records a **3-second video**
      - Saves it to `C:\\Users\\Ken\\OneDrive\\Desktop\\AI\\Human Detector\\Logs`
      - Sends the video via email  
    - Logs **all detections** to `detections_log.csv` in the Logs folder  
    - Logs **all errors** to `error_log.csv` in the Error folder  
    """)

    st.warning(
        "Once you click **Start Detector**, the camera will turn on and run in a loop.\n"
        "To stop the app, stop the Streamlit process (Ctrl+C in the terminal/command prompt)."
    )

    if st.button("Start Detector"):
        start_human_detector()


if __name__ == "__main__":
    main()
