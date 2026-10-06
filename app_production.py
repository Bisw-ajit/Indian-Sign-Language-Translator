"""
Standalone Production Desktop Application
Runs live webcam feed in an OpenCV GUI window with the production v2.0 pipeline:
- MediaPipe Landmarker (Metal GPU)
- 78d Level 3 Biomechanical Extraction
- Calibrated Residual MLP
- Dual-Barrier OOD Filter
- Real-time HUD (Prediction, Confidence, FPS, OOD Filter Status)
"""
import time
import cv2
import numpy as np
from src.inference.pipeline import ProductionISLPipeline

def run_desktop_app():
    print("=" * 70)
    print("Starting Desktop ISL Recognition Application (v2.0 Production)")
    print("Press 'q' or ESC in the video window to exit.")
    print("=" * 70)

    pipeline = ProductionISLPipeline("configs/production.json")
    
    # Open default webcam (0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam (index 0). Check camera permissions.")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to grab camera frame. Retrying...")
            time.sleep(0.05)
            continue

        # Process frame through production pipeline
        t0 = time.perf_counter()
        result = pipeline.process_frame(frame)
        infer_latency_ms = (time.perf_counter() - t0) * 1000.0

        # Calculate FPS
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (curr_time - prev_time + 1e-6))
        prev_time = curr_time

        # Visualization HUD
        pred = result.get("prediction", "None")
        conf = result.get("confidence", 0.0) * 100.0
        is_confident = result.get("is_confident", False)
        bbox = result.get("bbox")
        filter_reason = result.get("filter_reason", "")

        # Draw bounding box if available
        if bbox is not None:
            x, y, w, h = bbox
            box_color = (0, 255, 0) if is_confident else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

        # Header background banner
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 75), (20, 20, 25), -1)

        # Display Prediction
        if is_confident:
            text = f"Sign: {pred} ({conf:.1f}%)"
            color = (0, 255, 0)
        else:
            text = f"Sign: {pred}"
            color = (0, 165, 255) if pred != "No Hand" else (160, 160, 160)

        cv2.putText(frame, text, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Status & FPS diagnostics
        diag_text = f"FPS: {fps:.1f} | Latency: {infer_latency_ms:.1f} ms"
        cv2.putText(frame, diag_text, (frame.shape[1] - 280, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        if not is_confident and filter_reason:
            cv2.putText(frame, f"Filter: {filter_reason}", (frame.shape[1] - 280, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        cv2.imshow("Indian Sign Language Translator (v2.0 Production)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Application exited cleanly.")

if __name__ == "__main__":
    run_desktop_app()
