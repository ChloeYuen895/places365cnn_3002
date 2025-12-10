#!/usr/bin/env python3
"""
Real-time Places365 classification using webcam feed.

This script captures video from the default webcam and runs Places365 
classification on each frame, displaying the top prediction overlaid on the video.

Usage:
    python webcam_inference.py

Controls:
    - Press 'q' or ESC to quit
    - Press 's' to save current frame with prediction
    - Press 'p' to pause/unpause
"""

from pathlib import Path
import cv2
import numpy as np
import time
from datetime import datetime

# Paths relative to this script
BASE = Path(__file__).resolve().parent
PROTOTXT = BASE / 'deploy_vgg16_places365.prototxt'
CAFFEMODEL = BASE / 'vgg16_places365.caffemodel'
LABELS_FILE = BASE / 'categories_places365.txt'


def load_model():
    """Load the Places365 Caffe model."""
    if not PROTOTXT.exists() or not CAFFEMODEL.exists():
        raise FileNotFoundError('Model files missing: put prototxt and caffemodel next to this script')
    
    print("Loading Places365 model...")
    net = cv2.dnn.readNetFromCaffe(str(PROTOTXT), str(CAFFEMODEL))
    print("Model loaded successfully!")
    return net


def load_labels():
    """Load Places365 category labels."""
    if LABELS_FILE.exists():
        with open(LABELS_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        labels = [ln.split()[0].split('/')[-1].replace('_', ' ') for ln in lines]
        print(f"Loaded {len(labels)} category labels")
        return labels
    else:
        print("Warning: labels file not found, will use indices")
        return None


def preprocess_for_caffe(img: np.ndarray):
    """Preprocess image for Places365 model (same as gui_deploy.py)."""
    h, w = img.shape[:2]
    if h < w:
        new_h = 256
        new_w = int(w * 256 / h)
    else:
        new_w = 256
        new_h = int(h * 256 / w)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Center crop to 224x224
    y0 = (new_h - 224) // 2
    x0 = (new_w - 224) // 2
    crop = resized[y0:y0+224, x0:x0+224]
    
    # Create blob (note: Places365 uses BGR mean, no swapRB)
    blob = cv2.dnn.blobFromImage(
        crop, 
        scalefactor=1.0, 
        size=(224, 224), 
        mean=(104, 117, 123),  # BGR mean for Places365
        swapRB=False, 
        crop=False
    )
    return blob


def classify_frame(net, frame, labels=None):
    """Classify a single frame and return top prediction."""
    blob = preprocess_for_caffe(frame)
    net.setInput(blob)
    preds = net.forward()
    
    if preds is None:
        return None, 0.0
    
    probs = preds.flatten()
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    
    if labels and top_idx < len(labels):
        label_name = labels[top_idx]
    else:
        label_name = f'index_{top_idx}'
    
    return label_name, top_prob


def draw_prediction(frame, label, prob, fps=None):
    """Draw prediction text and FPS on frame."""
    h, w = frame.shape[:2]
    
    # Prepare text
    pred_text = f"{label}: {prob:.3f}"
    fps_text = f"FPS: {fps:.1f}" if fps is not None else ""
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    # Get text sizes
    (pred_w, pred_h), _ = cv2.getTextSize(pred_text, font, font_scale, thickness)
    (fps_w, fps_h), _ = cv2.getTextSize(fps_text, font, font_scale, thickness)
    
    # Draw prediction (top-left)
    padding = 10
    cv2.rectangle(frame, (5, 5), (5 + pred_w + padding, 5 + pred_h + padding), (0, 0, 0), -1)
    cv2.putText(frame, pred_text, (10, 5 + pred_h), font, font_scale, (255, 255, 255), thickness)
    
    # Draw FPS (top-right)
    if fps is not None:
        fps_x = w - fps_w - 15
        cv2.rectangle(frame, (fps_x - 5, 5), (fps_x + fps_w + 5, 5 + fps_h + padding), (0, 0, 0), -1)
        cv2.putText(frame, fps_text, (fps_x, 5 + fps_h), font, font_scale, (255, 255, 255), thickness)
    
    return frame


def main():
    """Main webcam inference loop."""
    try:
        # Load model and labels
        net = load_model()
        labels = load_labels()
        
        # Initialize webcam with toggle between camera 0 and 1
        current_camera = 0
        available_cameras = [0, 1]  # Only toggle between camera 0 and 1
        
        def switch_camera():
            nonlocal cap, current_camera
            cap.release()
            # Toggle between 0 and 1
            current_camera = 2 if current_camera == 0 else 0
            cap = cv2.VideoCapture(current_camera)
            if cap.isOpened():
                # Set camera properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                print(f"Switched to camera {current_camera}")
                return True
            else:
                # If switching failed, try to go back to previous camera
                current_camera = 1 if current_camera == 0 else 0
                cap = cv2.VideoCapture(current_camera)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    print(f"Camera switch failed, staying on camera {current_camera}")
                    return False
                return False
        
        print(f"Initializing camera {current_camera}...")
        cap = cv2.VideoCapture(current_camera)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set camera properties (optional)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Webcam opened successfully!")
        print("Controls:")
        print("  - Press 'q' or ESC to quit")
        print("  - Press 's' to save current frame")
        print("  - Press 'p' to pause/unpause")
        print("  - Press 'c' to switch camera")
        print("  - Press 'h' to toggle UI")
        
        # FPS calculation variables
        fps_counter = 0
        start_time = time.time()
        fps = 0.0
        
        paused = False
        show_help = True
        last_frame = None
        last_pred = ("", 0.0)
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from webcam")
                    break
                
                last_frame = frame.copy()
                
                # Classify frame
                start_inference = time.time()
                label, prob = classify_frame(net, frame, labels)
                inference_time = time.time() - start_inference
                
                if label is not None:
                    last_pred = (label, prob)
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter % 10 == 0:  # Update FPS every 10 frames
                    elapsed = time.time() - start_time
                    fps = fps_counter / elapsed
            
            # Use last frame if paused
            display_frame = last_frame.copy() if last_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Draw prediction and info
            if last_pred[0]:
                display_frame = draw_prediction(display_frame, last_pred[0], last_pred[1], fps)
            
            # Draw pause indicator
            if paused:
                h, w = display_frame.shape[:2]
                cv2.rectangle(display_frame, (w//2-50, h//2-15), (w//2+50, h//2+15), (0, 0, 255), -1)
                cv2.putText(display_frame, "PAUSED", (w//2-35, h//2+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw help and camera info
            if show_help:
                help_text = ["Controls:", "q/ESC: quit", "s: save", "p: pause", "c: camera", "h: toggle UI"]
                for i, text in enumerate(help_text):
                    y = display_frame.shape[0] - 20 - (len(help_text) - 1 - i) * 20
                    cv2.putText(display_frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Show current camera info in bottom-right
            camera_info = f"Camera: {current_camera}"
            h, w = display_frame.shape[:2]
            (text_w, text_h), _ = cv2.getTextSize(camera_info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(display_frame, camera_info, (w - text_w - 10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Places365 Webcam Inference', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('s') and last_frame is not None:  # Save frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"webcam_capture_{timestamp}.jpg"
                annotated_frame = draw_prediction(last_frame.copy(), last_pred[0], last_pred[1], fps)
                cv2.imwrite(filename, annotated_frame)
                print(f"Saved frame: {filename}")
            elif key == ord('p'):  # Pause/unpause
                paused = not paused
                print("PAUSED" if paused else "RESUMED")
            elif key == ord('h'):  # Toggle UI
                show_help = not show_help
            elif key == ord('c'):  # Switch camera (toggle between 0 and 1)
                if switch_camera():
                    # Reset FPS counter when switching cameras
                    fps_counter = 0
                    start_time = time.time()
                else:
                    print(f"Failed to switch to camera {current_camera}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        print("Webcam closed")


if __name__ == '__main__':
    main()