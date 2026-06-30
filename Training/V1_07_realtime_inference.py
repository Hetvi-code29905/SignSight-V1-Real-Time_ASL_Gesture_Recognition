import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import argparse
import json
import time
import datetime
import sys
import threading
from collections import deque
from pathlib import Path

                                    
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

                     
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"

                                                                              
                                    
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.60
WEBCAM_INDEX = 0

USE_POSE = True
USE_LEFT_HAND = True
USE_RIGHT_HAND = True
USE_FACE = False

FEATURE_DIM = (
    (33 * 4 if USE_POSE       else 0) +
    (21 * 3 if USE_LEFT_HAND  else 0) +
    (21 * 3 if USE_RIGHT_HAND else 0)
)

MEDIAPIPE_CONFIDENCE = 0.6
MEDIAPIPE_TRACKING = 0.5

                      
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

POSE_MODEL_PATH = MODEL_DIR / "pose_landmarker_full.task"
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

def _download_if_missing(url, dest):
    import urllib.request
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  ⬇  Downloading {dest.name} …")
        urllib.request.urlretrieve(url, dest)
        print(f"  ✅  Saved → {dest}")

def ensure_models():
    _download_if_missing(POSE_MODEL_URL, POSE_MODEL_PATH)
    _download_if_missing(HAND_MODEL_URL, HAND_MODEL_PATH)

                                         
_lock = threading.Lock()
_latest_pose_r = None
_latest_hand_r = None

def _pose_callback(result, output_image, timestamp_ms):
    global _latest_pose_r
    with _lock:
        _latest_pose_r = result

def _hand_callback(result, output_image, timestamp_ms):
    global _latest_hand_r
    with _lock:
        _latest_hand_r = result

                                                                              
def build_landmark_vector(pose_r, hand_r) -> np.ndarray:
    parts = []

    if USE_POSE:
        if pose_r and pose_r.pose_landmarks:
            lm = pose_r.pose_landmarks[0]
            pose = np.array([[p.x, p.y, p.z, p.visibility] for p in lm], dtype=np.float32).flatten()
        else:
            pose = np.zeros(33 * 4, dtype=np.float32)
        parts.append(pose)

    left_lm = right_lm = None
    if hand_r and hand_r.handedness and hand_r.hand_landmarks:
        for idx, hlist in enumerate(hand_r.handedness):
            label = hlist[0].category_name
            if label == "Left":    left_lm = hand_r.hand_landmarks[idx]
            elif label == "Right": right_lm = hand_r.hand_landmarks[idx]

    if USE_LEFT_HAND:
        parts.append(
            np.array([[p.x, p.y, p.z] for p in left_lm], dtype=np.float32).flatten()
            if left_lm is not None else np.zeros(21 * 3, dtype=np.float32)
        )
    if USE_RIGHT_HAND:
        parts.append(
            np.array([[p.x, p.y, p.z] for p in right_lm], dtype=np.float32).flatten()
            if right_lm is not None else np.zeros(21 * 3, dtype=np.float32)
        )
    if USE_FACE:
        parts.append(np.zeros(468 * 3, dtype=np.float32))

    return np.concatenate(parts)


def interpolate_zeros(sequence: np.ndarray) -> np.ndarray:
    seq = sequence.copy()
    T = len(seq)
    
                                                                               
    slices = [
        (0, 132),            
        (132, 195),               
        (195, 258)                 
    ]
    
    for start, end in slices:
        for t in range(T):
                                                                      
            if np.sum(np.abs(seq[t, start:end])) < 1e-6:
                                                                                  
                prev = next((i for i in range(t - 1, -1, -1) if np.sum(np.abs(seq[i, start:end])) > 1e-6), None)
                nxt  = next((i for i in range(t + 1, T)     if np.sum(np.abs(seq[i, start:end])) > 1e-6), None)
                
                if prev is not None and nxt is not None:
                    alpha = (t - prev) / (nxt - prev)
                    seq[t, start:end] = seq[prev, start:end] * (1 - alpha) + seq[nxt, start:end] * alpha
                elif prev is not None:
                    seq[t, start:end] = seq[prev, start:end]
                elif nxt is not None:
                    seq[t, start:end] = seq[nxt, start:end]
    return seq


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
                                                                                           
    if not USE_POSE:
        return sequence.astype(np.float32)

    pose_end = 33 * 4
    pose = sequence[:, :pose_end].reshape(-1, 33, 4)

                                                                        
    left_shoulder = pose[:, 11, :3]
    right_shoulder = pose[:, 12, :3]
    
    center = (left_shoulder + right_shoulder) / 2.0
    pose[:, :, :3] -= center[:, np.newaxis, :]

    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder, axis=1, keepdims=True)
    shoulder_dist = np.maximum(shoulder_dist, 1e-6)
    pose[:, :, :3] /= shoulder_dist[:, np.newaxis, :]

    sequence[:, :pose_end] = pose.reshape(-1, pose_end)

                                                                        
    if USE_LEFT_HAND:
        lh = sequence[:, 132:195].reshape(-1, 21, 3)
        lh_wrist = lh[:, 0, :]
        lh_centered = lh - lh_wrist[:, np.newaxis, :]
        sequence[:, 132:195] = lh_centered.reshape(-1, 63)

                                                                         
    if USE_RIGHT_HAND:
        rh = sequence[:, 195:258].reshape(-1, 21, 3)
        rh_wrist = rh[:, 0, :]
        rh_centered = rh - rh_wrist[:, np.newaxis, :]
        sequence[:, 195:258] = rh_centered.reshape(-1, 63)

    return sequence.astype(np.float32)

                                                                                
BG_DARK = (15, 15, 25)
PURPLE = (180, 50, 230)
CYAN = (230, 200, 0)
GREEN = (30, 230, 100)
RED_ACC = (50, 80, 255)
WHITE = (255, 255, 255)
GRAY = (160, 160, 175)
ACCENT = (200, 230, 255)
GOLD = (0, 200, 255)

def overlay_rect(img, x1, y1, x2, y2, color, alpha=0.4):
    sub = img[y1:y2, x1:x2]
    rect = np.full_like(sub, color)
    cv2.addWeighted(rect, alpha, sub, 1.0 - alpha, 0, sub)
    img[y1:y2, x1:x2] = sub

def draw_landmarks(img, pose_r, hand_r):
                                                    
    h, w = img.shape[:2]
    if hand_r and hand_r.hand_landmarks:
        for lm_list in hand_r.hand_landmarks:
            for p in lm_list:
                cv2.circle(img, (int(p.x * w), int(p.y * h)), 4, PURPLE, -1)
    if pose_r and pose_r.pose_landmarks:
        for p in pose_r.pose_landmarks[0]:
                                                             
            cv2.circle(img, (int(p.x * w), int(p.y * h)), 3, CYAN, -1)

def draw_ui(image, frame_buffer, word_buffer, top_preds, fps, threshold):
    h, w = image.shape[:2]

                     
    overlay_rect(image, 12, 12, 280, 95, BG_DARK, alpha=0.55)
    cv2.putText(image, "SignSight AI v1.0", (22, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, ACCENT, 2, cv2.LINE_AA)
    cv2.putText(image, f"Buffer: {len(frame_buffer)}/{SEQUENCE_LENGTH} frames", (22, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
    cv2.putText(image, f"Engine FPS: {fps:.1f}", (22, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1)

                              
    px = w - 300
    overlay_rect(image, px - 12, 12, w - 12, 195, BG_DARK, alpha=0.55)
    cv2.putText(image, "Top Predictions:", (px, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ACCENT, 1)

    for idx, (word, conf) in enumerate(top_preds):
        y = 70 + idx * 38
        is_top = (idx == 0)
        
                             
        bar_w = int(180 * conf)
        bar_col = GREEN if conf >= threshold else RED_ACC
        cv2.rectangle(image, (px + 5, y + 8), (px + 5 + bar_w, y + 14), bar_col, -1)
        cv2.rectangle(image, (px + 5, y + 8), (px + 185, y + 14), GRAY, 1)

                     
        font_scale = 0.55 if is_top else 0.45
        thickness = 2 if is_top else 1
        text_col = WHITE if is_top else GRAY
        cv2.putText(image, f"{word}  {conf:.1%}", (px + 5, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_col, thickness, cv2.LINE_AA)

        if is_top and conf >= threshold:
            cv2.putText(image, "MATCH", (px + 200, y + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, GOLD, 1, cv2.LINE_AA)

                                
    buf_y = h - 90
    overlay_rect(image, 0, buf_y - 12, w, h, BG_DARK, alpha=0.4)
    cv2.putText(image, "Recognized Signs Log:", (20, buf_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ACCENT, 1)

    words_text = "  ->  ".join(word_buffer[-7:]) if word_buffer else "(No signs recognized yet)"
    cv2.putText(image, words_text, (20, buf_y + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)

    cv2.putText(image, "[C] Clear   [S] Save   [Q] Quit", (20, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1)

    return image

                                                                               
def run_inference():
    print("=" * 65)
    print("  SignSight AI — Step 7: Real-Time Inference (V1)")
    print("=" * 65)

    model_path = MODEL_DIR / "signsight_bilstm.keras"
    label_map_path = DATA_DIR / "label_map.json"

    if not model_path.exists():
        print(f"\n  [ERROR] Model file not found at: {model_path}")
        print("  Please download signsight_bilstm.keras from Drive to models/ folder.")
        return

    if not label_map_path.exists():
        print(f"\n  [ERROR] label_map.json not found at: {label_map_path}")
        print("  Please run V1_04_build_dataset.py first.")
        return

                    
    print(f"\n  Loading BiLSTM model...")
    model = tf.keras.models.load_model(str(model_path))
    
    with open(label_map_path) as f:
        label_info = json.load(f)

    index_to_word = label_info["index_to_word"]
    vocab_size = label_info["num_classes"]

                                                       
    is_active_signing = False
    active_sequence = []
    prev_wrists = None
    quiet_frames_count = 0                                             
    motion_history = deque(maxlen=5)                                  
    flip_input = False                                                 

    print(f"  Vocabulary size: {vocab_size} words loaded.")
    
    print(f"\n  Opening webcam {WEBCAM_INDEX}...")
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open webcam index {WEBCAM_INDEX}")
        return

                                                                                          
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    word_buffer = []
    top_preds = []

    fps_frames = 0
    fps_start = time.time()
    fps = 0.0

    print("  Downloading / checking MediaPipe model files...")
    ensure_models()

                                                                    
    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=MEDIAPIPE_CONFIDENCE,
        min_pose_presence_confidence=MEDIAPIPE_CONFIDENCE,
        min_tracking_confidence=MEDIAPIPE_TRACKING,
        num_poses=1,
    )
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        min_hand_detection_confidence=MEDIAPIPE_CONFIDENCE,
        min_hand_presence_confidence=MEDIAPIPE_CONFIDENCE,
        min_tracking_confidence=MEDIAPIPE_TRACKING,
        num_hands=2,
    )

    print("\n  ✅ Live! Auto-Trigger Mode Active.")
    print("    Simply perform a sign in front of the camera.")
    print("    Controls:")
    print("      [F] - Toggle Input Mirror (Fix Swapped Left/Right Hands)")
    print("      [C] - Clear Log")
    print("      [S] - Save Session to JSON")
    print("      [Q] - Quit\n")

    with mp_vision.PoseLandmarker.create_from_options(pose_opts) as pose_lm,         mp_vision.HandLandmarker.create_from_options(hand_opts) as hand_lm:

        frame_ts = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

                                                                                                      
            processing_frame = cv2.flip(frame, 1) if flip_input else frame.copy()
            rgb = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                                         
            cur_pose_r = pose_lm.detect_for_video(mp_img, frame_ts)
            cur_hand_r = hand_lm.detect_for_video(mp_img, frame_ts)
            frame_ts += 33

                             
            landmarks = build_landmark_vector(cur_pose_r, cur_hand_r)
            frame_buffer.append(landmarks)

                                                                                
            current_motion = 0.0
            
            if cur_pose_r and cur_pose_r.pose_landmarks:
                lm = cur_pose_r.pose_landmarks[0]
                                               
                curr_wrists = np.array([
                    [lm[15].x, lm[15].y, lm[15].z],             
                    [lm[16].x, lm[16].y, lm[16].z]               
                ])
                
                if prev_wrists is not None:
                                                                               
                    motion_dist = np.linalg.norm(curr_wrists - prev_wrists, axis=1)
                    current_motion = float(np.max(motion_dist))                            
                
                prev_wrists = curr_wrists
            else:
                prev_wrists = None

            motion_history.append(current_motion)
            avg_motion = sum(motion_history) / len(motion_history) if motion_history else 0.0
            
                                  
            hands_detected = cur_hand_r and cur_hand_r.hand_landmarks

                                                                                
                                                                                      
            if not is_active_signing:
                if avg_motion > 0.018 and hands_detected:
                    is_active_signing = True
                    active_sequence = [landmarks]
                    quiet_frames_count = 0
                    print("  🎬 Motion detected. Recording sign sequence...")
            else:
                active_sequence.append(landmarks)
                
                if avg_motion < 0.011 or not hands_detected:
                    quiet_frames_count += 1
                else:
                    quiet_frames_count = 0

                                                                                        
                if quiet_frames_count >= 8 or len(active_sequence) > 120:
                    is_active_signing = False
                    total_frames = len(active_sequence) - quiet_frames_count                              
                    
                    if total_frames >= 10:
                        trimmed_seq = active_sequence[:total_frames]
                        print(f"  ⏹ Motion stopped. Captured {total_frames} frames. Classifying...")

                                                                        
                        target_len = SEQUENCE_LENGTH
                        if total_frames >= target_len:
                            indices = np.linspace(0, total_frames - 1, target_len, dtype=int)
                            seq_arr = np.array([trimmed_seq[i] for i in indices])
                        else:
                            pad_count = target_len - total_frames
                            feat_dim = len(trimmed_seq[0])
                            padding = [np.zeros(feat_dim, dtype=np.float32)] * pad_count
                            seq_arr = np.array(trimmed_seq + padding)

                                                  
                        seq_arr = interpolate_zeros(seq_arr)
                        seq_arr = normalize_sequence(seq_arr)
                        seq_input = seq_arr[np.newaxis]

                                 
                        probs = model.predict(seq_input, verbose=0)[0]
                        top_idx = np.argsort(probs)[::-1][:3]
                        top_preds = [(index_to_word[str(i)], float(probs[i])) for i in top_idx]

                        best_word, best_conf = top_preds[0]
                        print(f"  ⭐ Top Prediction: '{best_word}' ({best_conf:.2%})")
                        if best_conf >= 0.40:
                            if not word_buffer or word_buffer[-1] != best_word:
                                word_buffer.append(best_word)
                    else:
                        print("  [Info] Motion too short, discarded.")

                             
            fps_frames += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = fps_frames / elapsed
                fps_frames = 0
                fps_start = time.time()

                                                           
            draw_landmarks(frame, cur_pose_r, cur_hand_r)

                                               
            frame = cv2.flip(frame, 1)

                                                          
            if is_active_signing:
                h, w = frame.shape[:2]
                cv2.circle(frame, (30, h - 130), 10, (50, 50, 255), -1)                     
                cv2.putText(frame, f"RECORDING SIGN: {len(active_sequence)} frames", (50, h - 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 255), 2)
            else:
                h, w = frame.shape[:2]
                cv2.circle(frame, (30, h - 130), 10, (100, 255, 100), -1)                     
                cv2.putText(frame, "SYSTEM READY (WAITING FOR SIGN)", (50, h - 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)

                                      
            lh_detected = False
            rh_detected = False
            if cur_hand_r and cur_hand_r.handedness:
                for hlist in cur_hand_r.handedness:
                    label = hlist[0].category_name
                    if label == "Left":    lh_detected = True
                    elif label == "Right": rh_detected = True

                                                           
            cv2.putText(frame, f"Left Hand: {'Active' if lh_detected else 'Missing'}", (22, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100) if lh_detected else (100, 100, 255), 1)
            cv2.putText(frame, f"Right Hand: {'Active' if rh_detected else 'Missing'}", (22, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100) if rh_detected else (100, 100, 255), 1)
            cv2.putText(frame, f"Hardware Mirror Flip: {flip_input} [Press F]", (22, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255) if flip_input else (180, 180, 180), 1)

                                                                                    
            draw_ui(frame, frame_buffer, word_buffer, top_preds, fps, CONFIDENCE_THRESHOLD)

            cv2.imshow("SignSight AI — Step 7: Live Classification (V1)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("f"):
                flip_input = not flip_input
                print(f"  [Toggle] Input mirror flip: {flip_input}")
            elif key == ord("c"):
                word_buffer.clear()
                top_preds = []
                print("  [Buffer cleared]")
            elif key == ord("s"):
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = DATA_DIR / f"session_{ts}.json"
                with open(save_path, "w") as f:
                    json.dump({"timestamp": ts, "signs": word_buffer}, f, indent=2)
                print(f"  💾 Session log saved → {save_path}")
    cap.release()
    cv2.destroyAllWindows()
    print("\n  ✅ Real-time session ended.")
    print("=" * 65)

if __name__ == "__main__":
    run_inference()
