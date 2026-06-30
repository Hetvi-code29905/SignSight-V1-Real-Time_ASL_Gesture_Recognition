import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
import time
import os
import urllib.request
import threading
from pathlib import Path
from collections import deque
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_full.task"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"

mp_vision = mp.tasks.vision
RunningMode = mp.tasks.vision.RunningMode

GLOBAL_PREDICTIONS = []
GLOBAL_WORD_LOG = []
GLOBAL_IS_SIGNING = False
FLIP_INPUT = False
CAMERA_ACTIVE = False

GLOBAL_FRAME = None
CAMERA_THREAD = None
GLOBAL_LEFT_HAND_ACTIVE = False
GLOBAL_RIGHT_HAND_ACTIVE = False
GLOBAL_POSE_STATUS = "NOT_DETECTED"
GLOBAL_CONFIDENCE_THRESHOLD = 0.40

SEQUENCE_LENGTH = 30
FEATURE_DIM = 258
USE_POSE = True
USE_LEFT_HAND = True
USE_RIGHT_HAND = True

BG_DARK = (15, 15, 25)
PURPLE = (180, 50, 230)
WHITE = (245, 245, 245)

app = FastAPI(title="SignSight API Real-Time API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

def ensure_models():
    pose_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
    hand_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

    if not POSE_MODEL_PATH.exists():
        print(f"  Downloading Pose Landmarker model...")
        urllib.request.urlretrieve(pose_url, POSE_MODEL_PATH)
        print("  ✅ Download completed.")

    if not HAND_MODEL_PATH.exists():
        print(f"  Downloading Hand Landmarker model...")
        urllib.request.urlretrieve(hand_url, HAND_MODEL_PATH)
        print("  ✅ Download completed.")

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

def draw_landmarks(image, pose_r, hand_r):
    if pose_r and pose_r.pose_landmarks:
        lm = pose_r.pose_landmarks[0]
        for pair in [(11, 12), (11, 13), (12, 14), (13, 15), (14, 16)]:
            h, w = image.shape[:2]
            p1 = (int(lm[pair[0]].x * w), int(lm[pair[0]].y * h))
            p2 = (int(lm[pair[1]].x * w), int(lm[pair[1]].y * h))
            cv2.line(image, p1, p2, PURPLE, 2)
            cv2.circle(image, p1, 4, PURPLE, -1)
            cv2.circle(image, p2, 4, PURPLE, -1)

    if hand_r and hand_r.hand_landmarks:
        h, w = image.shape[:2]
        for hand in hand_r.hand_landmarks:
            for joint in hand:
                px = (int(joint.x * w), int(joint.y * h))
                cv2.circle(image, px, 3, (230, 180, 50), -1)

def run_model_prediction(seq_input, index_to_word, model):
    global GLOBAL_PREDICTIONS, GLOBAL_WORD_LOG, GLOBAL_CONFIDENCE_THRESHOLD
    try:
        try:
            probs = model(seq_input, training=False).numpy()[0]
        except Exception as tf_err:
            print(f"[WARN] Callable inference failed, falling back to predict: {tf_err}")
            probs = model.predict(seq_input, verbose=0)[0]
        top_idx = np.argsort(probs)[::-1][:5]
        
        preds = [
            {"word": index_to_word[str(i)], "confidence": float(probs[i])} 
            for i in top_idx
        ]
        GLOBAL_PREDICTIONS = preds
        
        best_word = preds[0]["word"]
        best_conf = preds[0]["confidence"]
        
        if best_conf >= GLOBAL_CONFIDENCE_THRESHOLD:
            if not GLOBAL_WORD_LOG or GLOBAL_WORD_LOG[-1] != best_word:
                GLOBAL_WORD_LOG.append(best_word)
    except Exception as e:
        print(f"[ERROR] Background inference failed: {e}")

def camera_loop():
    global GLOBAL_PREDICTIONS, GLOBAL_WORD_LOG, GLOBAL_IS_SIGNING, FLIP_INPUT, CAMERA_ACTIVE, GLOBAL_FRAME
    global GLOBAL_LEFT_HAND_ACTIVE, GLOBAL_RIGHT_HAND_ACTIVE, GLOBAL_POSE_STATUS, GLOBAL_CONFIDENCE_THRESHOLD

    model_path = MODELS_DIR / "signsight_bilstm.keras"
    label_map_path = DATA_DIR / "label_map.json"

    if not model_path.exists() or not label_map_path.exists():
        print("[ERROR] Model files missing. Cannot execute stream predictions.")
        return

    model = tf.keras.models.load_model(str(model_path))
    with open(label_map_path) as f:
        label_info = json.load(f)
    index_to_word = label_info["index_to_word"]

    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.5,
        num_poses=1,
    )
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
        num_hands=2,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    is_active_signing = False
    active_sequence = []
    prev_wrists = None
    quiet_frames_count = 0
    motion_history = deque(maxlen=5)

    try:
        with mp_vision.PoseLandmarker.create_from_options(pose_opts) as pose_lm, \
             mp_vision.HandLandmarker.create_from_options(hand_opts) as hand_lm:

            frame_ts = 0

            while cap.isOpened() and CAMERA_ACTIVE:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.015)
                    continue

                processing_frame = cv2.flip(frame, 1) if FLIP_INPUT else frame.copy()
                rgb = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                cur_pose_r = pose_lm.detect_for_video(mp_img, frame_ts)
                cur_hand_r = hand_lm.detect_for_video(mp_img, frame_ts)
                frame_ts += 33

                landmarks = build_landmark_vector(cur_pose_r, cur_hand_r)

                current_motion = 0.0
                pose_detected = cur_pose_r and cur_pose_r.pose_landmarks
                
                if pose_detected:
                    lm = cur_pose_r.pose_landmarks[0]
                    if lm[11].visibility > 0.5 and lm[12].visibility > 0.5:
                        shoulder_width = abs(lm[11].x - lm[12].x)
                        if shoulder_width > 0.45:
                            GLOBAL_POSE_STATUS = "TOO_CLOSE"
                        elif GLOBAL_POSE_STATUS != "TOO_FAST":
                            GLOBAL_POSE_STATUS = "OK"
                    elif GLOBAL_POSE_STATUS != "TOO_FAST":
                        GLOBAL_POSE_STATUS = "NOT_DETECTED"
                else:
                    GLOBAL_POSE_STATUS = "NOT_DETECTED"

                left_detected = False
                right_detected = False
                if cur_hand_r and cur_hand_r.handedness and cur_hand_r.hand_landmarks:
                    for hlist in cur_hand_r.handedness:
                        label = hlist[0].category_name
                        if label == "Left":    left_detected = True
                        elif label == "Right": right_detected = True
                
                GLOBAL_LEFT_HAND_ACTIVE = left_detected
                GLOBAL_RIGHT_HAND_ACTIVE = right_detected

                if pose_detected:
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
                        GLOBAL_IS_SIGNING = True
                        GLOBAL_POSE_STATUS = "OK"
                        active_sequence = [landmarks]
                        quiet_frames_count = 0
                else:
                    active_sequence.append(landmarks)
                    if avg_motion < 0.011 or not hands_detected:
                        quiet_frames_count += 1
                    else:
                        quiet_frames_count = 0

                    if quiet_frames_count >= 8 or len(active_sequence) > 120:
                        is_active_signing = False
                        GLOBAL_IS_SIGNING = False
                        total_frames = len(active_sequence) - quiet_frames_count
                        
                        if total_frames >= 10:
                            trimmed_seq = active_sequence[:total_frames]

                            if total_frames >= SEQUENCE_LENGTH:
                                indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)
                                seq_arr = np.array([trimmed_seq[i] for i in indices])
                            else:
                                pad_count = SEQUENCE_LENGTH - total_frames
                                feat_dim = len(trimmed_seq[0])
                                padding = [np.zeros(feat_dim, dtype=np.float32)] * pad_count
                                seq_arr = np.array(trimmed_seq + padding)

                            seq_arr = interpolate_zeros(seq_arr)
                            seq_arr = normalize_sequence(seq_arr)
                            seq_input = seq_arr[np.newaxis]

                            threading.Thread(
                                target=run_model_prediction, 
                                args=(seq_input, index_to_word, model), 
                                daemon=True
                            ).start()
                        elif total_frames > 2:
                            GLOBAL_POSE_STATUS = "TOO_FAST"

                draw_landmarks(frame, cur_pose_r, cur_hand_r)
                
                disp_frame = cv2.flip(frame, 1)

                ret, jpeg_buffer = cv2.imencode('.jpg', disp_frame)
                if not ret:
                    continue

                GLOBAL_FRAME = jpeg_buffer.tobytes()
                time.sleep(0.001)
    finally:
        cap.release()
        GLOBAL_FRAME = None
        GLOBAL_LEFT_HAND_ACTIVE = False
        GLOBAL_RIGHT_HAND_ACTIVE = False
        GLOBAL_POSE_STATUS = "NOT_DETECTED"

def generate_video_stream():
    global GLOBAL_FRAME, CAMERA_ACTIVE
    while True:
        if not CAMERA_ACTIVE or GLOBAL_FRAME is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Camera Offline", (190, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
            cv2.putText(placeholder, "Press [Start Camera] to begin translation", (130, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
            
            ret, jpeg = cv2.imencode('.jpg', placeholder)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.2)
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + GLOBAL_FRAME + b'\r\n')
            time.sleep(0.033)

@app.get("/")
def serve_dashboard():
    index_path = ROOT_DIR / "frontend" / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h2>[ERROR] frontend/index.html not found!</h2>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/video_feed")
def get_video_feed():
    ensure_models()
    return StreamingResponse(
        generate_video_stream(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/predictions")
def get_predictions():
    return {
        "top_predictions": GLOBAL_PREDICTIONS,
        "word_log": GLOBAL_WORD_LOG,
        "is_signing": GLOBAL_IS_SIGNING,
        "left_hand_active": GLOBAL_LEFT_HAND_ACTIVE,
        "right_hand_active": GLOBAL_RIGHT_HAND_ACTIVE,
        "pose_status": GLOBAL_POSE_STATUS
    }

@app.post("/api/clear")
def clear_predictions_log():
    global GLOBAL_WORD_LOG, GLOBAL_PREDICTIONS
    GLOBAL_WORD_LOG.clear()
    GLOBAL_PREDICTIONS.clear()
    return {"status": "cleared"}

@app.post("/api/toggle_mirror")
def toggle_mirror_flip():
    global FLIP_INPUT
    FLIP_INPUT = not FLIP_INPUT
    return {"mirror": FLIP_INPUT}

@app.post("/api/settings")
async def update_settings(request: Request):
    global GLOBAL_CONFIDENCE_THRESHOLD
    data = await request.json()
    threshold = data.get("confidence_threshold")
    if threshold is not None:
        GLOBAL_CONFIDENCE_THRESHOLD = float(threshold)
    return {"confidence_threshold": GLOBAL_CONFIDENCE_THRESHOLD}

@app.post("/api/camera/start")
def start_camera():
    global CAMERA_ACTIVE, CAMERA_THREAD
    if not CAMERA_ACTIVE:
        CAMERA_ACTIVE = True
        CAMERA_THREAD = threading.Thread(target=camera_loop, daemon=True)
        CAMERA_THREAD.start()
    return {"status": "started"}

@app.post("/api/camera/stop")
def stop_camera():
    global CAMERA_ACTIVE, GLOBAL_IS_SIGNING, GLOBAL_PREDICTIONS
    CAMERA_ACTIVE = False
    GLOBAL_IS_SIGNING = False
    GLOBAL_PREDICTIONS.clear()
    return {"status": "stopped"}

@app.get("/api/words")
def get_words_dictionary():
    label_map_path = DATA_DIR / "label_map.json"
    all_words = []
    
    if label_map_path.exists():
        with open(label_map_path) as f:
            label_info = json.load(f)
        all_words = sorted(list(label_info["word_to_index"].keys()))
        
    return all_words

@app.get("/api/video/{word}")
def get_reference_video(word: str):
    word_dir = ROOT_DIR / "Dataset_v1" / word
    if not word_dir.exists():
        word_dir = ROOT_DIR / "Dataset_v1" / word.lower()

    if word_dir.exists():
        video_files = list(word_dir.glob("*.mp4"))
        if video_files:
            return FileResponse(str(video_files[0]), media_type="video/mp4")
            
    return HTMLResponse("Reference clip not found.", status_code=404)

app.mount("/frontend", StaticFiles(directory=str(ROOT_DIR / "frontend")), name="static")
from mediapipe.tasks import python as mp_python

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)
