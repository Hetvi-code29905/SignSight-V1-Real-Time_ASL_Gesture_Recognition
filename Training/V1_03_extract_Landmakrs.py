   

import cv2
import numpy as np
import mediapipe as mp
import argparse
import json
import sys
import urllib.request
from pathlib import Path
from tqdm import tqdm

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

                 
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATASET_DIR, DATA_DIR, PHASE1_WORDS,
    SEQUENCE_LENGTH, MIN_FRAMES,
    USE_POSE, USE_LEFT_HAND, USE_RIGHT_HAND, USE_FACE,
    MEDIAPIPE_CONFIDENCE, FEATURE_DIM, MODEL_DIR
)

                                                                                
POSE_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
HAND_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

                                                                                
POSE_MODEL_PATH = MODEL_DIR / "pose_landmarker_full.task"
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"


                                                                                

def _download_if_missing(url: str, dest: Path) -> None:
                                                                            
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ⬇  Downloading {dest.name} …")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  ✅  Saved → {dest}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download {dest.name} from {url}.\n"
            f"  Original error: {e}\n"
            f"  Please download it manually and place it at: {dest}"
        ) from e


def ensure_models() -> None:
                                                            
    _download_if_missing(POSE_MODEL_URL, POSE_MODEL_PATH)
    _download_if_missing(HAND_MODEL_URL, HAND_MODEL_PATH)


                                                                               

class LandmarkExtractor:
           

    def __init__(self, mode: str = "video"):
                   
        ensure_models()

        running_mode = (
            RunningMode.VIDEO if mode == "video" else RunningMode.IMAGE
        )

                         
        pose_opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(POSE_MODEL_PATH)
            ),
            running_mode=running_mode,
            min_pose_detection_confidence=MEDIAPIPE_CONFIDENCE,
            min_pose_presence_confidence=MEDIAPIPE_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_CONFIDENCE,
            num_poses=1,
        )
        self._pose = mp_vision.PoseLandmarker.create_from_options(pose_opts)

                         
        hand_opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(HAND_MODEL_PATH)
            ),
            running_mode=running_mode,
            min_hand_detection_confidence=MEDIAPIPE_CONFIDENCE,
            min_hand_presence_confidence=MEDIAPIPE_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_CONFIDENCE,
            num_hands=2,
        )
        self._hand = mp_vision.HandLandmarker.create_from_options(hand_opts)

        self._mode  = running_mode
        self._ts_ms = 0                                                   

                                                                                

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self._pose.close()
        self._hand.close()

                                                                                

    def extract(self, bgr_frame: np.ndarray, frame_index: int = 0) -> np.ndarray:
                   
                                        
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                                                                          
                                                                                    
                                                                            
        ts_ms = self._ts_ms
        self._ts_ms += 33                              

        if self._mode == RunningMode.VIDEO:
            pose_result = self._pose.detect_for_video(mp_img, ts_ms)
            hand_result = self._hand.detect_for_video(mp_img, ts_ms)
        else:
            pose_result = self._pose.detect(mp_img)
            hand_result = self._hand.detect(mp_img)

        return self._build_vector(pose_result, hand_result)

                                                                                

    @staticmethod
    def _build_vector(pose_result, hand_result) -> np.ndarray:
                   
        parts = []

                                                                               
        if USE_POSE:
            if pose_result.pose_landmarks:
                lm = pose_result.pose_landmarks[0]                       
                pose = np.array(
                    [[p.x, p.y, p.z, p.visibility] for p in lm],
                    dtype=np.float32
                ).flatten()
            else:
                pose = np.zeros(33 * 4, dtype=np.float32)
            parts.append(pose)

                                                                                
        left_lm  = None
        right_lm = None

        if hand_result.handedness and hand_result.hand_landmarks:
            for idx, handedness_list in enumerate(hand_result.handedness):
                label = handedness_list[0].category_name                     
                if label == "Left":
                    left_lm  = hand_result.hand_landmarks[idx]
                elif label == "Right":
                    right_lm = hand_result.hand_landmarks[idx]

        if USE_LEFT_HAND:
            if left_lm is not None:
                lh = np.array(
                    [[p.x, p.y, p.z] for p in left_lm],
                    dtype=np.float32
                ).flatten()
            else:
                lh = np.zeros(21 * 3, dtype=np.float32)
            parts.append(lh)

        if USE_RIGHT_HAND:
            if right_lm is not None:
                rh = np.array(
                    [[p.x, p.y, p.z] for p in right_lm],
                    dtype=np.float32
                ).flatten()
            else:
                rh = np.zeros(21 * 3, dtype=np.float32)
            parts.append(rh)

        if USE_FACE:
                                                                              
                                                                           
            parts.append(np.zeros(468 * 3, dtype=np.float32))

        return np.concatenate(parts)


                                                                                

def interpolate_zeros(sequence: np.ndarray) -> np.ndarray:
           
    seq = sequence.copy()
    T = len(seq)

    for t in range(T):
        if np.sum(np.abs(seq[t])) < 1e-6:
            prev = next((i for i in range(t - 1, -1, -1) if not np.all(seq[i] == 0)), None)
            nxt  = next((i for i in range(t + 1, T)     if not np.all(seq[i] == 0)), None)

            if prev is not None and nxt is not None:
                alpha = (t - prev) / (nxt - prev)
                seq[t] = seq[prev] * (1 - alpha) + seq[nxt] * alpha
            elif prev is not None:
                seq[t] = seq[prev]
            elif nxt is not None:
                seq[t] = seq[nxt]

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

    shoulder_dist = np.linalg.norm(
        left_shoulder - right_shoulder,
        axis=1,
        keepdims=True
    )

    shoulder_dist = np.maximum(
        shoulder_dist,
        1e-6
    )

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


def pad_or_trim(frames: list, target_length: int) -> np.ndarray:
           
    if len(frames) >= target_length:
        indices = np.linspace(0, len(frames) - 1, target_length, dtype=int)
        return np.array([frames[i] for i in indices])
    else:
        pad_count = target_length - len(frames)
        padding   = [np.zeros(FEATURE_DIM, dtype=np.float32)] * pad_count
        return np.array(frames + padding)


def process_video(video_path: Path, extractor: LandmarkExtractor) -> np.ndarray | None:
           
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    frames    = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        landmark_vector = extractor.extract(frame, frame_index=frame_idx)
        frames.append(landmark_vector)
        frame_idx += 1

    cap.release()

    if len(frames) < MIN_FRAMES:
        return None

                              
    sequence = pad_or_trim(frames, SEQUENCE_LENGTH)

                                                
    sequence = interpolate_zeros(sequence)

               
    sequence = normalize_sequence(sequence)

    return sequence.astype(np.float32)


                                                                                

def extract_all(word_list: list, dataset_path: Path, output_path: Path):
                                                       

    print("=" * 60)
    print("  SignSight AI — Landmark Extraction (Tasks API)")
    print("=" * 60)
    print(f"\n  Words to process: {len(word_list)}")
    print(f"  Feature dim: {FEATURE_DIM}")
    print(f"  Sequence length: {SEQUENCE_LENGTH} frames\n")

                                                                                  
    print("  Checking model files …")
    ensure_models()
    print("  ✅  Models ready.\n")

    stats = {"processed": 0, "skipped": 0, "errors": 0, "class_counts": {}}

                                                                             
    with LandmarkExtractor(mode="video") as extractor:

        for word in tqdm(word_list, desc="Processing words"):
            word_dir   = dataset_path / word
            output_dir = output_path / "sequences" / word
            output_dir.mkdir(parents=True, exist_ok=True)

            if not word_dir.exists():
                print(f"\n  [SKIP] Word folder not found: {word}")
                stats["skipped"] += 1
                continue

                                                            
            videos = sorted([
                v for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv")
                for v in word_dir.glob(ext)
            ])

            if not videos:
                print(f"\n  [SKIP] No videos found in: {word}")
                stats["skipped"] += 1
                continue

            word_count = 0

                                                                    
            already_done = [
                v for v in videos
                if (output_dir / (v.stem + ".npy")).exists()
            ]
            if len(already_done) == len(videos):
                print(f"\n  [DONE] {word} — all {len(videos)} videos already extracted, skipping.")
                stats["class_counts"][word] = len(videos)
                continue

            for video_path in tqdm(videos, desc=f"  {word}", leave=False):
                out_file = output_dir / (video_path.stem + ".npy")

                                           
                if out_file.exists():
                    word_count += 1
                    continue

                try:
                    sequence = process_video(video_path, extractor)

                    if sequence is not None:
                        np.save(str(out_file), sequence)
                        word_count += 1
                        stats["processed"] += 1
                    else:
                        stats["skipped"] += 1

                except Exception as e:
                    print(f"\n  [ERROR] {video_path.name}: {e}")
                    stats["errors"] += 1

            stats["class_counts"][word] = word_count

                                                                
    stats_path = output_path / "extraction_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'─'*60}")
    print(f"  ✅ Sequences saved: {stats['processed']}")
    print(f"  ⚠️  Skipped:         {stats['skipped']}")
    print(f"  ❌ Errors:           {stats['errors']}")
    print(f"  📄 Stats saved → {stats_path}")
    print(f"{'─'*60}")
    print("\n  NEXT STEP:")
    print("  Run: python 03_build_dataset.py")
    print("=" * 60 + "\n")

    return stats


                                                                               

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignSight AI Landmark Extractor")
    parser.add_argument("--word", type=str, help="Extract single word only")
    parser.add_argument("--all",  action="store_true", help="Extract all words in dataset")
    args = parser.parse_args()

    if args.word:
        words = [args.word]
    elif args.all:
        words = [d.name for d in DATASET_DIR.iterdir() if d.is_dir()]
        print(f"[Info] Processing ALL {len(words)} classes")
    else:
        words = PHASE1_WORDS
        print(f"[Info] Processing Phase 1 words ({len(words)} classes)")

    extract_all(words, DATASET_DIR, DATA_DIR)
