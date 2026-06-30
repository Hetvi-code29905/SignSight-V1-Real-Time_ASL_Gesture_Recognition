   

import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
SEQUENCES_DIR = DATA_DIR / "sequences"

USE_POSE = True
USE_LEFT_HAND = True
USE_RIGHT_HAND = True

def normalize_hands_in_dataset():
    print("=" * 60)
    print("  SignSight AI — Normalizing Hand Coordinates")
    print("=" * 60)

    if not SEQUENCES_DIR.exists():
        print(f"[ERROR] Sequences directory not found at: {SEQUENCES_DIR}")
        return

                         
    npy_files = list(SEQUENCES_DIR.glob("**/*.npy"))
    print(f"  Found {len(npy_files)} sequence files.")

                       
    for npy_path in tqdm(npy_files, desc="Normalizing shoulders and hands"):
        try:
            seq = np.load(str(npy_path))                   
            
                                                                                        
            if USE_POSE:
                pose_end = 33 * 4
                pose = seq[:, :pose_end].reshape(-1, 33, 4)
                
                                                           
                left_shoulder = pose[:, 11, :3]
                right_shoulder = pose[:, 12, :3]
                shoulder_center = (left_shoulder + right_shoulder) / 2.0
                
                                                                 
                pose[:, :, :3] -= shoulder_center[:, np.newaxis, :]
                seq[:, :pose_end] = pose.reshape(-1, pose_end)

                                                              
            if USE_LEFT_HAND:
                lh = seq[:, 132:195].reshape(-1, 21, 3)
                lh_wrist = lh[:, 0, :]
                lh_centered = lh - lh_wrist[:, np.newaxis, :]
                seq[:, 132:195] = lh_centered.reshape(-1, 63)
                
                                                                
            if USE_RIGHT_HAND:
                rh = seq[:, 195:258].reshape(-1, 21, 3)
                rh_wrist = rh[:, 0, :]
                rh_centered = rh - rh_wrist[:, np.newaxis, :]
                seq[:, 195:258] = rh_centered.reshape(-1, 63)
                
                                           
            np.save(str(npy_path), seq)

        except Exception as e:
            print(f"  [ERROR] Failed to process {npy_path.name}: {e}")

    print("\n  ✅ All sequences updated with wrist-centered hand coordinates!")
    print("  NEXT STEP: Run 'python V1_04_build_dataset.py' to compile the new dataset.")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    normalize_hands_in_dataset()
