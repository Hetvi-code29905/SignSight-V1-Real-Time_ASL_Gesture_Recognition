import os
import cv2
import random
import shutil
import numpy as np
from pathlib import Path

                                    
        
                                    

DATASET_DIR = r"Dataset_v1"
TARGET_VIDEOS = 10

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

                                    
                     
                                    

def change_brightness(frame):
    alpha = random.uniform(0.9, 1.1)
    beta = random.randint(-15, 15)
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

def add_noise(frame):
    noise = np.random.normal(
        0,
        5,
        frame.shape
    ).astype(np.int16)

    noisy = frame.astype(np.int16) + noise

    return np.clip(noisy, 0, 255).astype(np.uint8)

def slight_zoom(frame):
    h, w = frame.shape[:2]

    scale = random.uniform(1.02, 1.08)

    nh = int(h / scale)
    nw = int(w / scale)

    y1 = (h - nh) // 2
    x1 = (w - nw) // 2

    crop = frame[y1:y1+nh, x1:x1+nw]

    return cv2.resize(crop, (w, h))

                                    
              
                                    

def augment_video(input_path, output_path):

    cap = cv2.VideoCapture(input_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    aug_type = random.choice([
        "brightness",
        "noise",
        "zoom"
    ])

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if aug_type == "brightness":
            frame = change_brightness(frame)

        elif aug_type == "noise":
            frame = add_noise(frame)

        elif aug_type == "zoom":
            frame = slight_zoom(frame)

        out.write(frame)

    cap.release()
    out.release()

                                    
      
                                    

classes = sorted(
    [
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ]
)

print("\nAugmenting classes...\n")

for cls in classes:

    cls_path = os.path.join(DATASET_DIR, cls)

    videos = [
        f for f in os.listdir(cls_path)
        if f.lower().endswith(VIDEO_EXTS)
    ]

    current_count = len(videos)

                          
    if current_count <= 15:
        target_count = 25
    else:
        print(f"{cls:<25} has {current_count} videos (no augmentation needed)")
        continue

    needed = target_count - current_count

    print(
        f"{cls:<25} "
        f"{current_count} -> {target_count} (augmenting {needed} videos)"
    )

                                                                                   
    original_videos = [v for v in videos if "_aug_" not in v.lower()]
    if not original_videos:
        original_videos = videos            

    for i in range(needed):

        source_video = random.choice(original_videos)

        src_path = os.path.join(
            cls_path,
            source_video
        )

        stem = Path(source_video).stem
        out_name = f"{stem}_aug_{i}.mp4"
        out_path = os.path.join(cls_path, out_name)

                                                                  
        counter = 1
        while os.path.exists(out_path):
            out_name = f"{stem}_aug_{i}_{counter}.mp4"
            out_path = os.path.join(cls_path, out_name)
            counter += 1

        augment_video(
            src_path,
            out_path
        )

print("\nDone!")