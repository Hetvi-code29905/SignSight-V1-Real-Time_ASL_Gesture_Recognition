import os
import shutil
from pathlib import Path

                                                    
                                                  
                                                    
SOURCE_DATASET = r"C:\Users\Hetvi\Desktop\pro-js\placements_pr-js\Sign_Sight_ai\WASL_kggle_ds"

          
                                                                      

TARGET_DATASET = "Dataset_v1"

WORDS = [
    "hello",
    "good",
    "bye",
    "please",
    "sorry",
    "yes",
    "no",
    "help",
    "want",
    "need",
    "like",
    "who",
    "what",
    "where",
    "when",
    "why",
    "how",
    "because",
    "wait",
    "go",
    "come",
    "stop",
    "call",
    "tell",
    "give",
    "take",
    "mother",
    "father",
    "brother",
    "sister",
    "family",
    "man",
    "woman",
    "doctor",
    "deaf",
    "cousin",
    "daughter",
    "eat",
    "drink",
    "water",
    "food",
    "pizza",
    "apple",
    "today",
    "tomorrow",
    "yesterday",
    "work",
    "study",
    "school",
    "computer",
    "phone",
    "cold"
]

                                                    
                          
                                                    
os.makedirs(TARGET_DATASET, exist_ok=True)

copied = 0
missing = []

for word in WORDS:

    src_folder = os.path.join(SOURCE_DATASET, word)
    dst_folder = os.path.join(TARGET_DATASET, word)

    os.makedirs(dst_folder, exist_ok=True)

    if not os.path.exists(src_folder):
        missing.append(word)
        continue

    video_count = 0

    for file in os.listdir(src_folder):

        if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            shutil.copy2(
                os.path.join(src_folder, file),
                os.path.join(dst_folder, file)
            )
            video_count += 1
            copied += 1

    print(f"{word:<15} -> {video_count} videos copied")

print("\n==============================")
print(f"Total videos copied: {copied}")
print(f"Total classes      : {len(WORDS)}")

if missing:
    print("\nWords not found:")
    for m in missing:
        print("-", m)