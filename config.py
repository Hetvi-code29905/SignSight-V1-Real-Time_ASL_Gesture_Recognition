import os
from pathlib import Path

ROOT_DIR            = Path(__file__).resolve().parent
FULL_DATASET_DIR    = ROOT_DIR / "WASL_kggle_ds"
PHASE1_DATASET_DIR  = ROOT_DIR / "dataset_phase1"
DATASET_DIR         = ROOT_DIR / "Dataset_v1"
DATA_DIR            = ROOT_DIR / "data"
MODEL_DIR           = ROOT_DIR / "models"
LOG_DIR             = ROOT_DIR / "logs"
CHECKPOINT_DIR      = ROOT_DIR / "checkpoints"

for d in [DATA_DIR, MODEL_DIR, LOG_DIR, CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PHASE1_WORDS = [
    "hello", "good", "bye", "please", "sorry", "yes", "no", "help",
    "want", "need", "like", "who", "what", "where", "when", "why",
    "how", "because", "wait", "go", "come", "stop", "call", "tell",
    "give", "take", "mother", "father", "brother", "sister", "family",
    "man", "woman", "doctor", "deaf", "cousin", "daughter", "eat",
    "drink", "water", "food", "pizza", "apple", "today", "tomorrow",
    "yesterday", "work", "study", "school", "computer", "phone", "cold"
]

PHASE1_WORDS = list(dict.fromkeys(PHASE1_WORDS))[:52]

assert len(PHASE1_WORDS) <= 52, f"Too many words: {len(PHASE1_WORDS)}"

SEQUENCE_LENGTH     = 30
MIN_FRAMES          = 10
FEATURE_DIM_V1      = 258

MEDIAPIPE_CONFIDENCE    = 0.6
MEDIAPIPE_TRACKING      = 0.5

USE_POSE        = True
USE_LEFT_HAND   = True
USE_RIGHT_HAND  = True
USE_FACE        = False

FEATURE_DIM = (
    (33 * 4 if USE_POSE       else 0) +
    (21 * 3 if USE_LEFT_HAND  else 0) +
    (21 * 3 if USE_RIGHT_HAND else 0) +
    (468 * 3 if USE_FACE      else 0)
)

LSTM_UNITS_1    = 128
LSTM_UNITS_2    = 64
DENSE_UNITS     = 128
DROPOUT_RATE    = 0.4
LEARNING_RATE   = 0.001
BATCH_SIZE      = 8
BATCH_SIZE_GPU  = 64
EPOCHS          = 100
PATIENCE        = 15

TRAIN_SPLIT     = 0.70
VAL_SPLIT       = 0.15
TEST_SPLIT      = 0.15
RANDOM_SEED     = 42

CONFIDENCE_THRESHOLD    = 0.60
WEBCAM_INDEX            = 0

CURRENT_PHASE   = 1

if __name__ == "__main__":
    print(f"Phase 1 vocabulary size : {len(PHASE1_WORDS)} words")
    print(f"Feature dimension       : {FEATURE_DIM}")
    print(f"Sequence length         : {SEQUENCE_LENGTH} frames")
    print(f"\nPhase 1 words:")
    for i, w in enumerate(PHASE1_WORDS, 1):
        print(f"  {i:>3}. {w}")
