   

import numpy as np
import json
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_DIR, SEQUENCE_LENGTH, FEATURE_DIM,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED
)


def load_sequences(sequences_dir: Path):
                                                                          
    
    X = []                      
    y = []                  
    label_map = {}                
    
    word_dirs = sorted([d for d in sequences_dir.iterdir() if d.is_dir()])
    
    if not word_dirs:
        print("[ERROR] No word directories found. Run 02_extract_landmarks.py first.")
        return None, None, None
    
    print(f"\nLoading sequences from: {sequences_dir}")
    print(f"Found {len(word_dirs)} word classes\n")
    
    for idx, word_dir in enumerate(word_dirs):
        word = word_dir.name
        label_map[word] = idx
        
        npy_files = sorted(word_dir.glob("*.npy"))
        
        for npy_file in npy_files:
            try:
                seq = np.load(str(npy_file))
                
                                
                if seq.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
                    print(f"  [SKIP] Shape mismatch in {npy_file.name}: {seq.shape}")
                    continue
                
                X.append(seq)
                y.append(idx)
            
            except Exception as e:
                print(f"  [ERROR] {npy_file.name}: {e}")
        
        count = len(npy_files)
        bar = "█" * min(count, 30)
        print(f"  [{idx:>4}] {word:<30} {count:>3} samples  {bar}")
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), label_map


def build_dataset():
    print("=" * 60)
    print("  SignSight AI — Build Training Dataset")
    print("=" * 60)
    
    sequences_dir = DATA_DIR / "sequences"
    
    if not sequences_dir.exists():
        print(f"\n[ERROR] Sequences directory not found: {sequences_dir}")
        print("  Please run 02_extract_landmarks.py first.")
        return
    
                                                                
    X, y, label_map = load_sequences(sequences_dir)
    
    if X is None:
        return
    
    num_classes = len(label_map)
    total       = len(X)
    
    if total == 0:
        print("[ERROR] No valid sequences found.")
        return
    
    print(f"\n{'─'*60}")
    print(f"  Total samples:  {total}")
    print(f"  Total classes:  {num_classes}")
    print(f"  Feature shape:  {X.shape}")
    print(f"  Label shape:    {y.shape}")
    
                                                                
                                    
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y if total > num_classes else None
    )
    
                                           
    val_relative = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_relative,
        random_state=RANDOM_SEED,
        stratify=y_trainval if len(X_trainval) > num_classes else None
    )
    
    print(f"\n  Train samples:  {len(X_train)}")
    print(f"  Val samples:    {len(X_val)}")
    print(f"  Test samples:   {len(X_test)}")
    
                                                                
    np.save(DATA_DIR / "X_train.npy", X_train)
    np.save(DATA_DIR / "y_train.npy", y_train)
    np.save(DATA_DIR / "X_val.npy",   X_val)
    np.save(DATA_DIR / "y_val.npy",   y_val)
    np.save(DATA_DIR / "X_test.npy",  X_test)
    np.save(DATA_DIR / "y_test.npy",  y_test)
    
                                                  
    label_map_path = DATA_DIR / "label_map.json"
    index_to_word  = {str(v): k for k, v in label_map.items()}
    
    full_map = {
        "word_to_index": label_map,
        "index_to_word": index_to_word,
        "num_classes": num_classes,
        "total_samples": total,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "sequence_length": SEQUENCE_LENGTH,
        "feature_dim": FEATURE_DIM,
    }
    
    with open(label_map_path, "w") as f:
        json.dump(full_map, f, indent=2)
    
    print(f"\n  ✅ Saved dataset files to: {DATA_DIR}")
    print(f"  ✅ Label map → {label_map_path}")
    
                                                                
    class_counts = {}
    for label in y:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    
    if max_count / min_count > 5:
        print(f"\n  ⚠️  Class imbalance detected!")
        print(f"      Min samples/class: {min_count}")
        print(f"      Max samples/class: {max_count}")
        print(f"      Consider class weighting during training.")
    
    print(f"\n{'─'*60}")
    print("  NEXT STEP:")
    print("  Run: python 04_train_model.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    build_dataset()
