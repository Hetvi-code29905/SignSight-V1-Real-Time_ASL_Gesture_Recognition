   

import numpy as np
import json
import os
import sys
from pathlib import Path

                  
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

               
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"

def evaluate_model():
    print("=" * 60)
    print("  SignSight AI — Step 6: V1 Model Evaluation")
    print("=" * 60)
    
                                                               
    model_path = MODEL_DIR / "signsight_bilstm.keras"
    X_test_path = DATA_DIR / "X_test.npy"
    y_test_path = DATA_DIR / "y_test.npy"
    label_map_path = DATA_DIR / "label_map.json"

                                                               
    if not model_path.exists():
        print(f"[ERROR] Trained model file not found at: {model_path}")
        print("Please download it from Google Drive and place it in the models/ folder.")
        return
        
    if not (X_test_path.exists() and y_test_path.exists()):
        print(f"[ERROR] Test split files not found in: {DATA_DIR}")
        print("Please run V1_04_build_dataset.py first.")
        return

                                                               
    print(f"\n  Loading model: {model_path.name}...")
    model = tf.keras.models.load_model(str(model_path))
    
    print("  Loading test dataset...")
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    
    with open(label_map_path, "r") as f:
        label_info = json.load(f)
        
    index_to_word = label_info["index_to_word"]
    num_classes = label_info["num_classes"]
    
    print(f"  Test Samples : {len(X_test)}")
    print(f"  Classes      : {num_classes}\n")

                                                               
    print("  Predicting test samples...")
    y_proba = model.predict(X_test, batch_size=64, verbose=1)
    y_pred = np.argmax(y_proba, axis=1)

                                                               
    top1_acc = np.mean(y_pred == y_test)
    top5_acc = top_k_accuracy_score(y_test, y_proba, k=min(5, num_classes))

    print(f"\n{'─'*60}")
    print(f"  🏆 Top-1 Test Accuracy: {top1_acc:.4f}  ({top1_acc*100:.2f}%)")
    print(f"  🏆 Top-5 Test Accuracy: {top5_acc:.4f}  ({top5_acc*100:.2f}%)")
    print(f"{'─'*60}\n")

                                                               
    word_labels = [index_to_word[str(i)] for i in range(num_classes)]
    report = classification_report(y_test, y_pred, target_names=word_labels, zero_division=0)
    
    report_path = LOG_DIR / "v1_evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Top-1 Accuracy: {top1_acc:.4f} ({top1_acc*100:.2f}%)\n")
        f.write(f"Top-5 Accuracy: {top5_acc:.4f} ({top5_acc*100:.2f}%)\n\n")
        f.write(report)
        
    print(f"  📄 Full classification report saved → {report_path}")

                                                               
    plot_top_confused_classes(y_test, y_pred, word_labels, num_classes)

def plot_top_confused_classes(y_true, y_pred, class_names, num_classes, top_n=20):
                                                                     
    cm = confusion_matrix(y_true, y_pred)
    
                                                                  
    error_matrix = cm.copy()
    np.fill_diagonal(error_matrix, 0)
    
                          
    class_errors = np.sum(error_matrix, axis=1)
    top_confused_indices = np.argsort(class_errors)[::-1][:top_n]
    
                                                   
    sub_cm = cm[top_confused_indices][:, top_confused_indices]
    sub_labels = [class_names[idx] for idx in top_confused_indices]
    
    plt.figure(figsize=(14, 11))
    sns.heatmap(
        sub_cm,
        annot=True,
        fmt="d",
        cmap="OrRd",
        xticklabels=sub_labels,
        yticklabels=sub_labels,
        linewidths=0.5
    )
    plt.title(f"Top {top_n} Most Confused ASL Classes (Error Zoom)", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    
    cm_path = LOG_DIR / "v1_top_confused_classes.png"
    plt.savefig(str(cm_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Heatmap of top errors saved → {cm_path}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    evaluate_model()
