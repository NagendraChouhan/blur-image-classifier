import os
import time
import json
import warnings
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import cv2
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

LABELS = {0: "Sharp", 1: "Motion-Blurred", 2: "Defocus-Blurred"}
LABEL_COLORS = {"Sharp": "#2ECC71", "Motion-Blurred": "#E74C3C", "Defocus-Blurred": "#3498DB"}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def extract_features(image_path: str, img_size: int = 256) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")

    img = cv2.resize(img, (img_size, img_size))
    gray_u8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_f = gray_u8.astype(np.float64)
    h, w = gray_u8.shape
    feats = []

    lap = cv2.Laplacian(gray_u8, cv2.CV_64F) # detecting edges
    feats += [lap.var(), np.abs(lap).mean()]

    fft_shift = np.fft.fftshift(np.fft.fft2(gray_f)) #converts image into frequency domain
    magnitude = np.abs(fft_shift) #gets frequency strength
    cx, cy = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cy)**2 + (Y - cx)**2)
    max_r = np.sqrt(cx**2 + cy**2)
    r_inner = int(min(h, w) * 0.2)
    feats += [magnitude[dist > r_inner].sum() / (magnitude.sum() + 1e-8)]

    radial = np.zeros(10)
    for i in range(10):
        ring = magnitude[(dist >= max_r * i / 10) & (dist < max_r * (i + 1) / 10)]
        radial[i] = ring.mean() if len(ring) > 0 else 0
    radial /= (radial.sum() + 1e-8)
    feats += radial.tolist()

    gx = cv2.Sobel(gray_u8, cv2.CV_64F, 1, 0, ksize=3) #Edge Detection in x
    gy = cv2.Sobel(gray_u8, cv2.CV_64F, 0, 1, ksize=3) #Edge Detection in y
    grad_mag = np.sqrt(gx**2 + gy**2) #calculates total edge strength
    grad_dir = np.arctan2(gy, gx) #Edge Direction
    feats += [grad_mag.mean(), grad_mag.std(), grad_mag.max(),
              np.abs(gx).mean() / (np.abs(gy).mean() + 1e-8)]

    mag_hist, _ = np.histogram(grad_mag.ravel(), bins=20, range=(0, grad_mag.max() + 1e-8), density=True)
    dir_hist, _ = np.histogram(grad_dir.ravel(), bins=20, range=(-np.pi, np.pi), density=True)
    feats += mag_hist.tolist() + dir_hist.tolist()

    bh, bw = h // 4, w // 4
    local_vars = [
        cv2.Laplacian(gray_u8[r*bh:(r+1)*bh, c*bw:(c+1)*bw], cv2.CV_64F).var() #Local Edge Detection in Blocks
        for r in range(4) for c in range(4)
    ]
    lv = np.array(local_vars)
    feats += [lv.mean(), lv.std(), lv.min(), lv.max()]

    return np.array(feats, dtype=np.float32)


def load_dataset(dataset_root: str, max_per_class: int = None, img_size: int = 256):
    folder_map = {
        0: ["sharp"],
        1: ["motion"],
        2: ["defocus"],
    }
    X, y, paths = [], [], []
    root = Path(dataset_root)

    for label, folder_names in folder_map.items():
        folder = next((root / n for n in folder_names if (root / n).exists()), None)
        if folder is None:
            print(f"  [WARN] No folder for label {label} ({LABELS[label]})")
            continue

        image_files = sorted([f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS])
        if max_per_class:
            image_files = image_files[:max_per_class]

        print(f"  Loading {len(image_files):4d} images  -> {LABELS[label]}  ({folder.name}/)")
        for img_path in image_files:
            try:
                feat = extract_features(str(img_path), img_size=img_size)
                X.append(feat); y.append(label); paths.append(str(img_path))
            except Exception as e:
                print(f"    [skip] {img_path.name}: {e}")

    return np.array(X, dtype=np.float32), np.array(y), paths


def build_and_tune_pipeline(X_tr, y_tr, tune: bool = True) -> Pipeline:
    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
    ])
    #automatically handles multi-class classification
    if not tune:
        svm_pipe.set_params(svm__C=1.0, svm__gamma="scale")
        svm_pipe.fit(X_tr, y_tr)
        return svm_pipe

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("      Running GridSearchCV (SVM) …")
    svm_grid = {"svm__C": [0.1, 1, 10, 100], "svm__gamma": ["scale", "auto", 0.001, 0.01]}
    svm_gs = GridSearchCV(svm_pipe, svm_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=0)
    svm_gs.fit(X_tr, y_tr)
    print(f"      SVM   best params : {svm_gs.best_params_}  |  CV acc : {svm_gs.best_score_:.4f}")

    print("      Running GridSearchCV (RandomForest) …")
    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(class_weight="balanced", random_state=42)),
    ])
    rf_grid = {"rf__n_estimators": [200, 400], "rf__max_depth": [None, 12, 20]}
    rf_gs = GridSearchCV(rf_pipe, rf_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=0)
    rf_gs.fit(X_tr, y_tr)
    print(f"      RF    best params : {rf_gs.best_params_}  |  CV acc : {rf_gs.best_score_:.4f}")

    print("      Running GridSearchCV (GradientBoosting) …")
    gb_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gb", GradientBoostingClassifier(random_state=42)),
    ])
    gb_grid = {"gb__n_estimators": [100, 200], "gb__learning_rate": [0.05, 0.1], "gb__max_depth": [2, 3]}
    gb_gs = GridSearchCV(gb_pipe, gb_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=0)
    gb_gs.fit(X_tr, y_tr)
    print(f"      GB    best params : {gb_gs.best_params_}  |  CV acc : {gb_gs.best_score_:.4f}")

    candidates = [("SVM", svm_gs), ("RandomForest", rf_gs), ("GradientBoosting", gb_gs)]
    best_name, best_gs = max(candidates, key=lambda c: c[1].best_score_)
    print(f"      Selected model : {best_name}  (CV acc : {best_gs.best_score_:.4f})")
    return best_gs.best_estimator_


def plot_confusion_matrix(y_true, y_pred, label_names, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=label_names).plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix — Blur Classification", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_sample_predictions(image_paths, y_true, y_pred, probs, n=9, save_path=None):
    indices = np.random.choice(len(image_paths), min(n, len(image_paths)), replace=False)
    cols = 3; rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = axes.flatten()
    fig.suptitle("Sample Predictions", fontsize=15, fontweight="bold")

    for ax_i, idx in enumerate(indices):
        img = cv2.cvtColor(cv2.resize(cv2.imread(image_paths[idx]), (256, 256)), cv2.COLOR_BGR2RGB)
        pred_name = LABELS[y_pred[idx]]; true_name = LABELS[y_true[idx]]
        correct = y_pred[idx] == y_true[idx]
        axes[ax_i].imshow(img); axes[ax_i].set_xticks([]); axes[ax_i].set_yticks([])
        for spine in axes[ax_i].spines.values():
            spine.set_edgecolor(LABEL_COLORS[pred_name]); spine.set_linewidth(4)
        axes[ax_i].set_title(
            f"{'✓' if correct else '✗'} GT: {true_name}\nPred: {pred_name}  ({probs[idx][y_pred[idx]]:.0%})",
            fontsize=8.5, color="green" if correct else "red")

    for ax_i in range(len(indices), len(axes)):
        axes[ax_i].axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cv_scores(cv_scores, cv_folds, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, cv_folds + 1), cv_scores, color="#3498DB", alpha=0.85)
    ax.axhline(cv_scores.mean(), color="red", linestyle="--", label=f"Mean = {cv_scores.mean():.4f}")
    ax.set_xlabel("Fold"); ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1); ax.legend()
    ax.set_title(f"{cv_folds}-Fold Cross-Validation Accuracy", fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def train_and_evaluate(dataset_root, output_dir="outputs", max_per_class=None,
                       test_size=0.20, cv_folds=5, img_size=256, tune=True):
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "="*60)
    print("  BLUR CLASSIFIER — TRAINING")
    print("="*60)

    print(f"\n[1/5] Extracting features (img_size={img_size}) …")
    t0 = time.time()
    X, y, paths = load_dataset(dataset_root, max_per_class=max_per_class, img_size=img_size)

    if len(y) == 0:
        print(f"\n[ERROR] No images loaded. Check dataset path: {dataset_root}")
        return None, None

    print(f"      Done in {time.time()-t0:.1f}s  |  Total: {len(y)} images, {X.shape[1]} features")
    for lbl, name in LABELS.items():
        c = (y == lbl).sum()
        if c: print(f"      {name}: {c}")

    if len(np.unique(y)) < 2:
        print("\n[ERROR] Need at least 2 classes to train.")
        return None, None

    X_tr, X_te, y_tr, y_te, p_tr, p_te = train_test_split(
        X, y, paths, test_size=test_size, stratify=y, random_state=42)
    print(f"\n[2/5] Train: {len(y_tr)}  |  Test: {len(y_te)}")

    print(f"\n[3/5] Training SVM {'+ GridSearchCV' if tune else '(no tuning)'} …")
    t0 = time.time()
    pipe = build_and_tune_pipeline(X_tr, y_tr, tune=tune)
    print(f"      Training time: {time.time()-t0:.2f}s")

    model_path = os.path.join(output_dir, "blur_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"pipeline": pipe, "img_size": img_size, "labels": LABELS}, f)
    print(f"      Model saved → {model_path}")

    print(f"\n[4/5] {cv_folds}-fold cross-validation …")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_s = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"      CV Accuracy: {cv_s.mean():.4f} ± {cv_s.std():.4f}")

    print("\n[5/5] Test set evaluation …")
    y_pred = pipe.predict(X_te)
    probs = pipe.predict_proba(X_te)
    acc = accuracy_score(y_te, y_pred)
    unique_labels = sorted(np.unique(np.concatenate([y_te, y_pred])))
    label_names = [LABELS[i] for i in unique_labels]
    print(f"\n  Test Accuracy : {acc:.4f}\n\n{classification_report(y_te, y_pred, labels=unique_labels, target_names=label_names)}")

    print("  Saving plots …")
    plot_confusion_matrix(y_te, y_pred, label_names, save_path=os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    plot_sample_predictions(p_te, y_te, y_pred, probs, save_path=os.path.join(output_dir, "sample_predictions.png"))
    plt.close()
    plot_cv_scores(cv_s, cv_folds, save_path=os.path.join(output_dir, "cv_scores.png"))
    plt.close()

    results = {
        "test_accuracy": round(float(acc), 4), "cv_mean": round(float(cv_s.mean()), 4),
        "cv_std": round(float(cv_s.std()), 4), "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)), "img_size": img_size,
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  All outputs saved to : {output_dir}/")
    print("="*60 + "\n")
    return pipe, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blur Classifier — Training")
    parser.add_argument("--dataset",   required=True)
    parser.add_argument("--output",    default="outputs")
    parser.add_argument("--max",       type=int, default=None)
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--cv_folds",  type=int, default=5)
    parser.add_argument("--img_size",  type=int, default=256)
    parser.add_argument("--no_tune",   action="store_true")
    args = parser.parse_args()
    train_and_evaluate(args.dataset, args.output, args.max,
                       args.test_size, args.cv_folds, args.img_size, not args.no_tune)
