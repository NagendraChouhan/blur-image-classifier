import os
import json
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

import cv2

warnings.filterwarnings("ignore")

LABELS = {0: "Sharp", 1: "Motion-Blurred", 2: "Defocus-Blurred"}
LABEL_COLORS = {"Sharp": "#2ECC71", "Motion-Blurred": "#E74C3C", "Defocus-Blurred": "#3498DB"}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CONFIDENCE_THRESHOLD = 0.90


def extract_features(image_path: str, img_size: int = 256) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")

    img = cv2.resize(img, (img_size, img_size))
    gray_u8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_f = gray_u8.astype(np.float64)
    h, w = gray_u8.shape
    feats = []

    lap = cv2.Laplacian(gray_u8, cv2.CV_64F)
    feats += [lap.var(), np.abs(lap).mean()]

    fft_shift = np.fft.fftshift(np.fft.fft2(gray_f))
    magnitude = np.abs(fft_shift)
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

    gx = cv2.Sobel(gray_u8, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_u8, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_dir = np.arctan2(gy, gx)
    feats += [grad_mag.mean(), grad_mag.std(), grad_mag.max(),
              np.abs(gx).mean() / (np.abs(gy).mean() + 1e-8)]

    mag_hist, _ = np.histogram(grad_mag.ravel(), bins=20, range=(0, grad_mag.max() + 1e-8), density=True)
    dir_hist, _ = np.histogram(grad_dir.ravel(), bins=20, range=(-np.pi, np.pi), density=True)
    feats += mag_hist.tolist() + dir_hist.tolist()

    bh, bw = h // 4, w // 4
    local_vars = [
        cv2.Laplacian(gray_u8[r*bh:(r+1)*bh, c*bw:(c+1)*bw], cv2.CV_64F).var()
        for r in range(4) for c in range(4)
    ]
    lv = np.array(local_vars)
    feats += [lv.mean(), lv.std(), lv.min(), lv.max()]

    return np.array(feats, dtype=np.float32)


def compute_blur_score(image_path: str) -> float:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    lap = cv2.Laplacian(img, cv2.CV_64F).var()
    return float(1.0 / (1.0 + lap / 100.0))


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nTrain first: python train.py --dataset <path>")

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    if isinstance(bundle, dict):
        pipeline = bundle["pipeline"]
        img_size = bundle.get("img_size", 256)
        labels = bundle.get("labels", LABELS)
    else:
        pipeline = bundle
        img_size = 256
        labels = LABELS
        print("  [INFO] Loaded v1 model. Using img_size=256.")

    print(f"  Model loaded : {model_path}  (img_size={img_size})")
    return pipeline, img_size, labels


def predict_single(image_path: str, pipeline, img_size: int, labels: dict) -> dict:
    feat = extract_features(image_path, img_size=img_size).reshape(1, -1)
    pred = pipeline.predict(feat)[0]
    probs = pipeline.predict_proba(feat)[0]
    blur = compute_blur_score(image_path)
    return {
        "path": image_path,
        "prediction": labels[pred],
        "confidence": float(probs[pred]),
        "blur_score": round(blur, 4),
        "probabilities": {labels[i]: round(float(p), 4) for i, p in enumerate(probs)},
    }


def predict_folder(folder_path: str, pipeline, img_size: int, labels: dict) -> list:
    image_files = sorted([f for f in Path(folder_path).iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS])
    if not image_files:
        print(f"  [WARN] No supported images found in: {folder_path}")
        return []

    results = []
    for img_path in image_files:
        try:
            results.append(predict_single(str(img_path), pipeline, img_size, labels))
        except Exception as e:
            print(f"  [skip] {img_path.name}: {e}")
    return results


def print_result(info: dict):
    print(f"\n  File        : {info['path']}")
    print(f"  Prediction  : {info['prediction']}")
    print(f"  Confidence  : {info['confidence']:.2%}")
    print(f"  Blur score  : {info['blur_score']}  (0 = sharp, 1 = blurry)")
    print("  Per-class probabilities:")
    for cls, prob in info["probabilities"].items():
        print(f"    {cls:<20} {prob:5.1%}  {'█' * int(prob * 30)}")


def print_summary(results: list):
    counts = {}
    for r in results:
        counts[r["prediction"]] = counts.get(r["prediction"], 0) + 1

    print(f"\n  {'─'*58}")
    print(f"  {'File':<32} {'Prediction':<18} {'Conf':>6}")
    print(f"  {'─'*58}")
    for r in results:
        fname = Path(r["path"]).name
        fname = (fname[:29] + "…") if len(fname) > 32 else fname
        flag = " ⚠" if r["confidence"] < CONFIDENCE_THRESHOLD else ""
        print(f"  {fname:<32} {r['prediction']:<18} {r['confidence']:>6.1%}{flag}")
    print(f"  {'─'*58}")
    print(f"\n  Summary ({len(results)} images):")
    for cls, cnt in sorted(counts.items()):
        print(f"    {cls:<20} : {cnt} image{'s' if cnt != 1 else ''}")

    low_conf = [r for r in results if r["confidence"] < CONFIDENCE_THRESHOLD]
    if low_conf:
        print(f"\n  ⚠ {len(low_conf)} image(s) below {CONFIDENCE_THRESHOLD:.0%} confidence — consider retraining with more data.")


def plot_predictions(results: list, save_path: str = None, max_display: int = 12):
    results = results[:max_display]
    n = len(results)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.8))
    axes = np.array(axes).flatten()
    fig.suptitle("Blur Classification — Predictions", fontsize=15, fontweight="bold")

    for i, info in enumerate(results):
        img = cv2.cvtColor(cv2.resize(cv2.imread(info["path"]), (256, 256)), cv2.COLOR_BGR2RGB)
        pred = info["prediction"]
        color = LABEL_COLORS.get(pred, "gray")
        conf = info["confidence"]

        axes[i].imshow(img)
        axes[i].set_xticks([]); axes[i].set_yticks([])
        for spine in axes[i].spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(4)

        fname = Path(info["path"]).name
        fname = (fname[:18] + "…") if len(fname) > 20 else fname
        warn = " ⚠" if conf < CONFIDENCE_THRESHOLD else ""
        axes[i].set_title(f"{fname}\n{pred}  ({conf:.0%}){warn}", fontsize=8.5, color=color, fontweight="bold")

    for i in range(len(results), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Plot saved → {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blur Classifier — Inference")
    parser.add_argument("--model", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image")
    group.add_argument("--folder")
    parser.add_argument("--save_plot", default=None)
    parser.add_argument("--save_json", default=None)
    parser.add_argument("--max_display", type=int, default=12)
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  BLUR CLASSIFIER — INFERENCE")
    print("="*60)

    pipeline, img_size, labels = load_model(args.model)

    if args.image:
        print(f"\n  Classifying: {args.image}")
        info = predict_single(args.image, pipeline, img_size, labels)
        print_result(info)
        if args.save_json:
            with open(args.save_json, "w") as f:
                json.dump(info, f, indent=2)
            print(f"\n  Saved → {args.save_json}")

    elif args.folder:
        print(f"\n  Classifying folder: {args.folder}")
        results = predict_folder(args.folder, pipeline, img_size, labels)
        if results:
            print_summary(results)
            plot_predictions(results, save_path=args.save_plot, max_display=args.max_display)
            if args.save_json:
                with open(args.save_json, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  JSON saved → {args.save_json}")

    print("\n" + "="*60 + "\n")
