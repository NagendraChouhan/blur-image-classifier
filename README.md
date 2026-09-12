# Blur Image Classifier

Classifies images as **Sharp**, **Motion-Blurred**, or **Defocus-Blurred** using handcrafted gradient/frequency-domain features and an SVM classifier (with an automatic comparison against RandomForest and GradientBoosting during training).

Inspired by *Blurred Image Detection and Classification* (Hsu & Chen, MMM 2008), which uses SVM on image gradient statistics to estimate blur extent, then further classifies blur type via point spread function (PSF) analysis.

## How it works

For each image, `extract_features` builds a feature vector from:

- **Laplacian variance & mean** — sharp edges produce high-variance Laplacian response; blur suppresses it.
- **FFT high-frequency energy ratio + 10-bin radial spectrum** — blur attenuates high spatial frequencies.
- **Sobel gradient magnitude/direction statistics + 20-bin histograms** — blurred images have gradient magnitude concentrated at small values, and motion blur biases the gradient direction histogram toward the blur axis.
- **4×4 block-local Laplacian variance (mean/std/min/max)** — captures whether blur is uniform (defocus/global) or localized (motion/partial).

These features feed a `StandardScaler` + classifier `Pipeline`, tuned via `GridSearchCV`.

## Project structure

```
train.py         # feature extraction + training (SVM / RandomForest / GradientBoosting comparison)
test.py          # inference on a single image or a folder
data/            # training images: data/sharp, data/motion, data/defocus  (not tracked in git)
test_img/        # sample images for quick inference checks
outputs/         # trained model (blur_model.pkl), metrics, and evaluation plots
```

## Setup

```bash
python -m venv myenv
source myenv/bin/activate
pip install opencv-python numpy scikit-learn matplotlib
```

## Training

Expects `data/<dataset_root>/{sharp,motion,defocus}/` folders of images. This project uses the [Blur Dataset](https://www.kaggle.com/datasets/kwentar/blur-dataset) from Kaggle:

```bibtex
@misc{blurdataset2020,
  title        = {Blur Dataset},
  author       = {Kwentar},
  year         = {2020},
  publisher    = {Kaggle},
  journal      = {Kaggle Datasets},
  howpublished = {\url{https://www.kaggle.com/datasets/kwentar/blur-dataset}}
}
```

```bash
python train.py --dataset data --output outputs
```

Options: `--max` (cap images per class), `--test_size`, `--cv_folds`, `--img_size`, `--no_tune` (skip GridSearchCV).

Training runs GridSearchCV over SVM, RandomForest, and GradientBoosting and keeps whichever wins on cross-validation accuracy. Saves `blur_model.pkl`, `results.json`, `confusion_matrix.png`, `cv_scores.png`, and `sample_predictions.png` to the output directory.

## Inference

```bash
# single image
python test.py --model outputs/blur_model.pkl --image test_img/test.jpeg

# folder of images
python test.py --model outputs/blur_model.pkl --folder test_img --save_plot outputs/predictions.png --save_json outputs/results.json
```

Output includes the predicted class, per-class probability bars, and a blur score (Laplacian-variance based, 0 = sharp, 1 = blurry). Predictions below 90% confidence are flagged with ⚠.

## Results

On the current dataset (350 images/class, 80/20 split):

| Metric | Value |
|---|---|
| Test accuracy | 90% |
| 5-fold CV accuracy | 88% ± 2.1% |

Per-image confidence on real-world photos can be lower than the aggregate accuracy suggests — especially for images that mix sharp and blurred regions in a single frame (e.g. a still foreground with motion-blurred people in the background), since the model predicts one label for the whole image rather than segmenting locally blurred regions the way the reference paper's full pipeline does.
