# Stress Detection in IT Professionals via Facial Emotion Recognition

A real-time stress-detection web application that classifies facial expressions from webcam input using a custom Convolutional Neural Network trained on FER2013, then infers stress from the proportion of "negative" emotions (angry, disgusted, fearful, sad). Built as a B.E. Mini Project at Lords Institute of Engineering & Technology (Osmania University).

> **Academic title:** *Stress Detection in IT Professionals by Image Processing and Machine Learning*

---

## What it does

Two complementary pipelines:

1. **Image / webcam pipeline** — Haar-cascade face detection on the input frame → face ROI → 48×48 grayscale → CNN classifies into one of 7 emotions {Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised}. Emotions in the negative set flag a "stressed" state.
2. **Tabular pipeline** — KNN classifier with PCA (6 components) over questionnaire/physiological features for stress-level classification. Also benchmarks against Gaussian Naive Bayes, SVM, Decision Tree, and a simple Neural Network.

Wrapped in a Django 4.x web app with authenticated user and admin flows for uploading images, running predictions, and reviewing results.

## Technical highlights

### CNN architecture (FER2013 emotion classifier)

```
Input (48×48×1 grayscale)
  → Conv2D(32, 3×3, relu)
  → Conv2D(64, 3×3, relu) → MaxPool(2×2) → Dropout(0.25)
  → Conv2D(128, 3×3, relu) → MaxPool(2×2)
  → Conv2D(128, 3×3, relu) → MaxPool(2×2) → Dropout(0.25)
  → Flatten
  → Dense(1024, relu) → Dropout(0.5)
  → Dense(7, softmax)
```

- **Optimizer:** Adam, learning rate 1e-4, decay 1e-6
- **Loss:** categorical cross-entropy
- **Training:** 50 epochs on FER2013 (28,709 train / 7,178 test), batch size 64
- **Face detection:** OpenCV `haarcascade_frontalface_default.xml`
- **Real-time inference:** OpenCV video capture loop with per-frame face ROI classification

### Tabular ML pipeline

- Feature reduction via PCA to 6 principal components
- KNN as primary classifier, with benchmark comparisons against GaussianNB, SVM, Decision Tree, and a simple feed-forward NN
- Implementations in `admins/utility/mymodels/` as separate modules per algorithm

## Tech stack

Python 3.10 · Keras · TensorFlow · NumPy · Matplotlib · scikit-learn · OpenCV · Pillow · Django 4.x · SQLite

## Repository layout

```
stress-detection/
├── StressDetection/               # Django project
│   ├── manage.py
│   ├── StressDetection/           # Settings, URLs, WSGI
│   ├── users/                     # User-facing views
│   │   └── utility/
│   │       ├── GetImageStressDetection.py
│   │       └── MyClassifier.py
│   ├── admins/                    # Admin views + ML utilities
│   │   └── utility/
│   │       ├── AlgorithmExecutions.py
│   │       └── mymodels/
│   │           ├── Stress_Detector_KNNClassifier.py
│   │           ├── Stress_Detector_GuassionNB.py
│   │           ├── Stress_Detector_SVM.py
│   │           ├── Stress_Detector_DecisionTreeClassifier.py
│   │           └── Stress_Detector_NN.py
│   └── kerasmodel.py              # CNN training / real-time webcam inference
└── README.md
```

## What's NOT in the repo (and why)

- `model.h5` — trained Keras weights (~9.4 MB)
- `haarcascade_frontalface_default.xml` — OpenCV-bundled, included with OpenCV install
- FER2013 dataset (~60 MB zipped)

To reproduce:
1. Download FER2013 from Kaggle: https://www.kaggle.com/datasets/msambare/fer2013
2. Place into `data/train/` and `data/test/` with the 7 emotion subfolders
3. Run `python kerasmodel.py` (with `mode = "train"` set) to retrain
4. `haarcascade_frontalface_default.xml` ships with OpenCV — available at `cv2.data.haarcascades`

## Running the Django app

```bash
cd StressDetection
python -m venv venv && venv\Scripts\activate        # Windows
# or:  source venv/bin/activate                     # Linux/Mac
pip install django tensorflow opencv-python numpy pillow scikit-learn matplotlib
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Real-time webcam inference (standalone, outside Django):

```bash
python kerasmodel.py           # mode switch inside script between "train" and "display"
```

## Status & limitations

- Stress is **inferred** from negative emotions, not directly measured — this is a proxy and does not replace clinical assessment.
- FER2013 is a challenging academic dataset with known label noise; state-of-the-art accuracy on FER2013 typically sits in the 65–75% range — absolute certainty on any single frame should not be expected.
- This is a Bachelor's Mini Project demonstrating CNN + classical ML pipelines end-to-end through a web interface, not a production wellness tool.

## License

MIT — see [LICENSE](LICENSE).
