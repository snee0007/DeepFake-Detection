# DeepFake-Detection
Problem Statement
Deepfakes are AI-generated or AI-manipulated videos where a person's face or voice
is replaced with a synthetic version so realistic that it is difficult to distinguish
from the original. They are increasingly used for:

Misinformation campaigns and political manipulation
Identity theft and financial fraud
Non-consensual synthetic media and cyber harassment

Existing detection methods often rely on single-frame analysis and miss the
temporal inconsistencies introduced during GAN-based face synthesis — such as
unnatural blinking, inconsistent lighting, or subtle frame-to-frame facial movement
anomalies. This project addresses that gap.

System Architecture
Input Video
     │
     ▼
┌─────────────────────┐
│  Preprocessing      │  Frame extraction → Face detection → Crop & resize
│  Module             │  (OpenCV Haar cascade)
└────────┬────────────┘
         │  Individual face frames
         ▼
┌─────────────────────┐
│  Feature Extraction │  ResNext CNN → spatial feature vectors per frame
│  Module (ResNext)   │  Captures colour, shape, texture patterns
└────────┬────────────┘
         │  Frame-level feature sequence
         ▼
┌─────────────────────┐
│  Temporal Analysis  │  LSTM-RNN → models temporal dependencies
│  Module (LSTM)      │  Detects unnatural movement across frames
└────────┬────────────┘
         │  Video-level classification
         ▼
┌─────────────────────┐
│  Classification     │  REAL or DEEPFAKE + confidence score
│  Output             │
└─────────────────────┘

Modules
1. Preprocessing Module

Splits input videos into individual frames
Detects and crops face regions using OpenCV
Normalises and resizes frames for CNN input

2. Feature Extraction Module (ResNext CNN)

Applies a ResNext convolutional neural network to each frame
Extracts high-level spatial features: colour distributions, texture patterns, edge artifacts
These features capture pixel-level inconsistencies introduced by GAN face synthesis

3. Temporal Analysis Module (LSTM-RNN)

Receives the sequence of per-frame feature vectors from the CNN
LSTM learns temporal dependencies — how features change across time
Flags inconsistencies such as unnatural blinking, lighting shifts, or movement artifacts

4. Training & Evaluation Module

Trains end-to-end on labelled real vs deepfake video datasets
Optimised using cross-entropy loss
Evaluated on: Accuracy, Precision, Recall, F1 Score

5. Deepfake Detection Module (Deployment)

Integrated into a Django web application
Accepts video upload → runs full pipeline → returns verdict + confidence score
Supports both real-time and batch analysis modes


Tech Stack
ComponentTechnologyLanguagePythonDeep LearningPyTorch (ResNext CNN, LSTM)Web FrameworkDjangoComputer VisionOpenCVData VisualisationMatplotlib, SeabornDataset FormatCSV / ExcelDatabaseMySQL
Hardware Requirements

Processor: Intel i5 or above
RAM: 4 GB minimum
GPU: NVIDIA GTX 1650 or above (recommended for training)
Storage: 50 GB free disk space
OS: Windows 7 or above


Setup & Installation
bash# 1. Clone the repository
git clone https://github.com/yourusername/deepfake-video-detection.git
cd deepfake-video-detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run database migrations
python manage.py migrate

# 5. Start the Django server
python manage.py runserver
Open http://localhost:8000 in your browser.


Why ResNext over standard ResNet?
ResNext uses grouped convolutions (cardinality), which improves feature diversity
without increasing model complexity — important for detecting subtle GAN artifacts.
Why LSTM for video?
Single-frame CNNs miss temporal artifacts. GANs often introduce inconsistent blinking,
eye movement, or lighting that only becomes visible when analysing multiple frames
in sequence. LSTM captures this temporal signal.

Key Results
The model was trained and evaluated on video datasets containing both authentic and
GAN-generated deepfake content. Evaluation metrics used:

Accuracy — overall correct classifications
Precision — of predicted deepfakes, how many were actually fake
Recall — of actual deepfakes, how many were caught
F1 Score — harmonic mean of precision and recall


Research Context
This project is directly relevant to the following research directions in deepfake detection:

Temporal inconsistency analysis — the LSTM component addresses a known gap in
single-frame detection methods
GAN artifact detection — ResNext features capture the pixel-level residuals
introduced by encoder-decoder GAN architectures
AV-Deepfake1M (Cai, Dhall et al., ACM Multimedia 2024) — the dataset methodology
and detection pipeline used in this project aligns with this benchmark

Key References

Güera & Delp — Deepfake Video Detection Using Recurrent Neural Networks, AVSS 2018
Li, Chang & Lyu — Exposing AI Fake Videos by Detecting Eye Blinking, WIFS 2021
Yang, Li & Lyu — Exposing Deep Fakes Using Inconsistent Head Poses, ICASSP 2019
Afchar et al. — MesoNet: A Compact Facial Video Forgery Detection Network, 2018
Dolhansky et al. — The Deepfake Detection Challenge (DFDC) Dataset, 2019


Related Work
PaperYearKey ContributionDeepfake Video Detection Using RNN2018First RNN-based temporal detectionMesoNet2018Lightweight CNN for forgery detectionFace Warping Artifacts2019Detecting spatial warping residualsHead Pose Inconsistency2019Temporal 3D pose analysisEye Blinking Detection2021Physiological signal-based detection

Future Work

Extend to audio-visual deepfake detection — detecting lip-sync inconsistencies
between audio and video (aligned with AV-Deepfake1M benchmark)
Add saliency visualisation — highlight which facial regions triggered the detection
Multi-face video support — handle group videos with multiple subjects
Evaluate on FaceForensics++ and AV-Deepfake1M benchmarks for standardised comparison
