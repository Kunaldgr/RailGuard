# 🚆 RailSentinel

### AI-Powered Predictive Maintenance & Fault Detection for Railway Infrastructure

`Python` `XGBoost` `TensorFlow` `FastAPI` `scikit-learn` `Accuracy` `AUC` `License`

Catching failures before they happen, one sensor reading and one photo at a time. This system combines gradient-boosted machine learning on live sensor telemetry with a computer-vision model for physical component inspection, to predict train failures, flag unknown fault patterns, and rank maintenance urgency automatically.

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Why This Matters](#-why-this-matters)
- [Features](#-features)
- [Technology Stack](#%EF%B8%8F-technology-stack)
- [Architecture](#%EF%B8%8F-architecture)
- [Dataset](#-dataset)
- [Model Performance](#-model-performance)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Results & Visualizations](#-results--visualizations)
- [Limitations](#%EF%B8%8F-limitations)
- [Future Enhancements](#-future-enhancements)
- [Contact](#-contact)

---

## 🔍 Overview

Railway maintenance today mostly runs on fixed schedules — replace a part every 6 months, inspect every X kilometres — regardless of how the component is actually wearing. That wastes money on parts that are still fine, and misses parts that are quietly degrading between scheduled checks.

This project takes a **condition-based** approach instead: sensor telemetry is fed into trained models that estimate failure probability, flag statistically abnormal behaviour outside any known failure category, and rank which trains need attention first. A separate computer vision model automates visual inspection of fasteners from photographs.

The system outputs three things a maintenance engineer actually needs:
- 📊 **Failure probability + likely cause** — e.g. *"Bearing Failure, 94% probability"*
- 🚨 **Anomaly alerts** for fault patterns that don't match any known category
- 🛠️ **A ranked repair queue** — Repair Now / Repair Soon / Safe For Now

## 🌍 Why This Matters

- 🔧 Fixed-interval maintenance wastes resources on healthy parts and misses early failures
- 👷 Manual visual inspection of fasteners and track segments doesn't scale and is prone to human error/fatigue
- 🆕 Rare or previously unseen fault patterns can go undetected by systems that only recognize known categories
- ⚖️ Some failures genuinely can't be caught with high confidence from limited sensor features — this project measures that ceiling honestly instead of hiding it
- 🤖 Automating both the "read the sensors" and "look at the part" steps frees engineers to focus on judgment calls, not repetitive checks

## ✨ Features

- **Failure probability prediction** — binary + multiclass XGBoost models across 5 failure categories
- **Unsupervised anomaly detection** — Isolation Forest catches fault patterns outside known categories
- **Severity-gated priority engine** — decision-support ranking that never deprioritizes a known-critical case
- **Fastener defect classification** — EfficientNetB0 CNN via transfer learning, live image upload
- **Two independent FastAPI backends** — tabular models and CNN served separately
- **Searchable fleet dashboard** — browse 1,000 held-out, genuinely unseen train predictions
- **Drag-and-drop inspection tool** — upload a component photo, get an instant Defective/Non-Defective call

## 🛠️ Technology Stack

**Machine Learning**
- XGBoost — gradient-boosted trees for failure/failure-type prediction
- scikit-learn — Isolation Forest, LabelEncoder, class-weighting, threshold tuning
- joblib — model persistence

**Deep Learning**
- TensorFlow / Keras — EfficientNetB0 transfer learning for image classification

**Backend**
- FastAPI — REST API for both the tabular and vision models
- Uvicorn — ASGI server
- pandas / NumPy — data handling

**Frontend**
- HTML5 / CSS3 / vanilla JavaScript — Fetch API, no framework overhead

**Development**
- Kaggle Notebooks (GPU) — training environment
- Visual Studio Code — backend/frontend development
- Git — version control

## 🏗️ Architecture

```
                Sensor Data (per train)              Inspection Photo
                        │                                    │
                        ▼                                    ▼
              ┌───────────────────┐                ┌──────────────────┐
              │  FastAPI Service A │                │ FastAPI Service B │
              │     (port 8000)    │                │    (port 8001)    │
              └─────────┬──────────┘                └─────────┬─────────┘
                        │                                     │
        ┌───────────────┼───────────────┐                     │
        ▼               ▼               ▼                     ▼
  ┌───────────┐   ┌────────────┐  ┌────────────┐      ┌────────────────┐
  │  Module 1  │   │  Module 3  │  │  Module 5  │      │  EfficientNetB0 │
  │  XGBoost   │   │  Isolation │  │  Priority  │      │  (fastener CNN) │
  │ (fail prob │   │   Forest   │  │  Engine    │      │                 │
  │ + type)    │   │ (anomaly)  │  │ (gated)    │      │                 │
  └───────────┘   └────────────┘  └────────────┘      └────────────────┘
        │               │               │                     │
        └───────────────┴───────────────┘                     │
                        ▼                                     ▼
              Combined priority queue                 Defect classification
                        │                                     │
                        └───────────────┬─────────────────────┘
                                         ▼
                              🖥️ Browser Dashboard
                        (Fleet Monitor + Component Inspection)
```

### Why Two Separate Backend Services?
TensorFlow and the XGBoost/scikit-learn stack ended up needing different Python versions in this environment (Python 3.14 vs 3.12). Rather than fight the dependency conflict in one service, the tabular pipeline and the vision pipeline run as two independent FastAPI services, each in its own environment, called separately by the same frontend.

## 📊 Dataset

### Sensor Dataset — Indian Railway Failure Detection & Maintenance (100K)

| Attribute | Value |
|---|---|
| Total records | 100,000 |
| Features | 21 sensor/operational columns (bearing temp, wheel wear %, brake pressure, vibration, etc.) + 3 encoded categorical columns |
| Failure classes | Track Defect, Signal Failure, Brake Failure, Wheel Defect, Bearing Failure |
| No-failure rate | 69.2% |
| Excluded feature | `risk_score` — excluded as a leaky feature, since it was derived from the outcome itself |

**Failure type distribution (of confirmed failures):**

| Failure Type | Count | Share |
|---|---|---|
| Track Defect | 16,236 | 52.7% |
| Signal Failure | 4,221 | 13.7% |
| Brake Failure | 4,194 | 13.6% |
| Wheel Defect | 3,164 | 10.3% |
| Bearing Failure | 2,968 | 9.6% |

**Split strategy:** 1,000 rows held out first as a permanent, never-trained-on "fleet" set, then the remaining 99,000 split into ~97,000 train / ~2,000 test.

### Image Dataset — Railway Track Fault Detection (Fastener)

| Split | Images |
|---|---|
| Train | 980 |
| Validation | 280 |
| Test | 140 |
| Classes | Defective, Non-Defective |

## 📈 Model Performance

### Stage A — Binary Failure Prediction (XGBoost)

| Metric | No Failure | Failure |
|---|---|---|
| Precision | 0.82 | 0.99 |
| Recall | 1.00 | 0.51 |
| F1-score | 0.90 | 0.68 |

> Recall on the Failure class plateaued at ~51% regardless of decision threshold (tested via precision-recall curve, best F1 threshold = 0.589 vs. default 0.5) — diagnosed as a genuine data-separability ceiling rather than a tuning issue. Addressed by outputting probability scores instead of forcing a binary call.

### Stage B — Failure Type Classification (XGBoost, class-weighted)

| Failure Type | Precision | Recall | F1-score |
|---|---|---|---|
| Track Defect | 0.85 | 0.85 | 0.85 |
| Brake Failure | 0.41 | 0.41 | 0.41 |
| Signal Failure | 0.39 | 0.44 | 0.42 |
| Wheel Defect | 0.25 | 0.23 | 0.24 |
| Bearing Failure | 0.19 | 0.16 | 0.17 |

### Module 3 — Anomaly Detection (Isolation Forest)

| Metric | Value |
|---|---|
| Total flagged anomalous | 7,457 / 100,000 |
| Overlapping with known failures | 3,996 |
| Novel anomalies (no failure label) | 3,461 |

### Module 5 — Priority Engine Validation

| Severity | Weighted formula (v1) | Severity-gated (v2) |
|---|---|---|
| Critical → "Repair Now" | 48.3% | **100.0%** |
| High → "Repair Now" | 49.5% | **97.3%** |

### Fastener CNN (EfficientNetB0, transfer learning)

| Metric | Value |
|---|---|
| Validation accuracy | 91.4% |
| Real-world accuracy (informal testing) | ~70% |

**Training configuration:**

| Parameter | Value |
|---|---|
| Base model | EfficientNetB0 (frozen, ImageNet weights) |
| Input size | 224 × 224 × 3 |
| Dropout | 0.3 |
| Optimizer | Adam |
| Loss | Sparse categorical crossentropy |
| Batch size | 32 |
| Epochs (max, early-stopped) | 20 |
| Early stopping patience | 5 (monitoring val_accuracy) |

## 🚀 Installation

### Prerequisites
- Python 3.12+ (for the vision service) and Python 3.10+ (for the tabular service)
- 8 GB RAM minimum
- No GPU required for inference

### Tabular Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

### Vision Backend (separate environment)
```bash
cd backend-image
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install fastapi uvicorn tensorflow pillow python-multipart
python -m uvicorn main:app --reload --port 8001
```

### Frontend
```bash
cd frontend
python -m http.server 5500
```

## 💻 Usage

1. Start all three services (tabular backend, vision backend, frontend server)
2. Open `http://localhost:5500` in your browser
3. **Fleet Monitor tab** — search or browse the 1,000-train fleet, sorted by priority; click any row for full detail (probability, predicted failure type, anomaly flag, key sensor readings)
4. **Component Inspection tab** — drag and drop or browse for a fastener photo, get an instant Defective/Non-Defective classification with confidence score

## 📁 Project Structure

```
railsentinel/
├── backend/
│   ├── main.py                          # Tabular models API (port 8000)
│   ├── checkpoint/
│   │   ├── stageA_failure_predictor.pkl
│   │   ├── stageB_failure_type.pkl
│   │   ├── isolation_forest_anomaly.pkl
│   │   ├── categorical_encoders.pkl
│   │   ├── failure_type_encoder.pkl
│   │   ├── feature_cols.pkl
│   │   ├── stageA_best_threshold.pkl
│   │   ├── severity_weights.pkl
│   │   ├── training_data_with_predictions.csv
│   │   └── fleet_1000_predictions.csv
│   └── requirements.txt
│
├── backend-image/
│   ├── main.py                          # CNN classifier API (port 8001)
│   ├── venv/                            # isolated Python 3.12 environment
│   └── checkpoint/
│       └── fastener_classifier_final.keras
│
├── frontend/
│   └── index.html                       # Fleet Monitor + Component Inspection UI
│
└── notebooks/
    ├── 01_tabular_pipeline.ipynb         # Modules 1, 3, 5 — training & validation
    └── 02_fastener_cnn.ipynb             # EfficientNetB0 training
```

## 🔬 How It Works

**1. Sensor data → failure probability.** XGBoost (gradient-boosted decision trees) learns from 21 sensor features which combinations correlate with each of the 5 known failure types, outputting a continuous probability rather than a forced binary call — since threshold analysis showed a real ceiling on how separable failures are from healthy operation given the available features.

**2. Sensor data → anomaly score.** A separate, unsupervised Isolation Forest — trained only on confirmed-healthy readings — flags trains whose sensor profile is statistically unusual, independent of whether it matches any labelled failure type. This is the system's safety net for genuinely novel faults.

**3. Combined scores → priority ranking.** A rule-based engine gates on known severity first (Critical/High severity is never deprioritized regardless of model confidence) and falls back to a probability-weighted score for everything else.

**4. Fastener photo → defect classification.** A frozen EfficientNetB0 backbone (pretrained on ImageNet) extracts general visual features; a small trained classification head maps those features to Defective / Non-Defective, since the available image dataset (980 training images) was too small to train a deep CNN from scratch.

## 📸 Results & Visualizations

**Top contributing features — Stage A failure prediction:**
Rail wear (mm) dominated feature importance by a wide margin, followed by brake pressure, humidity, and brake pad wear — consistent with physical intuition about what drives railway component failure.

**Precision-recall trade-off:**
Precision stayed near 1.0 across nearly all thresholds while recall plateaued around 0.5 — the visual signature of a genuine separability ceiling rather than a poorly chosen cutoff.

## ⚠️ Limitations

- **Remaining Useful Life (RUL) estimation was scoped out** — the dataset is one row per train, not a time-series of repeated measurements, so a defensible "days until failure" regression isn't trainable on this data without a different data source
- **Minority failure classes (notably Bearing Failure) have weak recall** even after class-weighting — a real data/feature limitation, not an unaddressed bug
- **CNN accuracy drops on real-world photos** (~91% curated test set → ~70% informal real-world testing) — a known domain-shift effect of small training datasets
- **Two separate backend services** are currently required due to a Python-version/TensorFlow compatibility constraint, rather than one unified service
- **Anomaly flags aren't yet factored into the priority engine** — a train flagged as a novel anomaly can still be ranked "Safe For Now" if its failure probability is low, which is a gap worth closing

## 🚀 Future Enhancements

**Short-term**
- [ ] Fold anomaly flags directly into the priority engine's gating logic
- [ ] Add SHAP-based per-prediction explanations alongside global feature importance
- [ ] Unify both backends into a single service once environment constraints allow
- [ ] Expand the fastener dataset for improved real-world accuracy

**Long-term Vision**
- [ ] Real sensor telemetry streaming instead of batch fleet snapshots
- [ ] Genuine time-series logging to enable true RUL regression
- [ ] Mobile app for field engineers with offline-capable inspection
- [ ] Drone-based image capture for large-scale automated track/fastener inspection
- [ ] Expansion to wheel, bearing, and brake image-based inspection modules
- [ ] Cloud deployment replacing the current local dual-service setup

## 📞 Contact

**[Your Name]**

📧 Email: [kunaldagar4298@gmail.com] 💼 LinkedIn: [https://www.linkedin.com/in/kunal-dagar-661161322/] 💻 GitHub: [https://github.com/Kunaldgr]

## 📚 References

1. T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," *Proc. 22nd ACM SIGKDD*, 2016.
2. F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation Forest," *Proc. 8th IEEE ICDM*, 2008.
3. M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," *ICML*, 2019.
4. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR*, vol. 12, 2011.
5. Kaggle dataset: "Indian Railway Failure Detection and Maintenance (100K)."
6. Kaggle dataset: "Railway Track Fault Detection Dataset — Fastener."

---

*Built as a mini-project combining classical ML and deep learning for railway predictive maintenance.*