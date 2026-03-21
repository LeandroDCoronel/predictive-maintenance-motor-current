# Changelog

All notable changes to this project will be documented in this file.

The format is based on *Keep a Changelog* and this project follows semantic versioning.

---

## v4.0.0 – Feature Engineering & Signal Intelligence - 2026-03-21 

### Added
- Feature extraction module (`feature_extractor.py`)
- Window-based signal processing (100-sample windows)
- Statistical and energy-based features:
  - mean, std, max, min
  - RMS, skewness, kurtosis
  - signal energy
- New dataset pipeline (`prepare_features.py`)

### Improved
- Model input upgraded from raw signals to engineered features
- Significant performance improvement (accuracy ~98%)
- Better representation of motor behavior

### Changed
- Training now uses `features_dataset.csv` instead of raw dataset
- Input features expanded from 3 → 24

### Notes
This version marks the transition from raw signal ML to
feature-based industrial signal intelligence.

### Next
- Model robustness validation
- Real-world signal simulation
- Fault detection in continuous streams

## [v3.0] - 2026-03-07

### Added

* MATLAB dataset loader for the ITSC motor fault dataset.
* Data preprocessing pipeline to convert `.mat` files into a structured dataset.
* Generation of `motor_current_dataset.csv` with **65,000 samples**.
* Fault classification model using Random Forest.
* Model persistence using `joblib`.
* Real-time prediction script for fault detection.

### Dataset

* Source dataset: ITSC stator current dataset.
* Total samples: **65,000**
* Features:

  * Ia
  * Ib
  * Ic
* Target:

  * Motor condition / fault class

### Fault Classes

* ITSC_A10, ITSC_A20, ITSC_A30, ITSC_A40
* ITSC_B10, ITSC_B20, ITSC_B30, ITSC_B40
* ITSC_C10, ITSC_C20, ITSC_C30, ITSC_C40
* ITSC_HLT (healthy motor)

### Model

* Algorithm: Random Forest Classifier
* Training samples: 52,000
* Test samples: 13,000
* Accuracy: ~86%

### Pipeline

Data_ITSC.mat
→ prepare_dataset.py
→ motor_current_dataset.csv
→ train_model.py
→ motor_fault_model.pkl
→ predict_fault.py

### Notes

This version represents the **first functional end-to-end pipeline** for industrial motor fault detection using stator current signals.

---

## [Unreleased]

### Planned

* Feature extraction from signal windows
* Improved model accuracy (>90%)
* Real-time monitoring pipeline
* Industrial dashboard
