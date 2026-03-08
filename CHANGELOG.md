# Changelog

All notable changes to this project will be documented in this file.

The format is based on *Keep a Changelog* and this project follows semantic versioning.

---

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
