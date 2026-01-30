# Model Changelog

Please record the following for each model:

- Initials of the contributor
- Subfolder name
- Date of introduction
- Summary of changes compared to previous versions

---

## 14/01/2026 Model Update - Binary Classifier v3.1.0

- **Author:** Daniele Lotito
- **Changes:**
  - Trained on updated lakehouse data (data up to 13th January 2026), no data left
    for the test set. Refit to improve inference speed, medium quality preset.
  - Autogluon updated to V1.5
- **Location:** `v3-1-0_ag-20260113_144232-refit-clone-opt-train_frac_1`

## 14/01/2026 Model Update - Binary Classifier v3.0.0

- **Author:** Daniele Lotito
- **Changes:**
  - Trained on updated lakehouse data (data up to 13th January 2026), no data left
    for the test set. No refit to improve prediction accuracy, medium quality
    preset.
  - Autogluon updated to V1.5
- **Location:** `v3-0-0_ag-20260113_144447-clone-opt-train_frac_1`

## 13/11/2025 Model Update - Binary Classifier v2.0.0

- **Author:** Daniele Lotito
- **Changes:**
  - Trained on updated lakehouse data (includes `granite-4.0` models, data up to
    30th October 2025).
  - Same dependencies as previous version (Autogluon v1.4).
  - 30/1 - Deprecated because of update to Autogluon version 1.5
- **Location:** `v2-0-0_ag-20251113_154241-refit-clone-opt`

## 12/11/2025 Model Update - Binary Classifier v1.1.0

- **Author:** Daniele Lotito
- **Changes:**
  - Removed dependencies on `limbomp` and `fastai`.
  - No changes to model size, datasets, or performance.
  - Requires PyTorch
  - Autogluon v1.4
  - 30/1 - Deprecated because of update to Autogluon version 1.5
- **Location:** `v1-1-0_ag-20251112_155927-refit-clone-opt`
