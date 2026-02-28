# ACCURATE: Arbitrary-shaped Continuum Reconstruction Under Robust Adaptive Two-view Estimation

Official implementation of:

**ACCURATE: Arbitrary-shaped Continuum Reconstruction Under Robust Adaptive Two-view Estimation (MICCAI 2026)**

---

## 📌 Overview

ACCURATE is a geometry-aware framework for accurate **3D reconstruction of arbitrary-shaped long slender continuum structures** (e.g., guidewires, catheters, and continuum robots) from **biplanar X-ray images**.

Unlike end-to-end learning-based reconstruction methods, ACCURATE explicitly integrates:

* Topology-aware Segmentation Network
* Geometry-consistent Curve Topology Traversal
* Epipolar-constrained Dynamic Programming

to enforce strict multi-view geometric consistency.

The pipeline consists of three sequential stages:

1. **Topology-aware Segmentation Network (TSN)**
   Extracts one-pixel-wide centerlines from X-ray images.

2. **Geometry-consistent Curve Topology Traversal (GCTT)**
   Converts unordered skeleton pixels into topology-ordered curves.

3. **Epipolar-constrained Dynamic Programming (ECDP)**
   Establishes globally optimal correspondences and reconstructs 3D shape via triangulation.

The method achieves **sub-millimeter reconstruction accuracy** on both simulated and real phantom datasets .

---

## 📂 Repository Structure

```
RECONSTRUCTION/
│
├── ACCURATE_dataset/        # dataset (simulation + phantom)
│
├── core/                    # reconstruction algorithms
│   ├── main.py              # full reconstruction pipeline
│   ├── main_point.py        # reconstruction using ordered points
│   ├── cur_dp.py            # ECDP dynamic programming
│   └── cur_dp_simple.py     # simplified DP implementation
│
├── utils/                   # utilities
│   ├── calibration.py       # camera geometry & triangulation
│   ├── model.py             # segmentation network definition
│   ├── seg_train.py         # segment network training
│   ├── seg_test.py          # segmentation inference
│   ├── seg_3d.py            # 3D reconstruction helpers
│   ├── process_figures.py   # preprocessing utilities
│   ├── draw_fig.py          # visualization
│   └── test.py              # reproducing paper results
│
├── experiment/              # experiment results
├── original_data/           # raw data
└── .gitignore
```

---


## 🚀 Usage

### 1️⃣ Segmentation (TSN)

Train segmentation network (For this demo we use simple Unet):

```bash
python utils/seg_train.py
```

Inference:

```bash
python utils/seg_test.py
```

Output:

```
centerline masks
```

---

### 2️⃣ Full Reconstruction Pipeline

Run end-to-end reconstruction:

```bash
python core/main.py
```

Pipeline:

```
Images
  ↓
Segmentation (TSN)
  ↓
Topology Traversal (GCTT)
  ↓
Dynamic Programming Matching (ECDP)
  ↓
Triangulation
  ↓
3D reconstruction
```

---

### 3️⃣ Reconstruction from Ordered Points

```bash
python core/main_point.py
```

This bypasses segmentation and traversal.

---

## 🧠 Core Algorithms

### Geometry-consistent Curve Topology Traversal (GCTT)

* Recovers curve ordering from unordered skeleton pixels
* Uses curvature, direction, and distance constraints
* Robust to occlusions and missing pixels

Implemented in:

```
core/main.py
```

---

### Epipolar-constrained Dynamic Programming (ECDP)

* Builds global correspondence matrix
* Minimizes cumulative point-to-epipolar-line distance
* Preserves topology ordering
* Handles one-to-many ambiguities via refinement

Implemented in:

```
core/cur_dp.py
```

---

### Triangulation

3D points are reconstructed using calibrated stereo geometry:

```
utils/calibration.py
```

---

## 📈 Evaluation Metrics

We follow standard reconstruction metrics:

* **Accuracy (Acc.)**
* **Completeness (Comp.)**
* **Chamfer Distance (Overall)**
* **Maximum Error (Max Err.)**

---

## 📷 Visualization

Generate reconstruction visualization:

```bash
python utils/draw_fig.py
```

Outputs:

* reconstructed 3D curves
* projection overlays
* comparison figures

---

## 📄 License

This project is released for academic research purposes only.