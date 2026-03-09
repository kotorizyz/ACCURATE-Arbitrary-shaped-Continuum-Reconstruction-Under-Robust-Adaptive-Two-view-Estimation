# ACCURATE Dataset

This dataset accompanies the paper:
"ACCURATE: Topology-aware Epipolar-consistent Reconstruction of Slender Continuum Structures"

## Contents
- 200 real phantom cases acquired using a clinical GE C-arm system
- 135 simulated cases with complex synthetic slender geometries
- Each case contains:
  - Two-view X-ray images
  - Binary segmentation masks
  - Camera calibration (intrinsic & extrinsic)
  - Ground-truth 3D centerline point cloud

## Folder Structure

```
ACCURATE_Dataset/
│
├── phantom/                 # 200 real phantom cases
│   └── case_001/
│   └── ...
│   └── case_200/
│
├── simulation/              # 135 simulated cases
│   └── case_001/
│   └── ...
│   └── case_135/
│
├── splits/                  # dataset splits
│   ├── simulation_train.txt
│   ├── simulation_test.txt
│   ├── phantom_train.txt
│   └── phantom_test.txt
│
└── README.md
```