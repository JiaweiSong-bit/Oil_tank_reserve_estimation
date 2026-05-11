# Oil Tank Detection
ps: Currently, only the code and dataset for oil tank object detection have been open-sourced. The detection model weights, as well as the code and data for subsequent oil tank storage estimation and futures price forecasting, will be released after the acceptance of the paper.

## Introduction

Oil tank object detection is implemented based on Cascade R\-CNN \+ Swin Transformer, which is used for oil tank identification and extraction from remote sensing images\.
![Oil Tank Example](img/show.png)
## Environment Dependencies

- Python 3\.7\+

- PyTorch 1\.8\+

- CUDA 10\.2\+

- MMCV \+ MMDetection

```bash
pip install -r requirements.txt
```

---

## Step 1: Prepare Directory Structure

```plaintext
Project Root
├── work_dir
│   └── cascade_rcnn_swin_oil   # Must store trained model weights (.pth files)
├── tank_file                   # Main business data directory
│   ├── Place1
│   │   ├── 20250101            # Date folder
│   │   │   └── 1_original_tiff  # Store images, metadata, rotation info
│   │   └── 20250102
│   │       └── 1_original_tiff
│   └── Place2
│       └── 20250101
│           └── 1_original_tiff
├── configs
│   └── cascade_rcnn_swin_oil.py
├── detect.py
└── outputs  # Auto-generated results directory
```

---

## Step 2: Place Weight Files

`work_dir\cascade_rcnn_swin_oil`
**Place your own trained model weight files .pth here\.**
Detection cannot run without weights\.

---

## Step 3: Organize Business Data

Rules for `tank_file` directory:

1. Level 1: **Place name** e.g., Factory A, Dalian, Ningbo

2. Level 2: **Date** e\.g\., 20250101

3. Directly under date folder: **1\_original\_tiff**

4. Contents in `1_original_tiff`:

    - Images to be detected

    - Metadata

    - Rotation information

    - **All cropped area files named as cut\_number** e\.g\., cut\_1\.tif, cut\_2\.tif

---

## Step 4: Run Detection

### Single Area Command

```bash
python detect.py \
  --config configs/cascade_rcnn_swin_oil.py \
  --checkpoint work_dir/cascade_rcnn_swin_oil/your_weights.pth \
  --data_path tank_file/Place1/20250101/1_original_tiff \
  --output_dir outputs/Place1/20250101
```

### Batch Run All Dates / Places

```bash
python batch_detect.py
```
---

## Step 5: Post-processing

If subsequent oil quantity estimation is required, it is necessary to run the 2_adjust_bbox.py and 3_match.py files. For detailed instructions, please refer to the Object Detection and Segmentation_Operation Guide.pdf

---

# Important Rules

1. **Weight files must be placed in**
`work_dir\cascade_rcnn_swin_oil`
Must be your own trained \.pth weights\.

2. **Data directory hierarchy must be strictly followed**
`tank_file → Place → Date → 1_original_tiff`
**Directly 1\_original\_tiff** after date, no intermediate cropping folders\.

3. **Cropped area naming rule**
All cropped files in `1_original_tiff`
**Must be named cut\_number**
Example: cut\_1\.tif, cut\_2\.tif, cut\_3\.tif


