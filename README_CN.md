# 油罐检测README

## 简介

基于 Cascade R\-CNN \+ Swin Transformer 实现油罐目标检测，用于遥感影像油罐识别与提取。

## 环境依赖

- Python 3\.7\+

- PyTorch 1\.8\+

- CUDA 10\.2\+

- MMCV \+ MMDetection

```bash
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
pip install mmdet==2.28.2
pip install opencv-python numpy pandas
```

---

## 步骤 1：目录结构准备
```Plain Text
项目根目录
├── work_dir
│   └── cascade_rcnn_swin_oil   # 必须放自己训练好的模型权重 .pth 文件
├── tank_file                   # 业务数据主目录
│   ├── 地名1
│   │   ├── 20250101            # 日期文件夹
│   │   │   └── 1_original_tiff  # 存放影像、元数据、旋转信息
│   │   └── 20250102
│   │       └── 1_original_tiff
│   └── 地名2
│       └── 20250101
│           └── 1_original_tiff
├── configs
│   └── cascade_rcnn_swin_oil.py
├── detect.py
└── outputs  # 自动生成结果目录
```

---

## 步骤 2：放置权重文件

`work_dir\cascade_rcnn_swin_oil`
**里面需要放自己训练好的模型权重文件（\.pth）**
不放权重无法运行检测。

---

## 步骤 3：组织业务数据

`tank_file` 目录规则：

1. 第一层：**地名**（如 厂区 A、大连、宁波）

2. 第二层：**日期**（如 20250101）

3. 日期文件夹内直接放：**1\_original\_tiff**

4. `1_original_tiff` 内存放：

    - 待检测影像

    - 元数据

    - 旋转信息（如果用到CPD需要旋转信息）

    - **所有裁剪区域文件按 cut\_数字 命名**（如 cut\_1\.tif、cut\_2\.tif）

---

## 步骤 4：运行检测

### 单区域运行命令

```bash
python detect.py \
  --config configs/cascade_rcnn_swin_oil.py \
  --checkpoint work_dir/cascade_rcnn_swin_oil/你的权重文件.pth \
  --data_path tank_file/地名1/20250101/1_original_tiff \
  --output_dir outputs/地名1/20250101
```

### 批量运行所有日期 / 地名

```bash
python batch_detect.py
```

---

## 步骤 5：后续处理

如果需要进行后续的油量估计，需要运行2_adjust_bbox.py和3_match.py文件，操作详见Object Detection and Segmentation_Operation Guide.pdf



---

# 重要规则

1. **权重文件必须放在**
`work_dir\cascade_rcnn_swin_oil`
必须是自己训练完成的 \.pth 权重。

2. **数据目录必须按层级**
`tank_file → 地名 → 日期 → 1_original_tiff`
日期之后**直接是 1\_original\_tiff**，没有中间裁剪文件夹。

3. **裁剪区域命名规则**
`1_original_tiff` 内的所有裁剪区域文件
**必须以 cut\_数字 命名**
例如：cut\_1\.tif、cut\_2\.tif、cut\_3\.tif


