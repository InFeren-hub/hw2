# HW2 期中作业

本仓库包含三个task的 代码框架与已完成的实验结果。

## 目录说明

```text
.
├── task1_pet_classification/       # 任务1：Oxford-IIIT Pet 分类
├── task2_detection_tracking/       # 任务2：VisDrone 检测、视频跟踪、越线计数
├── task3_segmentation/             # 任务3：U-Net 语义分割
├── experiments/                    # 任务1/任务3训练指标与权重
├── weights/                        # 最终模型权重
├── runs/                           # 训练、验证、跟踪输出
├── report/                         # PDF报告、LaTeX和图表
└── data/                           # 本地数据集
```

## 环境

```bash
pip install torch torchvision matplotlib pillow opencv-python ultralytics lap
```

## 数据路径

Oxford-IIIT Pet：

```text
data/
├── images/
└── annotations/
    ├── trainval.txt
    ├── test.txt
    └── trimaps/
```

VisDrone for YOLO ：

```text
data/VisDrone/
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

VisDrone 转换命令：

```bash
python -m task2_detection_tracking.convert_visdrone_to_yolo \
  --raw-root path/to/VisDrone2019-DET \
  --out-root data/VisDrone
```

## 任务1：宠物分类

训练：

```bash
python -m task1_pet_classification.train --experiment all --epochs 20
```

生成曲线：

```bash
python -m task1_pet_classification.plot_results --metrics-dir experiments --out-dir report/figures
```

结果：

| 实验 | Best Test Acc | Best Epoch |
|---|---:|---:|
| Scratch ResNet34 | 42.60% | 18 |
| ImageNet Pretrained ResNet34 | 92.01% | 13 |
| Pretrained + SEBlock | 92.26% | 19 |
| Lower LR | 92.07% | 10 |
| Higher LR | 90.90% | 17 |

## 任务2：检测、跟踪、越线计数

训练：

```bash
python -m task2_detection_tracking.train
```

最终模型权重：

```text
weights/best.pt
```

VisDrone 验证集结果（imgsz=1280）：

| 类别 | mAP50 | mAP50-95 |
|------|:---:|:---:|
| all | 0.533 | 0.329 |
| car | 0.876 | 0.636 |
| bus | 0.686 | 0.520 |
| pedestrian | 0.646 | 0.324 |
| motor | 0.624 | 0.307 |
| van | 0.566 | 0.415 |
| truck | 0.499 | 0.342 |
| people | 0.495 | 0.208 |
| tricycle | 0.399 | 0.237 |
| bicycle | 0.324 | 0.162 |
| awning-tricycle | 0.212 | 0.135 |

视频跟踪：

```bash
python -m task2_detection_tracking.track_video
```

越线计数：

```bash
python -m task2_detection_tracking.line_count
```

输出：

```text
runs/task2_tracking/track_output.mp4
runs/task2_tracking/line_count.mp4
```

遮挡片段截图：

```bash
python -m task2_detection_tracking.extract_occlusion_frames \
  --video runs/task2_tracking/track_output.mp4 \
  --start-frame 180 \
  --num-frames 4
```

## 任务3：U-Net 分割

```text
task3_segmentation/
├── dataset.py    # PetSegDataset
├── models.py     # DoubleConv, UNet, DiceLoss, build_loss
└── train.py      # 训练循环 + mIoU + argparse
```

实现 Diceloss ，完整 Unet。

正式训练：

```bash
python -m task3_segmentation.train --loss all --epochs 50 --batch-size 8 --image-size 256
```

输出文件：

```text
experiments/task3/
├── task3_unet_ce_best.pth
├── task3_unet_dice_best.pth
├── task3_unet_ce_dice_best.pth
└── task3_unet_*_metrics.csv
```

结果（50 epoch，val mIoU）：

| Loss | Train Loss | Train mIoU | **Val mIoU** |
|------|:---:|:---:|:---:|
| CE | 0.1814 | 0.7827 | 0.7395 |
| Dice | 0.1515 | 0.7542 | 0.7440 |
| CE + Dice | 0.3103 | 0.8070 | **0.7578** |

