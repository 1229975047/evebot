# EVE Bot 工业船 CNN 训练指南

## 当前数据集状态

- **已标注样本**: 1 张图片，7 个目标
- **类别**: 3 类
  - `industrial_nerus` (工业涅鲁斯)
  - `industrial_zuotouyu` (工业座头鲸)
  - `industrial_yimisi` (工业依米斯)

## 训练流程

### 步骤 1: 标注更多样本

使用 `annotate_tool.py` 标注更多工业船样本：
- 标注名称需包含：`工业涅鲁斯`、`工业座头鲸`、`工业依米斯`
- 建议每个类别至少 **20-50 张** 不同位置/角度的样本

### 步骤 2: 转换标注数据

```bash
python convert_to_yolo.py
```

这会将 `screenshots/annotations_*.json` 转换为 YOLO 格式：
- 图片保存到 `dataset/images/`
- 标签保存到 `dataset/labels/`
- 配置文件 `dataset/data.yaml`

### 步骤 3: 安装依赖

```bash
pip install ultralytics
```

### 步骤 4: 训练模型

```bash
python train_cnn.py
```

训练参数（可调整）：
- `epochs`: 50（训练轮数，越多越准但越慢）
- `imgsz`: 640（输入图片尺寸）
- `batch`: 16（批大小，显存不够可减小）

### 步骤 5: 测试模型

```bash
# 测试单张图片
python cnn_detector.py --image dataset/images/xxx.png

# 摄像头实时检测
python cnn_detector.py --camera
```

## 模型输出

训练完成后：
- 模型: `runs/detect/industrial_ship_detector/weights/best.pt`
- ONNX: `runs/detect/industrial_ship_detector/weights/best.onnx`

## 集成到刷怪脚本

训练完成后，可以将 `rat_farm_v2.py` 中的 `check_monsters()` 替换为 CNN 检测：

```python
from cnn_detector import CNNEvaluator

class RatFarmV2:
    def __init__(self, ...):
        self.cnn_detector = CNNEvaluator()

    def check_monsters(self, screenshot=None) -> bool:
        if not self.cnn_detector.enabled:
            # 回退到模板匹配
            ...
        found, _ = self.cnn_detector.detect(screenshot)
        return found
```

## 数据增强（可选）

如需扩充样本，可使用数据增强：
- 旋转 (±15°)
- 缩放 (0.8x - 1.2x)
- 亮度调整 (±20%)
- 水平翻转

## 建议

1. **先收集更多样本**：当前只有 7 个目标，建议扩充到 100+ 个
2. **保持样本多样性**：不同距离、角度、光照条件
3. **标注一致性**：同类目标使用相同的标注名称
