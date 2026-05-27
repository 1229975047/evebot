# -*- coding: utf-8 -*-
"""
YOLO训练脚本

功能：
1. 使用ultralytics YOLOv8训练EVE Bot目标检测模型
2. 支持断点续训、数据增强、超参数调优
3. 自动导出推理模型

使用方法：
    python train_yolo.py                     # 使用默认参数训练
    python train_yolo.py --epochs 100        # 指定训练轮数
    python train_yolo.py --resume            # 断点续训
    python train_yolo.py --export            # 仅导出模型
"""

import os
import sys
import argparse
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'yolo_dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'yolo_models')
DATA_YAML = os.path.join(DATASET_DIR, 'data.yaml')


def check_environment():
    """检查训练环境"""
    print("\n" + "=" * 60)
    print("环境检查")
    print("=" * 60)

    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA: {'可用' if torch.cuda.is_available() else '不可用'}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
        else:
            print("  [警告] 未检测到GPU，将使用CPU训练（速度较慢）")
    except ImportError:
        print("  [错误] PyTorch未安装")
        return False

    try:
        import ultralytics
        print(f"  Ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("  [错误] Ultralytics未安装，请运行: pip install ultralytics")
        return False

    if not os.path.exists(DATA_YAML):
        print(f"  [错误] 数据集配置不存在: {DATA_YAML}")
        print(f"  请先运行: python prepare_yolo_dataset.py")
        return False

    print(f"  数据集: {DATA_YAML}")
    print("=" * 60)
    return True


def train(
    epochs: int = 100,
    batch_size: int = 8,
    img_size: int = 640,
    model_size: str = 'n',
    resume: bool = False,
    learning_rate: float = 0.01,
    device: str = None
):
    """
    训练YOLO模型

    Args:
        epochs: 训练轮数
        batch_size: 批次大小
        img_size: 输入图像尺寸
        model_size: 模型大小 ('n', 's', 'm', 'l', 'x')
        resume: 是否断点续训
        learning_rate: 学习率
        device: 训练设备 ('cpu', '0', '0,1' 等)
    """
    from ultralytics import YOLO

    os.makedirs(MODEL_DIR, exist_ok=True)

    if device is None:
        import torch
        device = '0' if torch.cuda.is_available() else 'cpu'

    model_name = f'yolov8{model_size}.pt'
    print(f"\n[训练] 模型: yolov8{model_size}")
    print(f"[训练] 轮数: {epochs}, 批次: {batch_size}, 图像尺寸: {img_size}")
    print(f"[训练] 设备: {device}, 学习率: {learning_rate}")

    model = YOLO(model_name)

    train_args = {
        'data': DATA_YAML,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,
        'project': MODEL_DIR,
        'name': f'eve_detector_{model_size}',
        'exist_ok': True,
        'patience': 30,
        'lr0': learning_rate,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 2.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.1,
        'copy_paste': 0.1,
        'workers': 4,
        'amp': True,
        'save': True,
        'save_period': 10,
        'plots': True,
        'verbose': True,
    }

    if resume:
        last_weights = os.path.join(MODEL_DIR, f'eve_detector_{model_size}', 'weights', 'last.pt')
        if os.path.exists(last_weights):
            print(f"[续训] 从 {last_weights} 继续训练")
            model = YOLO(last_weights)
        else:
            print(f"[警告] 未找到断点权重，从头开始训练")

    results = model.train(**train_args)

    best_weights = os.path.join(MODEL_DIR, f'eve_detector_{model_size}', 'weights', 'best.pt')
    print(f"\n[完成] 最佳权重: {best_weights}")

    return best_weights


def export_model(weights_path: str, formats: list = None):
    """
    导出模型为不同格式

    Args:
        weights_path: 权重文件路径
        formats: 导出格式列表，默认 ['onnx']
    """
    from ultralytics import YOLO

    if not os.path.exists(weights_path):
        print(f"[错误] 权重文件不存在: {weights_path}")
        return

    if formats is None:
        formats = ['onnx']

    model = YOLO(weights_path)
    for fmt in formats:
        try:
            print(f"[导出] 格式: {fmt}")
            model.export(format=fmt)
            print(f"[导出] 完成: {fmt}")
        except Exception as e:
            print(f"[导出] 失败 {fmt}: {e}")


def evaluate(weights_path: str):
    """评估模型性能"""
    from ultralytics import YOLO

    if not os.path.exists(weights_path):
        print(f"[错误] 权重文件不存在: {weights_path}")
        return

    model = YOLO(weights_path)
    results = model.val(data=DATA_YAML)

    print("\n" + "=" * 60)
    print("模型评估结果")
    print("=" * 60)
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  精确率: {results.box.mp:.4f}")
    print(f"  召回率: {results.box.mr:.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='YOLO训练脚本')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch', type=int, default=8, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--model', choices=['n', 's', 'm', 'l', 'x'], default='s',
                        help='模型大小 (n=最小最快, x=最大最准)')
    parser.add_argument('--resume', action='store_true', help='断点续训')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--device', type=str, default=None, help='设备 (cpu, 0, 0,1)')
    parser.add_argument('--export', type=str, default=None, help='导出模型路径')
    parser.add_argument('--eval', type=str, default=None, help='评估模型路径')
    args = parser.parse_args()

    if args.export:
        export_model(args.export)
        return

    if args.eval:
        evaluate(args.eval)
        return

    if not check_environment():
        print("\n环境检查失败，请解决上述问题后重试")
        sys.exit(1)

    print(f"\n开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    best_weights = train(
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        model_size=args.model,
        resume=args.resume,
        learning_rate=args.lr,
        device=args.device
    )

    if best_weights and os.path.exists(best_weights):
        print("\n[导出] 自动导出ONNX格式...")
        export_model(best_weights, ['onnx'])

        target = os.path.join(MODEL_DIR, 'eve_detector.pt')
        import shutil
        shutil.copy2(best_weights, target)
        print(f"[完成] 模型已复制到: {target}")

    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
